"""
A module for creating paired-condition neural signature datasets.

This module provides the NeuralSignatureDataset class for preparing paired
condition data for neural signature analysis, with subject-level train/test
splitting to prevent data leakage.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class NeuralSignatureDataset:
    """
    A dataset for paired-condition neural signature analysis.

    Accepts two DataFrames (one per condition) with matched subjects, performs
    subject-level train/test splitting to prevent leakage, and exposes
    preprocessed arrays ready for ``NeuralSignature.fit`` and
    ``compute_signature_scores``.

    Parameters
    ----------
    condition1_df : pd.DataFrame
        DataFrame for condition 1 (positive class). Rows are subjects,
        columns are features.
    condition0_df : pd.DataFrame
        DataFrame for condition 0 (negative class). Must have the same shape
        and columns as *condition1_df*.
    subject_id_col : str or None, default=None
        Column name containing subject identifiers. If provided, the column
        is extracted and removed from features. Subject IDs must match between
        the two DataFrames. If None, integer indices are used.
    missing_threshold : float, default=0.5
        Columns with a fraction of missing values exceeding this threshold
        are dropped.
    preprocessor : sklearn.compose.ColumnTransformer or None, default=None
        Custom preprocessor. If None, a default preprocessor is created that
        standardizes numeric features and one-hot encodes categorical features.
    test_size : float, default=0.2
        Proportion of subjects to include in the test split.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, log information about dropped columns and rows.

    Attributes
    ----------
    subject_ids : np.ndarray
        All subject IDs after cleaning, matching row order of condition arrays.
    condition1 : np.ndarray
        Preprocessed condition-1 data for all subjects, shape ``(N, F)``.
    condition0 : np.ndarray
        Preprocessed condition-0 data for all subjects, shape ``(N, F)``.
    X_train : np.ndarray
        Combined training features (condition-1 rows then condition-0 rows),
        shape ``(2*N_train, F)``.
    X_test : np.ndarray
        Combined test features, shape ``(2*N_test, F)``.
    y_train : np.ndarray
        Binary labels for training data: 1s then 0s.
    y_test : np.ndarray
        Binary labels for test data: 1s then 0s.
    feature_names : np.ndarray
        Feature names from ``preprocessor.get_feature_names_out()``.
    target_labels : dict
        ``{"condition": [0, 1]}``.
    preprocessor : sklearn.compose.ColumnTransformer
        The fitted preprocessor (fit on training data only).
    dropped_summary : dict
        Summary of dropped data with keys ``all_missing_cols``,
        ``high_missing_cols``, and ``subjects_dropped``.
    """

    def __init__(
        self,
        condition1_df: pd.DataFrame,
        condition0_df: pd.DataFrame,
        subject_id_col: str | None = None,
        missing_threshold: float = 0.5,
        preprocessor: ColumnTransformer | None = None,
        test_size: float = 0.2,
        random_state: int | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        # --- Validate shape ---
        if condition1_df.shape[0] != condition0_df.shape[0]:
            msg = (
                f"condition1_df and condition0_df must have the same number "
                f"of rows, got {condition1_df.shape[0]} and "
                f"{condition0_df.shape[0]}"
            )
            raise ValueError(msg)

        condition1_df = condition1_df.copy()
        condition0_df = condition0_df.copy()

        # --- Extract subject IDs ---
        if subject_id_col is not None:
            if subject_id_col not in condition1_df.columns:
                msg = f"subject_id_col '{subject_id_col}' not found in condition1_df"
                raise ValueError(msg)
            if subject_id_col not in condition0_df.columns:
                msg = f"subject_id_col '{subject_id_col}' not found in condition0_df"
                raise ValueError(msg)

            ids1 = condition1_df[subject_id_col].to_numpy()
            ids0 = condition0_df[subject_id_col].to_numpy()

            if not np.array_equal(ids1, ids0):
                msg = (
                    "Subject IDs do not match between condition1_df and "
                    "condition0_df"
                )
                raise ValueError(msg)

            subject_ids = ids1
            condition1_df = condition1_df.drop(columns=[subject_id_col])
            condition0_df = condition0_df.drop(columns=[subject_id_col])
        else:
            subject_ids = np.arange(condition1_df.shape[0])

        # --- Handle missing data ---
        condition1_df, condition0_df, subject_ids, dropped_summary = (
            self._handle_missing_data(
                condition1_df,
                condition0_df,
                subject_ids,
                missing_threshold,
                verbose=verbose,
            )
        )
        self.dropped_summary = dropped_summary
        self.subject_ids = subject_ids

        # --- Subject-level split ---
        n_subjects = len(subject_ids)
        subject_indices = np.arange(n_subjects)

        train_idx, test_idx = train_test_split(
            subject_indices,
            test_size=test_size,
            random_state=random_state,
        )

        cond1_train = condition1_df.iloc[train_idx]
        cond1_test = condition1_df.iloc[test_idx]
        cond0_train = condition0_df.iloc[train_idx]
        cond0_test = condition0_df.iloc[test_idx]

        # --- Build combined train/test DataFrames ---
        X_train_df = pd.concat([cond1_train, cond0_train], ignore_index=True)
        X_test_df = pd.concat([cond1_test, cond0_test], ignore_index=True)

        n_train = len(train_idx)
        n_test = len(test_idx)
        self.y_train = np.array([1] * n_train + [0] * n_train)
        self.y_test = np.array([1] * n_test + [0] * n_test)

        # --- Build / fit preprocessor ---
        if preprocessor is None:
            preprocessor = self._build_default_preprocessor(X_train_df)

        self.preprocessor = preprocessor
        self.X_train = self.preprocessor.fit_transform(X_train_df)
        self.X_test = self.preprocessor.transform(X_test_df)

        # --- Transform full condition arrays ---
        self.condition1 = self.preprocessor.transform(condition1_df)
        self.condition0 = self.preprocessor.transform(condition0_df)

        # --- Metadata ---
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.target_labels = {"condition": [0, 1]}

    @staticmethod
    def _build_default_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        """
        Build the default preprocessor matching Dataset conventions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame used to identify numeric and categorical columns.

        Returns
        -------
        preprocessor : sklearn.compose.ColumnTransformer
            An unfitted ColumnTransformer.
        """
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(exclude=[np.number]).columns

        categorical_categories = []
        for col in categorical_features:
            if X[col].dtype.name == "category":
                categorical_categories.append(X[col].cat.categories.tolist())
            else:
                categorical_categories.append(sorted(X[col].unique()))

        return ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        categories=categorical_categories,
                        handle_unknown="ignore",
                    ),
                    categorical_features,
                ),
            ],
        )

    @staticmethod
    def _handle_missing_data(
        cond1: pd.DataFrame,
        cond0: pd.DataFrame,
        subject_ids: np.ndarray,
        missing_threshold: float,
        *,
        verbose: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
        """
        Handle missing data for paired condition DataFrames.

        Drops columns that are entirely missing or exceed the missing threshold
        (computed on the combined data), then drops subjects where either
        condition has remaining NaNs to keep pairing intact.

        Parameters
        ----------
        cond1 : pd.DataFrame
            Condition 1 features.
        cond0 : pd.DataFrame
            Condition 0 features.
        subject_ids : np.ndarray
            Subject identifiers.
        missing_threshold : float
            Threshold for dropping columns.
        verbose : bool
            Whether to log information.

        Returns
        -------
        cond1_clean : pd.DataFrame
            Cleaned condition 1 DataFrame.
        cond0_clean : pd.DataFrame
            Cleaned condition 0 DataFrame.
        subject_ids_clean : np.ndarray
            Subject IDs after dropping subjects with missing data.
        dropped_summary : dict
            Summary with keys ``all_missing_cols``, ``high_missing_cols``,
            and ``subjects_dropped``.
        """
        original_n_subjects = len(subject_ids)
        dropped_summary = {
            "all_missing_cols": [],
            "high_missing_cols": [],
            "subjects_dropped": 0,
        }

        # Compute missingness on combined data
        combined = pd.concat([cond1, cond0], ignore_index=True)

        # Drop completely missing columns
        all_missing = combined.columns[combined.isna().all()]
        cond1 = cond1.drop(columns=all_missing)
        cond0 = cond0.drop(columns=all_missing)
        dropped_summary["all_missing_cols"] = list(all_missing)

        # Drop columns above threshold
        combined = pd.concat([cond1, cond0], ignore_index=True)
        missing_pct = combined.isna().mean()
        high_missing = missing_pct[missing_pct > missing_threshold].index
        cond1 = cond1.drop(columns=high_missing)
        cond0 = cond0.drop(columns=high_missing)
        dropped_summary["high_missing_cols"] = list(high_missing)

        # Drop subjects where either condition has NaNs
        has_nan = cond1.isna().any(axis=1) | cond0.isna().any(axis=1)
        keep_mask = ~has_nan.to_numpy()

        cond1 = cond1.loc[keep_mask].reset_index(drop=True)
        cond0 = cond0.loc[keep_mask].reset_index(drop=True)
        subject_ids = subject_ids[keep_mask]
        dropped_summary["subjects_dropped"] = int(
            original_n_subjects - keep_mask.sum(),
        )

        if verbose:
            logger.info(
                "Original subjects: %d, after cleaning: %d",
                original_n_subjects,
                len(subject_ids),
            )
            if dropped_summary["all_missing_cols"]:
                logger.info(
                    "All-missing columns dropped: %s",
                    dropped_summary["all_missing_cols"],
                )
            if dropped_summary["high_missing_cols"]:
                logger.info(
                    "High-missing columns dropped: %s",
                    dropped_summary["high_missing_cols"],
                )
            if dropped_summary["subjects_dropped"] > 0:
                logger.info(
                    "Subjects dropped: %d",
                    dropped_summary["subjects_dropped"],
                )

        return cond1, cond0, subject_ids, dropped_summary
