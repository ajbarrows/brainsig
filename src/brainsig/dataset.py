"""
A module that creates the dataset object.

This module provides the Dataset class for preprocessing and preparing data
for machine learning tasks, including handling missing data, feature preprocessing,
and train/test splitting.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class Dataset:
    """
    A class for preprocessing and preparing datasets for machine learning.

    This class handles missing data, feature preprocessing (scaling and encoding),
    and train/test splitting. It's designed to work with pandas DataFrames and
    integrates with scikit-learn pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target variables.
    target : str or list of str
        Name(s) of the target variable column(s).
    missing_threshold : float, default=0.5
        Threshold for dropping columns with missing data. Columns with a fraction
        of missing values greater than this threshold will be dropped.
    preprocessor : sklearn.compose.ColumnTransformer or None, default=None
        Custom preprocessor for features. If None, a default preprocessor is created
        that standardizes numeric features and one-hot encodes categorical features.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split. If 0, no train/test
        split is performed and all data is stored in `X` and `y` attributes.
    random_state : int or None, default=None
        Random seed for reproducibility of train/test split.
    verbose : bool, default=True
        If True, print information about dropped columns and rows.

    Attributes
    ----------
    original_df : pd.DataFrame
        Copy of the original input DataFrame.
    target : str or list of str
        Name(s) of target variable(s).
    dropped_summary : dict
        Summary of dropped data with keys 'all_missing_cols', 'high_missing_cols',
        and 'rows_dropped'.
    preprocessor : sklearn.compose.ColumnTransformer
        The fitted preprocessor used for feature transformation.
    feature_names : np.ndarray
        Names of features after preprocessing.
    target_labels : dict
        Dictionary mapping target column names to their unique class labels.
    X_train : np.ndarray
        Training features (only if test_size > 0).
    X_test : np.ndarray
        Test features (only if test_size > 0).
    y_train : np.ndarray
        Training targets (only if test_size > 0).
    y_test : np.ndarray
        Test targets (only if test_size > 0).
    X : np.ndarray
        All features (only if test_size = 0).
    y : np.ndarray
        All targets (only if test_size = 0).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'income': [50000, 60000, 70000, 80000],
    ...     'outcome': ['A', 'B', 'A', 'B']
    ... })
    >>> dataset = Dataset(df, target='outcome', test_size=0.25, random_state=42)
    >>> print(dataset.X_train.shape)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | list[str],
        missing_threshold: float = 0.5,
        preprocessor: ColumnTransformer | None = None,
        test_size: float = 0.2,
        random_state: int | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        self.original_df = df.copy()
        self.target = target

        # Clean missing data
        cleaned_df, self.dropped_summary = self._drop_missing_data(
            df,
            missing_threshold,
            verbose=verbose,
        )

        # Create sklearn dataset
        self._create_sklearn_dataset(
            cleaned_df,
            target,
            preprocessor,
            test_size,
            random_state,
        )

    def _drop_missing_data(
        self,
        df: pd.DataFrame,
        missing_threshold: float,
        *,
        verbose: bool,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Remove columns and rows with missing data based on threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to clean.
        missing_threshold : float
            Threshold for dropping columns. Columns with missing values exceeding
            this proportion will be removed.
        verbose : bool
            Whether to print information about dropped data.

        Returns
        -------
        df_clean : pd.DataFrame
            Cleaned DataFrame with missing data removed.
        dropped_summary : dict
            Dictionary containing information about dropped columns and rows with keys:
            - 'all_missing_cols': List of columns that were entirely missing
            - 'high_missing_cols': List of columns above the missing threshold
            - 'rows_dropped': Number of rows removed
        """
        original_shape = df.shape
        dropped_summary = {
            "all_missing_cols": [],
            "high_missing_cols": [],
            "rows_dropped": 0,
        }

        # Drop completely missing columns
        all_missing = df.columns[df.isna().all()]
        df_clean = df.drop(columns=all_missing)
        dropped_summary["all_missing_cols"] = list(all_missing)

        # Drop columns above threshold
        missing_pct = df_clean.isna().mean()
        high_missing = missing_pct[missing_pct > missing_threshold].index
        df_clean = df_clean.drop(columns=high_missing)
        dropped_summary["high_missing_cols"] = list(high_missing)

        # Drop rows with missing values
        rows_before = len(df_clean)
        df_clean = df_clean.dropna()
        dropped_summary["rows_dropped"] = rows_before - len(df_clean)

        if verbose:
            logger.info("Original shape: %s", original_shape)
            logger.info("Final shape: %s", df_clean.shape)
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
            logger.info("Rows dropped: %s", dropped_summary["rows_dropped"])

        return df_clean, dropped_summary

    def _create_sklearn_dataset(
        self,
        df: pd.DataFrame,
        target: str | list[str],
        preprocessor: ColumnTransformer | None,
        test_size: float,
        random_state: int | None,
    ) -> None:
        """
        Create scikit-learn compatible dataset with preprocessing and splitting.

        This method handles target extraction, feature preprocessing (standardization
        for numeric features and one-hot encoding for categorical features), and
        optional train/test splitting.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame to process.
        target : str or list of str
            Name(s) of target column(s).
        preprocessor : ColumnTransformer or None
            Custom preprocessor. If None, creates default preprocessor.
        test_size : float
            Proportion of data to use for testing.
        random_state : int or None
            Random seed for train/test split.

        Returns
        -------
        None
            Sets instance attributes: preprocessor, feature_names, target_labels,
            and either (X_train, X_test, y_train, y_test) or (X, y).
        """
        # Handle targets
        targets = [target] if isinstance(target, str) else list(target)

        X = df.drop(columns=targets)
        y = df[targets]

        # Extract target labels for categorical targets (preserve pd.Categorical order)
        self.target_labels = {}
        for col in targets:
            if df[col].dtype.name == "category":
                # Use the explicit categories you set with pd.Categorical
                self.target_labels[col] = df[col].cat.categories.tolist()
            elif df[col].dtype == "object":
                self.target_labels[col] = df[col].unique().tolist()
            else:
                self.target_labels[col] = None

        # Default preprocessor
        if preprocessor is None:
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(exclude=[np.number]).columns

            # Preserve categorical order for all categorical columns
            categorical_categories = []
            for col in categorical_features:
                if X[col].dtype.name == "category":
                    categorical_categories.append(X[col].cat.categories.tolist())
                else:
                    categorical_categories.append(sorted(X[col].unique()))

            preprocessor = ColumnTransformer(
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

        self.preprocessor = preprocessor
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # Train/test split
        if test_size > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed,
                y.to_numpy(),
                test_size=test_size,
                random_state=random_state,
            )
        else:
            self.X = X_processed
            self.y = y.to_numpy()
