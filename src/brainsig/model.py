"""
A module to fit elastic net logistic regression.

This module implements the ElasticNetClassifier for neural signature analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

class ElasticNetClassifier:
    """
    Elastic Net Logistic Regression classifier for neural signature analysis.

    This classifier performs nested cross-validation with elastic net regularization
    for binary or multi-class classification tasks.

    Parameters
    ----------
    inner_folds : int, default=5
        Number of folds for inner cross-validation (hyperparameter tuning).
    outer_folds : int, default=5
        Number of folds for outer cross-validation (performance evaluation).
    inner_scoring : str, default='roc_auc_ovr'
        Scoring metric for inner CV hyperparameter selection.
    outer_scoring : dict or None, default=None
        Dictionary of scoring metrics for outer CV. If None, uses default metrics.
    cs : list or None, default=None
        Regularization parameter values to test. If None, uses default values.
    l1_ratios : list or None, default=None
        L1 penalty ratios for elastic net. If None, uses default values.
    max_iter : int, default=1000
        Maximum number of iterations for solver convergence.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all processors.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    models : dict
        Fitted models for each target variable.
    cv_results : dict
        Cross-validation results for each target variable.
    target_names : list
        Names of target variables.
    """

    def __init__(
        self,
        inner_folds: int = 5,
        outer_folds: int = 5,
        inner_scoring: str = "roc_auc_ovr",
        outer_scoring: dict | None = None,
        cs: list | None = None,
        l1_ratios: list | None = None,
        max_iter: int = 1000,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.inner_scoring = inner_scoring
        self.outer_scoring = outer_scoring or {
            "acc": "accuracy",
            "f1": "f1_macro",
            "auc": "roc_auc_ovr",
        }
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.Cs = cs or [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        self.l1_ratios = l1_ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.inner_cv = self.build_cv_scheme(self.inner_folds, self.random_state)
        self.outer_cv = self.build_cv_scheme(
            self.outer_folds,
            self.random_state + 50,
        )
        self.dataset = None

        # Initialize containers for multiple targets
        self.models = {}
        self.cv_results = {}
        self.target_names = []

    def define_model(self, random_state: int = 42) -> LogisticRegressionCV:
        """
        Create a LogisticRegressionCV model with elastic net penalty.

        Parameters
        ----------
        random_state : int, default=42
            Random seed for model initialization.

        Returns
        -------
        LogisticRegressionCV
            Configured logistic regression model with cross-validation.
        """
        return LogisticRegressionCV(
            cv=self.inner_cv,
            penalty="elasticnet",
            solver="saga",
            scoring=self.inner_scoring,
            Cs=self.Cs,
            l1_ratios=self.l1_ratios,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=random_state,
        )

    def build_cv_scheme(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> StratifiedKFold:
        """
        Build a stratified k-fold cross-validation scheme.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds for cross-validation.
        random_state : int, default=42
            Random seed for reproducible splits.

        Returns
        -------
        StratifiedKFold
            Configured cross-validation splitter.
        """
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    def fit_model(self, dataset, *, keep_dataset: bool = True) -> None:
        """
        Fit elastic net models for each target variable.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing X_train and y_train arrays.
        keep_dataset : bool, default=True
            Whether to store the dataset as an instance attribute.
        """
        self.target_names = list(dataset.target_labels.keys())
        n_targets = len(self.target_names)

        if keep_dataset:
            self.dataset = dataset

        for i, target_name in enumerate(self.target_names):
            y_target = (
                dataset.y_train[:, i] if n_targets > 1 else dataset.y_train.ravel()
            )

            model = self.define_model(self.random_state + 100)
            model.fit(dataset.X_train, y_target)
            self.models[target_name] = model

    def predict(self, dataset) -> dict:
        """
        Make predictions on test set for all targets.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing X_test arrays.

        Returns
        -------
        dict
            Dictionary with predictions for each target, containing y_pred,
            y_pred_proba, and y_true arrays.
        """
        self.predictions = {}

        for target_name, model in self.models.items():
            y_pred = model.predict(dataset.X_test)
            y_pred_proba = (
                model.predict_proba(dataset.X_test)
                if hasattr(model, "predict_proba")
                else None
            )

            self.predictions[target_name] = {
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "y_true": dataset.y_test[
                    :,
                    list(dataset.target_labels.keys()).index(target_name),
                ]
                if len(dataset.target_labels) > 1
                else dataset.y_test.ravel(),
            }

        return self.predictions

    def cross_validate(self, dataset) -> dict:
        """
        Perform nested cross-validation for each target variable.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing training data.

        Returns
        -------
        dict
            Cross-validation results for each target, including scores
            and fitted estimators for each fold.
        """
        self.target_names = list(dataset.target_labels.keys())
        n_targets = len(self.target_names)

        for i, target_name in enumerate(self.target_names):
            y_target = (
                dataset.y_train[:, i] if n_targets > 1 else dataset.y_train.ravel()
            )

            model = self.define_model()
            cv_result = cross_validate(
                model,
                dataset.X_train,
                y_target,
                cv=self.outer_cv,
                scoring=self.outer_scoring,
                return_indices=True,
                return_estimator=True,
                return_train_score=True,
            )
            self.cv_results[target_name] = cv_result

        return self.cv_results

    def get_cv_coefs(self, dataset, *, exponentiate: bool = False) -> pd.DataFrame:
        """
        Extract coefficients from cross-validated models.

        Parameters
        ----------
        dataset : Dataset
            Dataset object with feature names and target labels.
        exponentiate : bool, default=False
            If True, exponentiate coefficients to get odds ratios.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficients indexed by cv_fold, target_variable,
            and target class.
        """
        all_coefs = pd.DataFrame()

        # Clean feature names more carefully
        clean_feature_names = []
        for name in dataset.feature_names:
            if name.startswith("num__"):
                clean_feature_names.append(name.replace("num__", ""))
            elif name.startswith("cat__"):
                clean_feature_names.append(name.replace("cat__", ""))
            else:
                clean_feature_names.append(name)

        for target_name, cv_result in self.cv_results.items():
            models = cv_result["estimator"]
            target_labels = dataset.target_labels[target_name]

            # If target_labels is None (numeric target), get unique classes from data
            if target_labels is None:
                # Get unique values from training data
                target_idx = list(dataset.target_labels.keys()).index(target_name)
                y_train = (
                    dataset.y_train[:, target_idx]
                    if len(dataset.target_labels) > 1
                    else dataset.y_train.ravel()
                )
                target_labels = sorted(np.unique(y_train))

            for i, model in enumerate(models):
                coef_matrix = model.coef_

                # For binary classification, coef_matrix has shape (1, n_features)
                # For multiclass, it has shape (n_classes, n_features)
                if coef_matrix.shape[0] == 1:
                    # Binary classification - single coefficient vector
                    coef_df = pd.DataFrame(
                        [coef_matrix[0]], columns=clean_feature_names,
                    )
                    target_label = (
                        target_labels[1] if len(target_labels) > 1 else target_labels[0]
                    )
                    coef_df = coef_df.assign(
                        cv_fold=i,
                        target_variable=target_name,
                        target=target_label,
                    ).set_index(["cv_fold", "target_variable", "target"])
                    all_coefs = pd.concat([all_coefs, coef_df])
                else:
                    # Multiclass - one coefficient vector per class
                    for class_idx, class_label in enumerate(target_labels):
                        coef_row = coef_matrix[class_idx]

                        coef_df = pd.DataFrame([coef_row], columns=clean_feature_names)
                        coef_df = coef_df.assign(
                            cv_fold=i,
                            target_variable=target_name,
                            target=class_label,
                        ).set_index(["cv_fold", "target_variable", "target"])

                        all_coefs = pd.concat([all_coefs, coef_df])

        if exponentiate:
            numeric_cols = all_coefs.select_dtypes(include=[np.number]).columns
            all_coefs[numeric_cols] = np.exp(all_coefs[numeric_cols])

        return all_coefs

    def get_model_scores(self, dataset=None) -> pd.DataFrame:
        """
        Get all available model scores from CV and/or fitted models.

        Parameters
        ----------
        dataset : Dataset or None, default=None
            Dataset object with target labels for score interpretation.

        Returns
        -------
        pd.DataFrame
            DataFrame with scores from cross-validation and/or fitted models,
            including accuracy, F1, and AUC metrics per class.
        """
        all_scores = pd.DataFrame()

        # Add CV scores if available
        if hasattr(self, "cv_results") and self.cv_results:
            for target_name, cv_result in self.cv_results.items():
                # Get the class labels for this target from dataset
                target_labels = None
                if dataset and hasattr(dataset, "target_labels"):
                    target_labels = dataset.target_labels.get(target_name)

                for k, v in cv_result.items():
                    splits = k.split("_")
                    if splits[0] in ["train", "test"]:
                        # If this is a multi-class problem, scores are per class
                        if target_labels and len(target_labels) > 2:  # Multi-class
                            for i, class_label in enumerate(target_labels):
                                cv_scores = (
                                    pd.DataFrame(
                                        {"value": v[:, i] if v.ndim > 1 else v},
                                    )
                                    .assign(
                                        partition=splits[0],
                                        metric=splits[1],
                                        target=target_name,
                                        score_type="cv",
                                        target_label=class_label,
                                    )
                                    .reset_index(names=["cv_fold"])
                                )
                                all_scores = pd.concat([all_scores, cv_scores])
                        else:  # Binary or unknown classification
                            cv_scores = (
                                pd.DataFrame({"value": v})
                                .assign(
                                    partition=splits[0],
                                    metric=splits[1],
                                    target=target_name,
                                    score_type="cv",
                                    target_label=None,
                                )
                                .reset_index(names=["cv_fold"])
                            )
                            all_scores = pd.concat([all_scores, cv_scores])

            # Add fitted model scores if available (rest remains the same)
            if hasattr(self, "predictions") and self.predictions:
                for target_name, pred_data in self.predictions.items():
                    y_true, y_pred = pred_data["y_true"], pred_data["y_pred"]
                    y_pred_proba = pred_data["y_pred_proba"]

                    unique_classes = np.unique(y_true)
                    f1_per_class = f1_score(y_true, y_pred, average=None)

                    if y_pred_proba is not None:
                        auc_per_class = roc_auc_score(
                            y_true,
                            y_pred_proba,
                            multi_class="ovr",
                            average=None,
                        )

                    acc_score = accuracy_score(y_true, y_pred)

                    for i, class_label in enumerate(unique_classes):
                        class_scores = {"acc": acc_score, "f1": f1_per_class[i]}

                        if y_pred_proba is not None:
                            class_scores["auc"] = auc_per_class[i]

                        for metric, score in class_scores.items():
                            fitted_scores = pd.DataFrame(
                                {
                                    "cv_fold": [0],
                                    "value": [score],
                                    "partition": ["test_holdout"],
                                    "metric": [metric],
                                    "target": [target_name],
                                    "target_label": [class_label],
                                    "score_type": ["fitted"],
                                },
                            )
                            all_scores = pd.concat([all_scores, fitted_scores])

        return all_scores


class NeuralSignature:
    """
    Neural Signature classifier for fMRI task condition discrimination.

    This class fits an elastic net logistic regression model to discriminate between
    two fMRI task conditions (labeled 1 and 0) and computes neural signature scores
    as the difference in predicted probabilities between conditions for each subject.

    The neural signature score for a subject is computed as:
        score = P(condition=1 | fMRI_condition1) - P(condition=1 | fMRI_condition0)

    Parameters
    ----------
    inner_folds : int, default=5
        Number of folds for inner cross-validation (hyperparameter tuning).
    outer_folds : int, default=5
        Number of folds for outer cross-validation (performance evaluation).
    inner_scoring : str, default='roc_auc'
        Scoring metric for inner CV hyperparameter selection.
    outer_scoring : dict or None, default=None
        Dictionary of scoring metrics for outer CV. If None, uses default metrics.
    cs : list or None, default=None
        Regularization parameter values to test. If None, uses default values.
    l1_ratios : list or None, default=None
        L1 penalty ratios for elastic net. If None, uses default values.
    max_iter : int, default=1000
        Maximum number of iterations for solver convergence.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all processors.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classifier : ElasticNetClassifier
        Underlying elastic net classifier for condition discrimination.
    signature_scores : pd.DataFrame or None
        Computed neural signature scores for each subject.

    Examples
    --------
    >>> # Prepare data with condition labels (1 and 0)
    >>> neural_sig = NeuralSignature(random_state=42)
    >>> neural_sig.fit(dataset)
    >>> scores = neural_sig.compute_signature_scores(condition1_data, condition0_data)
    """

    def __init__(
        self,
        inner_folds: int = 5,
        outer_folds: int = 5,
        inner_scoring: str = "roc_auc",
        outer_scoring: dict | None = None,
        cs: list | None = None,
        l1_ratios: list | None = None,
        max_iter: int = 1000,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        # Use binary roc_auc for neural signature (2 conditions)
        if outer_scoring is None:
            outer_scoring = {
                "acc": "accuracy",
                "f1": "f1",
                "auc": "roc_auc",
            }

        self.classifier = ElasticNetClassifier(
            inner_folds=inner_folds,
            outer_folds=outer_folds,
            inner_scoring=inner_scoring,
            outer_scoring=outer_scoring,
            cs=cs,
            l1_ratios=l1_ratios,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.signature_scores = None

    def fit(self, dataset, *, keep_dataset: bool = True) -> None:
        """
        Fit the neural signature model to discriminate between task conditions.

        The dataset should contain a binary target variable where:
        - Label 1 represents the first task condition
        - Label 0 represents the second task condition

        Parameters
        ----------
        dataset : Dataset
            Dataset object with binary condition labels (1 and 0).
        keep_dataset : bool, default=True
            Whether to store the dataset in the classifier.
        """
        self.classifier.fit_model(dataset, keep_dataset=keep_dataset)

    def cross_validate(self, dataset) -> dict:
        """
        Perform nested cross-validation for the neural signature model.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing training data with binary condition labels.

        Returns
        -------
        dict
            Cross-validation results including scores and fitted estimators.
        """
        return self.classifier.cross_validate(dataset)

    def compute_signature_scores(
        self,
        condition1_data: np.ndarray,
        condition0_data: np.ndarray,
        *,
        subject_ids: list | None = None,
    ) -> pd.DataFrame:
        """
        Compute neural signature scores for each subject.

        The neural signature score is computed as the difference in predicted
        probabilities for condition 1 between the two task conditions:
            score = P(y=1 | condition1_data) - P(y=1 | condition0_data)

        Parameters
        ----------
        condition1_data : np.ndarray
            Preprocessed fMRI data for condition 1 (shape: n_subjects x n_features).
        condition0_data : np.ndarray
            Preprocessed fMRI data for condition 0 (shape: n_subjects x n_features).
        subject_ids : list or None, default=None
            Optional list of subject identifiers. If None, uses sequential indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: subject_id, condition1_prob, condition0_prob,
            signature_score.

        Raises
        ------
        ValueError
            If the model hasn't been fitted yet or if data shapes don't match.
        """
        if not self.classifier.models:
            msg = (
                "Model must be fitted before computing signature scores. "
                "Call fit() first."
            )
            raise ValueError(msg)

        if condition1_data.shape != condition0_data.shape:
            msg = (
                f"Data shapes must match. Got condition1: {condition1_data.shape}, "
                f"condition0: {condition0_data.shape}"
            )
            raise ValueError(msg)

        n_subjects = condition1_data.shape[0]

        if subject_ids is None:
            subject_ids = list(range(n_subjects))
        elif len(subject_ids) != n_subjects:
            msg = (
                f"Number of subject_ids ({len(subject_ids)}) must match "
                f"number of subjects ({n_subjects})"
            )
            raise ValueError(msg)

        # Get the model (assuming single target for binary classification)
        target_name = next(iter(self.classifier.models.keys()))
        model = self.classifier.models[target_name]

        # Get prediction probabilities for both conditions
        # For binary classification, predict_proba returns [P(y=0), P(y=1)]
        condition1_probs = model.predict_proba(condition1_data)[:, 1]  # P(y=1 | cond1)
        condition0_probs = model.predict_proba(condition0_data)[:, 1]  # P(y=1 | cond0)

        # Compute signature score
        signature_scores = condition1_probs - condition0_probs

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "condition1_prob": condition1_probs,
                "condition0_prob": condition0_probs,
                "signature_score": signature_scores,
            },
        )

        self.signature_scores = results
        return results

    def get_cv_signature_scores(
        self,
        dataset,
        condition1_indices: np.ndarray,
        condition0_indices: np.ndarray,
        *,
        subject_ids: list | None = None,
    ) -> pd.DataFrame:
        """
        Compute neural signature scores using cross-validated models.

        This method computes signature scores for each CV fold, which is useful
        for estimating the generalizability of the neural signature.

        Parameters
        ----------
        dataset : Dataset
            Dataset object with full data (both conditions for all subjects).
        condition1_indices : np.ndarray
            Indices in the dataset corresponding to condition 1 trials.
        condition0_indices : np.ndarray
            Indices in the dataset corresponding to condition 0 trials.
        subject_ids : list or None, default=None
            Optional list of subject identifiers.

        Returns
        -------
        pd.DataFrame
            DataFrame with signature scores from each CV fold, including columns:
            cv_fold, subject_id, condition1_prob, condition0_prob, signature_score.

        Raises
        ------
        ValueError
            If cross-validation hasn't been performed yet.
        """
        if not self.classifier.cv_results:
            msg = "Must run cross_validate() before computing CV signature scores."
            raise ValueError(msg)

        all_scores = []

        # Get CV results for the target
        target_name = next(iter(self.classifier.cv_results.keys()))
        cv_result = self.classifier.cv_results[target_name]
        estimators = cv_result["estimator"]

        # For each fold
        for fold_idx, estimator in enumerate(estimators):
            # Get test indices for this fold
            test_indices = cv_result["indices"]["test"][fold_idx]

            # Find which test samples correspond to each condition
            cond1_test = np.intersect1d(test_indices, condition1_indices)
            cond0_test = np.intersect1d(test_indices, condition0_indices)

            if len(cond1_test) > 0 and len(cond0_test) > 0:
                # Get data for this fold
                X_cond1 = dataset.X_train[cond1_test]
                X_cond0 = dataset.X_train[cond0_test]

                # Predict probabilities
                cond1_probs = estimator.predict_proba(X_cond1)[:, 1]
                cond0_probs = estimator.predict_proba(X_cond0)[:, 1]

                # Compute signature scores
                # Note: This assumes paired data (same subjects in both conditions)
                min_len = min(len(cond1_probs), len(cond0_probs))
                sig_scores = cond1_probs[:min_len] - cond0_probs[:min_len]

                # Create fold results
                fold_subject_ids = (
                    subject_ids[:min_len] if subject_ids else list(range(min_len))
                )

                fold_df = pd.DataFrame(
                    {
                        "cv_fold": fold_idx,
                        "subject_id": fold_subject_ids,
                        "condition1_prob": cond1_probs[:min_len],
                        "condition0_prob": cond0_probs[:min_len],
                        "signature_score": sig_scores,
                    },
                )
                all_scores.append(fold_df)

        if all_scores:
            return pd.concat(all_scores, ignore_index=True)
        return pd.DataFrame()

    def get_coefficients(self, dataset, *, exponentiate: bool = False) -> pd.DataFrame:
        """
        Get model coefficients (feature weights) from cross-validated models.

        Parameters
        ----------
        dataset : Dataset
            Dataset object with feature names.
        exponentiate : bool, default=False
            If True, exponentiate coefficients to get odds ratios.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficients for each feature across CV folds.
        """
        return self.classifier.get_cv_coefs(dataset, exponentiate=exponentiate)

    def get_model_scores(self, dataset=None) -> pd.DataFrame:
        """
        Get classification performance scores.

        Parameters
        ----------
        dataset : Dataset or None, default=None
            Dataset object with target labels.

        Returns
        -------
        pd.DataFrame
            DataFrame with classification accuracy, F1, and AUC scores.
        """
        return self.classifier.get_model_scores(dataset)
