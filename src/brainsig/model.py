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

            for i, model in enumerate(models):
                coef_matrix = model.coef_

                for class_idx, class_label in enumerate(target_labels):
                    coef_row = (
                        coef_matrix[class_idx] if coef_matrix.ndim > 1 else coef_matrix
                    )

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
