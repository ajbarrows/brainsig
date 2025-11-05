"""Unit tests for ElasticNetClassifier and NeuralSignature classes."""

import numpy as np
import pandas as pd
import pytest

from brainsig.dataset import Dataset
from brainsig.model import ElasticNetClassifier, NeuralSignature


@pytest.fixture
def binary_dataset():
    """Create a binary classification dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Create separable classes
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    return Dataset(df, target="target", test_size=0.2, random_state=42, verbose=False)


@pytest.fixture
def multiclass_dataset():
    """Create a multiclass classification dataset."""
    np.random.seed(42)
    n_samples = 150
    n_features = 8

    X = np.random.randn(n_samples, n_features)
    # Create 3 separable classes
    y = np.zeros(n_samples, dtype=int)
    y[X[:, 0] + X[:, 1] > 1] = 2
    y[(X[:, 0] + X[:, 1] <= 1) & (X[:, 0] + X[:, 1] > -1)] = 1

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    return Dataset(df, target="target", test_size=0.2, random_state=42, verbose=False)


class TestElasticNetClassifierInitialization:
    """Test ElasticNetClassifier initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        classifier = ElasticNetClassifier()

        assert classifier.inner_folds == 5
        assert classifier.outer_folds == 5
        assert classifier.inner_scoring == "roc_auc_ovr"
        assert classifier.max_iter == 1000
        assert classifier.n_jobs == -1
        assert classifier.random_state == 42

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        classifier = ElasticNetClassifier(
            inner_folds=3,
            outer_folds=4,
            cs=[0.1, 1.0],
            l1_ratios=[0.5, 1.0],
            max_iter=500,
            random_state=123,
        )

        assert classifier.inner_folds == 3
        assert classifier.outer_folds == 4
        assert classifier.Cs == [0.1, 1.0]
        assert classifier.l1_ratios == [0.5, 1.0]
        assert classifier.max_iter == 500
        assert classifier.random_state == 123

    def test_cv_scheme_initialization(self):
        """Test that CV schemes are properly initialized."""
        classifier = ElasticNetClassifier()

        assert classifier.inner_cv is not None
        assert classifier.outer_cv is not None
        assert classifier.inner_cv.n_splits == 5
        assert classifier.outer_cv.n_splits == 5


class TestElasticNetClassifierFitting:
    """Test model fitting."""

    def test_fit_binary_classification(self, binary_dataset):
        """Test fitting on binary classification data."""
        classifier = ElasticNetClassifier(
            inner_folds=2, outer_folds=2, random_state=42
        )
        classifier.fit_model(binary_dataset)

        assert len(classifier.models) == 1
        assert len(classifier.target_names) == 1
        assert "target" in classifier.models

    def test_fit_stores_dataset(self, binary_dataset):
        """Test that dataset is stored when keep_dataset=True."""
        classifier = ElasticNetClassifier()
        classifier.fit_model(binary_dataset, keep_dataset=True)

        assert classifier.dataset is not None
        assert classifier.dataset == binary_dataset

    def test_fit_no_dataset_storage(self, binary_dataset):
        """Test that dataset is not stored when keep_dataset=False."""
        classifier = ElasticNetClassifier()
        classifier.fit_model(binary_dataset, keep_dataset=False)

        assert classifier.dataset is None

    def test_fitted_model_has_attributes(self, binary_dataset):
        """Test that fitted model has expected attributes."""
        classifier = ElasticNetClassifier(random_state=42)
        classifier.fit_model(binary_dataset)

        model = classifier.models["target"]
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestElasticNetClassifierPrediction:
    """Test model prediction."""

    def test_predict_binary(self, binary_dataset):
        """Test prediction on binary classification."""
        classifier = ElasticNetClassifier(random_state=42)
        classifier.fit_model(binary_dataset)
        predictions = classifier.predict(binary_dataset)

        assert "target" in predictions
        assert "y_pred" in predictions["target"]
        assert "y_pred_proba" in predictions["target"]
        assert "y_true" in predictions["target"]

        # Check shapes
        assert len(predictions["target"]["y_pred"]) == len(binary_dataset.y_test)
        assert predictions["target"]["y_pred_proba"].shape[0] == len(
            binary_dataset.y_test
        )

    def test_prediction_values_valid(self, binary_dataset):
        """Test that predictions are valid."""
        classifier = ElasticNetClassifier(random_state=42)
        classifier.fit_model(binary_dataset)
        predictions = classifier.predict(binary_dataset)

        y_pred = predictions["target"]["y_pred"]
        y_proba = predictions["target"]["y_pred_proba"]

        # Predictions should be 0 or 1
        assert set(y_pred).issubset({0, 1})
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), 1.0)
        # Probabilities should be in [0, 1]
        assert (y_proba >= 0).all() and (y_proba <= 1).all()


class TestElasticNetClassifierCrossValidation:
    """Test cross-validation."""

    def test_cross_validate_binary(self, binary_dataset):
        """Test cross-validation on binary data."""
        classifier = ElasticNetClassifier(
            inner_folds=2, outer_folds=2, random_state=42
        )
        cv_results = classifier.cross_validate(binary_dataset)

        assert "target" in cv_results
        assert "estimator" in cv_results["target"]
        assert "test_acc" in cv_results["target"]
        assert "test_f1" in cv_results["target"]
        assert "test_auc" in cv_results["target"]

    def test_cv_returns_estimators(self, binary_dataset):
        """Test that CV returns fitted estimators."""
        classifier = ElasticNetClassifier(
            inner_folds=2, outer_folds=3, random_state=42
        )
        cv_results = classifier.cross_validate(binary_dataset)

        estimators = cv_results["target"]["estimator"]
        assert len(estimators) == 3  # outer_folds
        assert all(hasattr(est, "coef_") for est in estimators)

    def test_cv_scores_shape(self, binary_dataset):
        """Test that CV scores have correct shape."""
        classifier = ElasticNetClassifier(
            inner_folds=2, outer_folds=3, random_state=42
        )
        cv_results = classifier.cross_validate(binary_dataset)

        assert len(cv_results["target"]["test_acc"]) == 3
        assert len(cv_results["target"]["test_f1"]) == 3
        assert len(cv_results["target"]["test_auc"]) == 3


class TestElasticNetClassifierCoefficients:
    """Test coefficient extraction."""

    def test_get_cv_coefs(self, binary_dataset):
        """Test getting coefficients from CV models."""
        classifier = ElasticNetClassifier(
            inner_folds=2, outer_folds=2, random_state=42
        )
        classifier.cross_validate(binary_dataset)
        coefs = classifier.get_cv_coefs(binary_dataset)

        assert isinstance(coefs, pd.DataFrame)
        assert "cv_fold" in coefs.index.names
        assert "target_variable" in coefs.index.names
        assert len(coefs) > 0

    def test_coefs_exponentiate(self, binary_dataset):
        """Test coefficient exponentiation (odds ratios)."""
        classifier = ElasticNetClassifier(
            inner_folds=2, outer_folds=2, random_state=42
        )
        classifier.cross_validate(binary_dataset)

        coefs_normal = classifier.get_cv_coefs(binary_dataset, exponentiate=False)
        coefs_exp = classifier.get_cv_coefs(binary_dataset, exponentiate=True)

        # Exponentiated coefs should be positive
        numeric_cols = coefs_exp.select_dtypes(include=[np.number]).columns
        assert (coefs_exp[numeric_cols] > 0).all().all()


class TestNeuralSignatureInitialization:
    """Test NeuralSignature initialization."""

    def test_default_initialization(self):
        """Test default NeuralSignature initialization."""
        neural_sig = NeuralSignature()

        assert neural_sig.classifier is not None
        assert isinstance(neural_sig.classifier, ElasticNetClassifier)
        assert neural_sig.signature_scores is None

    def test_binary_scoring_defaults(self):
        """Test that binary scoring is used by default."""
        neural_sig = NeuralSignature()

        # Should use binary 'roc_auc' instead of 'roc_auc_ovr'
        assert neural_sig.classifier.inner_scoring == "roc_auc"
        assert neural_sig.classifier.outer_scoring["auc"] == "roc_auc"


class TestNeuralSignatureFitting:
    """Test NeuralSignature fitting."""

    def test_fit(self, binary_dataset):
        """Test fitting neural signature model."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        assert len(neural_sig.classifier.models) == 1
        assert "target" in neural_sig.classifier.models


class TestNeuralSignatureScoreComputation:
    """Test neural signature score computation."""

    def test_compute_signature_scores(self, binary_dataset):
        """Test computing signature scores."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        # Create dummy condition data
        n_subjects = 20
        n_features = binary_dataset.X_train.shape[1]
        condition1_data = np.random.randn(n_subjects, n_features)
        condition0_data = np.random.randn(n_subjects, n_features)

        scores = neural_sig.compute_signature_scores(condition1_data, condition0_data)

        assert isinstance(scores, pd.DataFrame)
        assert "subject_id" in scores.columns
        assert "condition1_prob" in scores.columns
        assert "condition0_prob" in scores.columns
        assert "signature_score" in scores.columns
        assert len(scores) == n_subjects

    def test_signature_score_calculation(self, binary_dataset):
        """Test that signature scores are correctly calculated."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        n_subjects = 20
        n_features = binary_dataset.X_train.shape[1]
        condition1_data = np.random.randn(n_subjects, n_features)
        condition0_data = np.random.randn(n_subjects, n_features)

        scores = neural_sig.compute_signature_scores(condition1_data, condition0_data)

        # Verify calculation: score = P(cond1) - P(cond0)
        expected_scores = (
            scores["condition1_prob"].values - scores["condition0_prob"].values
        )
        np.testing.assert_array_almost_equal(
            scores["signature_score"].values, expected_scores
        )

    def test_custom_subject_ids(self, binary_dataset):
        """Test using custom subject IDs."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        n_subjects = 10
        n_features = binary_dataset.X_train.shape[1]
        condition1_data = np.random.randn(n_subjects, n_features)
        condition0_data = np.random.randn(n_subjects, n_features)
        subject_ids = [f"sub-{i:03d}" for i in range(n_subjects)]

        scores = neural_sig.compute_signature_scores(
            condition1_data, condition0_data, subject_ids=subject_ids
        )

        assert list(scores["subject_id"]) == subject_ids

    def test_error_if_not_fitted(self):
        """Test that error is raised if model not fitted."""
        neural_sig = NeuralSignature()

        condition1_data = np.random.randn(10, 5)
        condition0_data = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Model must be fitted"):
            neural_sig.compute_signature_scores(condition1_data, condition0_data)

    def test_error_if_shape_mismatch(self, binary_dataset):
        """Test that error is raised if data shapes don't match."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        n_features = binary_dataset.X_train.shape[1]
        condition1_data = np.random.randn(10, n_features)
        condition0_data = np.random.randn(15, n_features)  # Different number of subjects

        with pytest.raises(ValueError, match="Data shapes must match"):
            neural_sig.compute_signature_scores(condition1_data, condition0_data)

    def test_error_if_subject_ids_mismatch(self, binary_dataset):
        """Test error if number of subject_ids doesn't match data."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        n_features = binary_dataset.X_train.shape[1]
        condition1_data = np.random.randn(10, n_features)
        condition0_data = np.random.randn(10, n_features)
        subject_ids = ["sub1", "sub2"]  # Only 2 IDs for 10 subjects

        with pytest.raises(ValueError, match="Number of subject_ids"):
            neural_sig.compute_signature_scores(
                condition1_data, condition0_data, subject_ids=subject_ids
            )


class TestNeuralSignatureCVScores:
    """Test CV signature score computation."""

    def test_get_cv_signature_scores_error_if_no_cv(self, binary_dataset):
        """Test error if CV not performed."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.fit(binary_dataset)

        condition1_indices = np.array([0, 1, 2])
        condition0_indices = np.array([3, 4, 5])

        with pytest.raises(ValueError, match="Must run cross_validate"):
            neural_sig.get_cv_signature_scores(
                binary_dataset, condition1_indices, condition0_indices
            )


class TestNeuralSignatureHelperMethods:
    """Test helper methods."""

    def test_get_coefficients(self, binary_dataset):
        """Test getting model coefficients."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.cross_validate(binary_dataset)

        coefs = neural_sig.get_coefficients(binary_dataset)

        assert isinstance(coefs, pd.DataFrame)
        assert len(coefs) > 0

    def test_get_model_scores(self, binary_dataset):
        """Test getting model performance scores."""
        neural_sig = NeuralSignature(random_state=42)
        neural_sig.cross_validate(binary_dataset)

        scores = neural_sig.get_model_scores(binary_dataset)

        assert isinstance(scores, pd.DataFrame)
        assert "metric" in scores.columns
        assert "value" in scores.columns
