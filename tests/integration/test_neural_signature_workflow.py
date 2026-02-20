"""Integration tests for the complete neural signature workflow."""

import numpy as np
import pandas as pd
import pytest

from brainsig.dataset import Dataset
from brainsig.model import NeuralSignature


@pytest.fixture
def fmri_task_data():
    """
    Simulate fMRI task data for neural signature analysis.

    Creates data where:
    - Each subject has data for two conditions (labeled 1 and 0)
    - Condition 1 and Condition 0 represent different task states
    """
    np.random.seed(42)
    n_subjects = 50
    n_features = 20  # e.g., brain regions or voxels

    # Create condition labels: each subject appears twice (once per condition)
    subject_ids = np.repeat(np.arange(n_subjects), 2)
    conditions = np.tile([1, 0], n_subjects)  # Alternating conditions

    # Generate brain activity data
    # Add signal that discriminates conditions
    X = np.random.randn(n_subjects * 2, n_features)
    # Make condition 1 different from condition 0
    X[conditions == 1, :5] += 1.5  # Increase activity in first 5 features for condition 1
    X[conditions == 0, :5] -= 1.0  # Decrease for condition 0

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"region_{i}" for i in range(n_features)])
    df["subject_id"] = subject_ids
    df["condition"] = conditions

    return df


class TestNeuralSignatureEndToEnd:
    """Test complete neural signature workflow."""

    def test_full_workflow(self, fmri_task_data):
        """Test the complete neural signature analysis workflow."""
        # Step 1: Prepare dataset
        dataset = Dataset(
            fmri_task_data,
            target="condition",
            test_size=0.2,
            random_state=42,
            verbose=False,
        )

        # Step 2: Initialize and fit neural signature
        neural_sig = NeuralSignature(
            inner_folds=3, outer_folds=3, random_state=42, n_jobs=1
        )
        neural_sig.fit(dataset)

        # Step 3: Compute signature scores on new data
        # Simulate new subjects
        n_new_subjects = 10
        n_features = dataset.X_train.shape[1]

        condition1_data = np.random.randn(n_new_subjects, n_features)
        condition1_data[:, :5] += 1.5  # Similar pattern to training data

        condition0_data = np.random.randn(n_new_subjects, n_features)
        condition0_data[:, :5] -= 1.0

        scores = neural_sig.compute_signature_scores(condition1_data, condition0_data)

        # Step 4: Verify results
        assert len(scores) == n_new_subjects
        assert "signature_score" in scores.columns

        # Signature scores should generally be positive for discriminable data
        # (since condition1 has higher probability for class 1)
        assert scores["signature_score"].mean() > 0

    def test_cross_validation_workflow(self, fmri_task_data):
        """Test neural signature with cross-validation."""
        # Prepare dataset (use small test_size for CV)
        dataset = Dataset(
            fmri_task_data, target="condition", test_size=0.1, random_state=42, verbose=False
        )

        # Initialize and cross-validate
        neural_sig = NeuralSignature(
            inner_folds=2, outer_folds=3, random_state=42, n_jobs=1
        )
        cv_results = neural_sig.cross_validate(dataset)

        # Check CV results
        assert "condition" in cv_results
        assert "estimator" in cv_results["condition"]
        assert len(cv_results["condition"]["estimator"]) == 3

        # Get model performance per CV fold
        scores = neural_sig.get_cv_model_scores(dataset)
        assert "acc" in scores["metric"].values
        assert "f1" in scores["metric"].values
        assert "auc" in scores["metric"].values

        # Check that we have train and test scores
        assert "train" in scores["partition"].values
        assert "test" in scores["partition"].values

    def test_coefficient_analysis(self, fmri_task_data):
        """Test extracting and analyzing model coefficients."""
        # Prepare dataset
        dataset = Dataset(
            fmri_task_data, target="condition", test_size=0.1, random_state=42, verbose=False
        )

        # Fit and get coefficients
        neural_sig = NeuralSignature(
            inner_folds=2, outer_folds=2, random_state=42, n_jobs=1
        )
        neural_sig.cross_validate(dataset)

        coefs = neural_sig.get_coefficients(dataset)

        # Verify coefficient structure
        assert isinstance(coefs, pd.DataFrame)
        assert "cv_fold" in coefs.index.names
        assert len(coefs) > 0

        # Get odds ratios
        odds_ratios = neural_sig.get_coefficients(dataset, exponentiate=True)
        numeric_cols = odds_ratios.select_dtypes(include=[np.number]).columns
        assert (odds_ratios[numeric_cols] > 0).all().all()


class TestRealWorldScenarios:
    """Test scenarios closer to real-world usage."""

    def test_with_subject_identifiers(self, fmri_task_data):
        """Test workflow with subject identifiers."""
        dataset = Dataset(
            fmri_task_data, target="condition", test_size=0.2, random_state=42, verbose=False
        )

        neural_sig = NeuralSignature(random_state=42, n_jobs=1)
        neural_sig.fit(dataset)

        # Use custom subject IDs
        n_new_subjects = 5
        n_features = dataset.X_train.shape[1]
        subject_ids = [f"sub-{i:03d}" for i in range(100, 100 + n_new_subjects)]

        condition1_data = np.random.randn(n_new_subjects, n_features)
        condition0_data = np.random.randn(n_new_subjects, n_features)

        scores = neural_sig.compute_signature_scores(
            condition1_data, condition0_data, subject_ids=subject_ids
        )

        assert list(scores["subject_id"]) == subject_ids

    def test_performance_on_discriminable_data(self, fmri_task_data):
        """Test that model achieves good performance on discriminable data."""
        dataset = Dataset(
            fmri_task_data, target="condition", test_size=0.1, random_state=42, verbose=False
        )

        neural_sig = NeuralSignature(
            inner_folds=3, outer_folds=3, random_state=42, n_jobs=1
        )
        neural_sig.cross_validate(dataset)

        scores = neural_sig.get_cv_model_scores(dataset)

        # Get test AUC scores
        test_auc = scores[
            (scores["partition"] == "test") & (scores["metric"] == "auc")
        ]["value"]

        # Should achieve reasonable performance on this discriminable dataset
        assert test_auc.mean() > 0.6  # Better than random chance

    def test_handles_less_discriminable_data(self):
        """Test that model handles less discriminable data gracefully."""
        np.random.seed(42)
        n_samples = 100
        n_features = 15

        # Create nearly random data (hard to discriminate)
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples)

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        dataset = Dataset(df, target="target", test_size=0.2, random_state=42, verbose=False)

        neural_sig = NeuralSignature(random_state=42, n_jobs=1)
        neural_sig.fit(dataset)

        # Should still produce signature scores (even if not very discriminative)
        n_new = 10
        condition1_data = np.random.randn(n_new, n_features)
        condition0_data = np.random.randn(n_new, n_features)

        scores = neural_sig.compute_signature_scores(condition1_data, condition0_data)

        assert len(scores) == n_new
        assert not scores["signature_score"].isna().any()


class TestDataPreprocessingIntegration:
    """Test integration with Dataset preprocessing."""

    def test_with_categorical_features(self):
        """Test neural signature with categorical features."""
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "numeric1": np.random.randn(n_samples),
                "numeric2": np.random.randn(n_samples),
                "category": np.random.choice(["A", "B", "C"], size=n_samples),
                "condition": np.random.choice([0, 1], size=n_samples),
            }
        )

        dataset = Dataset(
            df, target="condition", test_size=0.2, random_state=42, verbose=False
        )

        neural_sig = NeuralSignature(random_state=42, n_jobs=1)
        neural_sig.fit(dataset)

        # Should handle mixed feature types
        assert len(neural_sig.classifier.models) == 1

    def test_with_missing_data(self):
        """Test that Dataset handles missing data before modeling."""
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": [np.nan if i % 10 == 0 else np.random.randn() for i in range(n_samples)],
                "condition": np.random.choice([0, 1], size=n_samples),
            }
        )

        # Dataset should clean the data
        dataset = Dataset(
            df, target="condition", test_size=0.2, random_state=42, verbose=False
        )

        # Model should train on cleaned data
        neural_sig = NeuralSignature(random_state=42, n_jobs=1)
        neural_sig.fit(dataset)

        # Verify rows with missing data were dropped
        assert dataset.dropped_summary["rows_dropped"] > 0
