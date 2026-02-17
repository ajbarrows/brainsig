"""Unit tests for NeuralSignatureDataset."""

import numpy as np
import pandas as pd
import pytest

from brainsig.neural_dataset import NeuralSignatureDataset


@pytest.fixture
def paired_dfs():
    """Create paired condition DataFrames with subject IDs."""
    np.random.seed(42)
    n_subjects = 50
    n_features = 10
    subject_ids = [f"sub-{i:03d}" for i in range(n_subjects)]

    cond1 = pd.DataFrame(
        np.random.randn(n_subjects, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    cond1["subject_id"] = subject_ids

    cond0 = pd.DataFrame(
        np.random.randn(n_subjects, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    cond0["subject_id"] = subject_ids

    return cond1, cond0


@pytest.fixture
def paired_dfs_no_ids():
    """Create paired condition DataFrames without subject IDs."""
    np.random.seed(42)
    n_subjects = 50
    n_features = 10

    cond1 = pd.DataFrame(
        np.random.randn(n_subjects, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    cond0 = pd.DataFrame(
        np.random.randn(n_subjects, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    return cond1, cond0


class TestNeuralSignatureDatasetShapes:
    """Test shape properties of the dataset."""

    def test_x_train_shape(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            test_size=0.2, random_state=42, verbose=False,
        )
        n_train = int(len(dataset.subject_ids) * 0.8)
        # Allow for rounding
        assert dataset.X_train.shape[0] == 2 * (len(dataset.subject_ids) - dataset.X_test.shape[0] // 2)

    def test_x_train_is_double_subjects(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            test_size=0.2, random_state=42, verbose=False,
        )
        n_total = len(dataset.subject_ids)
        n_test = dataset.X_test.shape[0] // 2
        n_train = n_total - n_test
        assert dataset.X_train.shape[0] == 2 * n_train
        assert dataset.X_test.shape[0] == 2 * n_test

    def test_condition_shapes(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        n_subjects = len(dataset.subject_ids)
        n_features = dataset.X_train.shape[1]
        assert dataset.condition1.shape == (n_subjects, n_features)
        assert dataset.condition0.shape == (n_subjects, n_features)

    def test_feature_count_consistent(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        assert dataset.X_train.shape[1] == dataset.X_test.shape[1]
        assert dataset.X_train.shape[1] == len(dataset.feature_names)


class TestSubjectLevelSplit:
    """Test that train/test splitting is done at the subject level."""

    def test_no_subject_overlap(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            test_size=0.3, random_state=42, verbose=False,
        )
        # Since y_train is [1]*n_train + [0]*n_train, we can recover n_train
        n_train = dataset.y_train.sum()
        n_test = dataset.y_test.sum()

        # All subjects should be accounted for
        assert n_train + n_test == len(dataset.subject_ids)

    def test_subject_ids_preserved(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        assert len(dataset.subject_ids) == 50


class TestLabels:
    """Test label construction."""

    def test_y_train_structure(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        n_train = dataset.X_train.shape[0] // 2
        expected = np.array([1] * n_train + [0] * n_train)
        np.testing.assert_array_equal(dataset.y_train, expected)

    def test_y_test_structure(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        n_test = dataset.X_test.shape[0] // 2
        expected = np.array([1] * n_test + [0] * n_test)
        np.testing.assert_array_equal(dataset.y_test, expected)


class TestDuckTypeCompatibility:
    """Test that NeuralSignatureDataset has required attributes for NeuralSignature."""

    def test_has_feature_names(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        assert hasattr(dataset, "feature_names")
        assert len(dataset.feature_names) > 0

    def test_has_target_labels(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        assert dataset.target_labels == {"condition": [0, 1]}

    def test_has_preprocessor(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        assert hasattr(dataset, "preprocessor")
        assert dataset.preprocessor is not None

    def test_has_train_test_arrays(self, paired_dfs):
        cond1, cond0 = paired_dfs
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        assert hasattr(dataset, "X_train")
        assert hasattr(dataset, "X_test")
        assert hasattr(dataset, "y_train")
        assert hasattr(dataset, "y_test")


class TestValidation:
    """Test input validation."""

    def test_mismatched_row_counts(self):
        cond1 = pd.DataFrame({"a": [1, 2, 3]})
        cond0 = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="same number of rows"):
            NeuralSignatureDataset(cond1, cond0, verbose=False)

    def test_mismatched_subject_ids(self):
        cond1 = pd.DataFrame({"a": [1, 2], "subject_id": ["s1", "s2"]})
        cond0 = pd.DataFrame({"a": [3, 4], "subject_id": ["s1", "s3"]})
        with pytest.raises(ValueError, match="Subject IDs do not match"):
            NeuralSignatureDataset(
                cond1, cond0, subject_id_col="subject_id", verbose=False,
            )

    def test_missing_subject_id_col(self):
        cond1 = pd.DataFrame({"a": [1, 2]})
        cond0 = pd.DataFrame({"a": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            NeuralSignatureDataset(
                cond1, cond0, subject_id_col="subject_id", verbose=False,
            )


class TestWithoutSubjectIdCol:
    """Test that the dataset works without subject_id_col."""

    def test_integer_subject_ids(self, paired_dfs_no_ids):
        cond1, cond0 = paired_dfs_no_ids
        dataset = NeuralSignatureDataset(
            cond1, cond0, random_state=42, verbose=False,
        )
        np.testing.assert_array_equal(
            dataset.subject_ids, np.arange(50),
        )

    def test_shapes_without_ids(self, paired_dfs_no_ids):
        cond1, cond0 = paired_dfs_no_ids
        dataset = NeuralSignatureDataset(
            cond1, cond0, random_state=42, verbose=False,
        )
        assert dataset.condition1.shape[0] == 50
        assert dataset.condition0.shape[0] == 50
        assert dataset.X_train.shape[0] + dataset.X_test.shape[0] == 100


class TestMissingData:
    """Test missing data handling."""

    def test_drops_paired_subjects(self):
        np.random.seed(42)
        cond1 = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "subject_id": ["s1", "s2", "s3", "s4", "s5"],
        })
        cond0 = pd.DataFrame({
            "a": [5.0, 6.0, 7.0, np.nan, 9.0],
            "b": [50.0, 60.0, 70.0, 80.0, 90.0],
            "subject_id": ["s1", "s2", "s3", "s4", "s5"],
        })
        dataset = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            test_size=0.0001, random_state=42, verbose=False,
        )
        # s2 has NaN in cond1, s4 has NaN in cond0 => both dropped
        assert len(dataset.subject_ids) == 3
        assert "s2" not in dataset.subject_ids
        assert "s4" not in dataset.subject_ids
        assert dataset.dropped_summary["subjects_dropped"] == 2

    def test_drops_high_missing_columns(self):
        np.random.seed(42)
        n = 10
        cond1 = pd.DataFrame({
            "good": np.random.randn(n),
            "bad": [np.nan] * 8 + [1.0, 2.0],
        })
        cond0 = pd.DataFrame({
            "good": np.random.randn(n),
            "bad": [np.nan] * 8 + [3.0, 4.0],
        })
        dataset = NeuralSignatureDataset(
            cond1, cond0, missing_threshold=0.5,
            test_size=0.2, random_state=42, verbose=False,
        )
        assert "bad" not in [
            fn.replace("num__", "") for fn in dataset.feature_names
        ]
        assert "bad" in dataset.dropped_summary["high_missing_cols"]


class TestReproducibility:
    """Test that random_state ensures reproducibility."""

    def test_same_random_state_same_split(self, paired_dfs):
        cond1, cond0 = paired_dfs
        ds1 = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        ds2 = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.X_test, ds2.X_test)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)
        np.testing.assert_array_equal(ds1.y_test, ds2.y_test)

    def test_different_random_state_different_split(self, paired_dfs):
        cond1, cond0 = paired_dfs
        ds1 = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=42, verbose=False,
        )
        ds2 = NeuralSignatureDataset(
            cond1, cond0, subject_id_col="subject_id",
            random_state=99, verbose=False,
        )
        # Very unlikely to be the same split
        assert not np.array_equal(ds1.X_train, ds2.X_train)
