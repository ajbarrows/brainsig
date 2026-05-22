"""Unit tests for the Dataset class."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from brainsig.dataset import Dataset


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "city": ["NYC", "LA", "NYC", "SF", "LA"],
            "outcome": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def dataframe_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40, 45],
            "income": [50000, np.nan, 70000, 80000, 90000],
            "all_missing": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "mostly_missing": [1.0, np.nan, np.nan, np.nan, np.nan],
            "outcome": [0, 1, 0, 1, 0],
        }
    )


class TestDatasetInitialization:
    """Test Dataset initialization."""

    def test_basic_initialization(self, sample_dataframe):
        """Test basic Dataset initialization."""
        dataset = Dataset(sample_dataframe, target="outcome", random_state=42)

        assert hasattr(dataset, "X_train")
        assert hasattr(dataset, "X_test")
        assert hasattr(dataset, "y_train")
        assert hasattr(dataset, "y_test")
        assert dataset.target == "outcome"

    def test_no_split(self, sample_dataframe):
        """Test Dataset with no train/test split."""
        dataset = Dataset(sample_dataframe, target="outcome", test_size=0)

        assert hasattr(dataset, "X")
        assert hasattr(dataset, "y")
        assert not hasattr(dataset, "X_train")
        assert dataset.X.shape[0] == 5

    def test_original_df_preserved(self, sample_dataframe):
        """Test that original DataFrame is preserved."""
        dataset = Dataset(sample_dataframe, target="outcome")

        assert dataset.original_df.equals(sample_dataframe)
        # Ensure it's a copy, not a reference
        dataset.original_df.loc[0, "age"] = 999
        assert sample_dataframe.loc[0, "age"] == 25


class TestMissingDataHandling:
    """Test missing data handling."""

    def test_drop_all_missing_columns(self, dataframe_with_missing):
        """Test that completely missing columns are dropped."""
        dataset = Dataset(
            dataframe_with_missing, target="outcome", verbose=False, random_state=42
        )

        assert "all_missing" in dataset.dropped_summary["all_missing_cols"]
        assert "all_missing" not in dataset.original_df.columns[
            dataset.original_df.columns.isin(dataset.feature_names)
        ]

    def test_drop_high_missing_columns(self, dataframe_with_missing):
        """Test that columns above threshold are dropped."""
        dataset = Dataset(
            dataframe_with_missing,
            target="outcome",
            missing_threshold=0.5,
            verbose=False,
            random_state=42,
        )

        assert "mostly_missing" in dataset.dropped_summary["high_missing_cols"]

    def test_drop_missing_rows(self, dataframe_with_missing):
        """Test that rows with missing values are dropped."""
        dataset = Dataset(
            dataframe_with_missing, target="outcome", verbose=False, random_state=42
        )

        assert dataset.dropped_summary["rows_dropped"] > 0

    def test_missing_threshold_parameter(self):
        """Test different missing threshold values."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5],
                "col2": [1, np.nan, np.nan, np.nan, 5],  # 60% missing
                "outcome": [0, 1, 0, 1, 0],
            }
        )

        # With threshold 0.5, col2 should be dropped
        dataset1 = Dataset(df, target="outcome", missing_threshold=0.5, verbose=False)
        assert "col2" in dataset1.dropped_summary["high_missing_cols"]

        # With threshold 0.7, col2 should be kept
        dataset2 = Dataset(df, target="outcome", missing_threshold=0.7, verbose=False)
        assert "col2" not in dataset2.dropped_summary["high_missing_cols"]


class TestPreprocessing:
    """Test feature preprocessing."""

    def test_numeric_feature_scaling(self, sample_dataframe):
        """Test that numeric features are scaled by RobustScaler."""
        dataset = Dataset(sample_dataframe, target="outcome", random_state=42)

        numeric_indices = [
            i
            for i, name in enumerate(dataset.feature_names)
            if name.startswith("num__")
        ]

        if numeric_indices:
            numeric_features = dataset.X_train[:, numeric_indices]
            # Scaled values must be of order 1, not the original scale (e.g. income = 50,000)
            assert np.abs(numeric_features).max() < 10
            # RobustScaler centres on the median: training-set median is exactly 0
            np.testing.assert_allclose(
                np.median(numeric_features, axis=0), 0.0, atol=1e-10
            )

    def test_categorical_encoding(self, sample_dataframe):
        """Test that categorical features are one-hot encoded."""
        dataset = Dataset(sample_dataframe, target="outcome", random_state=42)

        cat_features = [
            name for name in dataset.feature_names if name.startswith("cat__")
        ]
        assert len(cat_features) > 0
        # drop="if_binary" keeps all k columns for multi-category features
        assert "cat__city_LA" in cat_features
        assert "cat__city_NYC" in cat_features

    def test_ohe_drop_if_binary(self):
        """drop='if_binary' drops one column only for binary categoricals."""
        df = pd.DataFrame(
            {
                "multi_cat": ["a", "b", "c", "a", "b"],  # 3 categories: no drop
                "binary_cat": ["x", "y", "x", "y", "x"],  # 2 categories: drop one
                "outcome": [0, 1, 0, 1, 0],
            }
        )
        dataset = Dataset(df, target="outcome", random_state=42, verbose=False)
        cat_features = [n for n in dataset.feature_names if n.startswith("cat__")]
        multi_cols = [n for n in cat_features if "multi_cat" in n]
        binary_cols = [n for n in cat_features if "binary_cat" in n]
        assert len(multi_cols) == 3
        assert len(binary_cols) == 1

    def test_custom_preprocessor(self, sample_dataframe):
        """Test using a custom preprocessor."""
        from sklearn.preprocessing import MinMaxScaler

        custom_preprocessor = ColumnTransformer(
            [("num", MinMaxScaler(), ["age", "income"])]
        )

        dataset = Dataset(
            sample_dataframe,
            target="outcome",
            preprocessor=custom_preprocessor,
            random_state=42,
        )

        # Values should be scaled to [0, 1] range
        assert dataset.X_train.min() >= 0
        assert dataset.X_train.max() <= 1


class TestTargetHandling:
    """Test target variable handling."""

    def test_single_target(self, sample_dataframe):
        """Test with single target variable."""
        dataset = Dataset(sample_dataframe, target="outcome", random_state=42)

        assert dataset.y_train.ndim == 1 or dataset.y_train.shape[1] == 1
        assert "outcome" in dataset.target_labels

    def test_target_labels_extraction(self):
        """Test that target labels are correctly extracted."""
        # Use categorical target for label extraction
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "target": pd.Categorical(["A", "B", "A", "B", "A"]),
            }
        )
        dataset = Dataset(df, target="target", random_state=42, verbose=False)

        assert "target" in dataset.target_labels
        assert set(dataset.target_labels["target"]) == {"A", "B"}

    def test_categorical_target(self):
        """Test with categorical target variable."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "target": pd.Categorical(["A", "B", "A", "B", "A"], categories=["A", "B"]),
            }
        )

        dataset = Dataset(df, target="target", random_state=42)

        assert dataset.target_labels["target"] == ["A", "B"]


class TestTrainTestSplit:
    """Test train/test splitting."""

    def test_split_sizes(self, sample_dataframe):
        """Test that train/test split produces correct sizes."""
        dataset = Dataset(sample_dataframe, target="outcome", test_size=0.2, random_state=42)

        total_samples = len(sample_dataframe)
        assert dataset.X_train.shape[0] == int(total_samples * 0.8)
        assert dataset.X_test.shape[0] == int(total_samples * 0.2)

    def test_random_state_reproducibility(self, sample_dataframe):
        """Test that random_state ensures reproducibility."""
        dataset1 = Dataset(sample_dataframe, target="outcome", random_state=42)
        dataset2 = Dataset(sample_dataframe, target="outcome", random_state=42)

        np.testing.assert_array_equal(dataset1.X_train, dataset2.X_train)
        np.testing.assert_array_equal(dataset1.y_train, dataset2.y_train)

    def test_different_random_states(self, sample_dataframe):
        """Test that different random states produce different splits."""
        dataset1 = Dataset(sample_dataframe, target="outcome", random_state=42)
        dataset2 = Dataset(sample_dataframe, target="outcome", random_state=123)

        # Splits should be different
        assert not np.array_equal(dataset1.y_train, dataset2.y_train)


class TestFeatureNames:
    """Test feature name handling."""

    def test_feature_names_preserved(self, sample_dataframe):
        """Test that feature names are accessible."""
        dataset = Dataset(sample_dataframe, target="outcome", random_state=42)

        assert hasattr(dataset, "feature_names")
        assert len(dataset.feature_names) > 0
        assert len(dataset.feature_names) == dataset.X_train.shape[1]
