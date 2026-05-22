"""Unit tests for LinearResidualizer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from brainsig.residualizer import LinearResidualizer


@pytest.fixture
def simple_df():
    """DataFrame with one covariate and two brain-like features."""
    rng = np.random.default_rng(42)
    n = 100
    age = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "age": age,
            "feat_a": 2.0 * age + rng.standard_normal(n),
            "feat_b": -1.5 * age + rng.standard_normal(n),
        }
    )


@pytest.fixture
def multi_covar_df():
    """DataFrame with two covariates and three features."""
    rng = np.random.default_rng(0)
    n = 200
    age = rng.standard_normal(n)
    sex = rng.choice([0.0, 1.0], size=n)
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "roi_1": 1.5 * age + 0.8 * sex + rng.standard_normal(n),
            "roi_2": -0.5 * age + 1.2 * sex + rng.standard_normal(n),
            "roi_3": rng.standard_normal(n),
        }
    )


class TestLinearResidualizerAPI:
    """Test sklearn API compliance."""

    def test_fit_returns_self(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        result = resid.fit(simple_df)
        assert result is resid

    def test_not_fitted_raises(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        with pytest.raises(NotFittedError):
            resid.transform(simple_df)

    def test_get_params(self):
        resid = LinearResidualizer(covariate_cols=["age", "sex"])
        params = resid.get_params()
        assert params["covariate_cols"] == ["age", "sex"]


class TestLinearResidualizerOutput:
    """Test output shape and content."""

    def test_output_drops_covariate_cols(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        out = resid.fit(simple_df).transform(simple_df)
        assert "age" not in out.columns
        assert set(out.columns) == {"feat_a", "feat_b"}

    def test_output_shape(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        out = resid.fit(simple_df).transform(simple_df)
        assert out.shape == (len(simple_df), 2)

    def test_output_is_dataframe(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        out = resid.fit(simple_df).transform(simple_df)
        assert isinstance(out, pd.DataFrame)

    def test_index_preserved(self, simple_df):
        df = simple_df.set_index(np.arange(100, 200))
        resid = LinearResidualizer(covariate_cols=["age"])
        out = resid.fit(df).transform(df)
        pd.testing.assert_index_equal(out.index, df.index)

    def test_feature_cols_attribute(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        resid.fit(simple_df)
        assert resid.feature_cols_ == ["feat_a", "feat_b"]

    def test_multi_covariate(self, multi_covar_df):
        resid = LinearResidualizer(covariate_cols=["age", "sex"])
        out = resid.fit(multi_covar_df).transform(multi_covar_df)
        assert out.shape == (len(multi_covar_df), 3)
        assert set(out.columns) == {"roi_1", "roi_2", "roi_3"}


class TestLinearResidualizerResidualisation:
    """Test that covariate effects are actually removed."""

    def test_residuals_uncorrelated_with_covariate(self, simple_df):
        # Fit on the first half; test on the held-out second half.
        # In-sample OLS residuals are mathematically guaranteed to be orthogonal
        # to the regressors (normal equations), so only out-of-sample correlation
        # is a meaningful test of the implementation.
        n = len(simple_df)
        train = simple_df.iloc[: n // 2]
        test = simple_df.iloc[n // 2 :]

        resid = LinearResidualizer(covariate_cols=["age"])
        resid.fit(train)
        out_test = resid.transform(test)

        for col in out_test.columns:
            r_raw = np.corrcoef(test["age"], test[col])[0, 1]
            r_resid = np.corrcoef(test["age"], out_test[col])[0, 1]
            # Residualization must substantially reduce the covariate correlation
            assert abs(r_resid) < abs(r_raw) * 0.25

    def test_residuals_differ_from_raw(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        out = resid.fit(simple_df).transform(simple_df)
        assert not np.allclose(out["feat_a"].to_numpy(), simple_df["feat_a"].to_numpy())

    def test_train_test_leakage_prevention(self, multi_covar_df):
        """Transform uses train-fitted OLS coefficients, not re-fitted on test data."""
        n = len(multi_covar_df)
        train = multi_covar_df.iloc[: n // 2].copy()
        test = multi_covar_df.iloc[n // 2 :].copy()

        resid = LinearResidualizer(covariate_cols=["age", "sex"])
        resid.fit(train)
        coef_before = resid.reg_.coef_.copy()

        out_test = resid.transform(test)

        # transform() must not mutate the fitted OLS coefficients
        np.testing.assert_array_equal(resid.reg_.coef_, coef_before)
        # Test residuals must differ from what fitting directly on test data gives,
        # confirming that train-fitted (not test-fitted) coefficients were used
        resid_on_test = LinearResidualizer(covariate_cols=["age", "sex"])
        resid_on_test.fit(test)
        out_test_refit = resid_on_test.transform(test)
        assert not np.allclose(out_test.to_numpy(), out_test_refit.to_numpy())


class TestLinearResidualizerValidation:
    """Test input validation."""

    def test_missing_covariate_col_at_fit(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["nonexistent"])
        with pytest.raises(ValueError, match="covariate_cols not found"):
            resid.fit(simple_df)

    def test_missing_covariate_col_at_transform(self, simple_df):
        resid = LinearResidualizer(covariate_cols=["age"])
        resid.fit(simple_df)
        df_no_age = simple_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="covariate_cols not found"):
            resid.transform(df_no_age)
