"""
Linear residualization transformer for removing covariate effects from features.

Mirrors the BPt ``LinearResidualizer`` used in ``brain_signature`` but is
implemented as a pure scikit-learn transformer with no BPt dependency.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


class LinearResidualizer(BaseEstimator, TransformerMixin):
    """
    Remove linear covariate effects from features via OLS residualization.

    Fits an OLS regression predicting each feature column from the covariate
    columns, then subtracts the predicted values so the output features are
    orthogonal to the covariates.  Covariate columns are dropped from the
    output; only brain feature columns remain.

    Mirrors the ``LinearResidualizer`` used in ``brain_signature`` (BPt) but
    has no BPt dependency.

    Parameters
    ----------
    covariate_cols : list of str
        Column names to treat as covariates.  These columns must be present in
        every DataFrame passed to ``fit`` and ``transform``.  They are removed
        from the output.

    Attributes
    ----------
    reg_ : LinearRegression
        OLS model fitted on the training data.
    feature_cols_ : list of str
        Names of the non-covariate columns, in their original order.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "age": rng.standard_normal(100),
    ...     "feat_a": rng.standard_normal(100),
    ...     "feat_b": rng.standard_normal(100),
    ... })
    >>> resid = LinearResidualizer(covariate_cols=["age"])
    >>> resid.fit(df)
    LinearResidualizer(covariate_cols=['age'])
    >>> out = resid.transform(df)
    >>> list(out.columns)
    ['feat_a', 'feat_b']
    """

    def __init__(self, covariate_cols: list[str]) -> None:
        self.covariate_cols = covariate_cols

    def fit(self, X: pd.DataFrame, y=None) -> "LinearResidualizer":
        """
        Fit OLS on covariates → features using the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data containing both covariate columns and feature columns.
        y : ignored
            Unused; present for sklearn API compatibility.

        Returns
        -------
        self : LinearResidualizer
            Fitted transformer.

        Raises
        ------
        ValueError
            If any ``covariate_cols`` are absent from ``X``.
        """
        missing = [c for c in self.covariate_cols if c not in X.columns]
        if missing:
            msg = f"covariate_cols not found in X: {missing}"
            raise ValueError(msg)

        self.feature_cols_ = [c for c in X.columns if c not in self.covariate_cols]

        C = X[self.covariate_cols].to_numpy()
        F = X[self.feature_cols_].to_numpy()
        self.reg_ = LinearRegression(fit_intercept=True).fit(C, F)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Residualize features and return a DataFrame without covariate columns.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing both covariate columns and feature columns.
        y : ignored
            Unused; present for sklearn API compatibility.

        Returns
        -------
        pd.DataFrame
            Residualized feature DataFrame with covariate columns removed.
            Shape is ``(n_samples, n_features)`` where ``n_features`` equals
            ``len(X.columns) - len(covariate_cols)``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not been called.
        ValueError
            If any ``covariate_cols`` are absent from ``X``.
        """
        check_is_fitted(self, "reg_")

        missing = [c for c in self.covariate_cols if c not in X.columns]
        if missing:
            msg = f"covariate_cols not found in X: {missing}"
            raise ValueError(msg)

        C = X[self.covariate_cols].to_numpy()
        F = X[self.feature_cols_].to_numpy()
        residuals = F - self.reg_.predict(C)
        return pd.DataFrame(residuals, index=X.index, columns=self.feature_cols_)
