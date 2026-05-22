# `brainsig` vs `brain_signature` — Pipeline Comparison

This document compares the neural signature pipeline implemented in `brain_signature`
(the full ABCD-specific analysis project) with the `brainsig` standalone package,
and describes what changes are required **to `brainsig`** so that it reproduces
`brain_signature` results within stochastic error.

**All changes described here are to be made in `brainsig`
(`/gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/brainsig`).**
`brain_signature` (`/users/a/j/ajbarrow/ABCD/brain_signature`) is the reference
implementation and is not modified.

---

## 1. Package Overview

| | `brain_signature` (reference) | `brainsig` (target) |
|---|---|---|
| **Path** | `ABCD/brain_signature/brain_signature/` | `ABCD/brainsig/src/brainsig/` |
| **Purpose** | Full ABCD analysis pipeline | Generalizable, distributable package |
| **ML framework** | BPt (Brain Prediction Toolbox) | Pure scikit-learn |
| **Dependencies** | BPt, pyrelimri, statsmodels, abcd_tools, rpy2 | numpy, pandas, scikit-learn only |
| **Entry points** | `slurm/fit_neuralsig.py` | `NeuralSignature`, `NeuralSignatureDataset` classes |

---

## 2. ML Model

Both projects use **ElasticNet logistic regression** for binary classification between
two task conditions. The implementations differ in how hyperparameter tuning and CV are
structured.

### `brain_signature` (reference)

```python
# BPt pipeline
enet = bp.Model("elastic", params=1)  # BPt's built-in ElasticNet param grid
param_search = bp.ParamSearch()       # Default: randomized search
pipe = bp.Pipeline([scaler, ohe_tr, resid_tr, enet], param_search=param_search)
cv = bp.CV(splits=5, n_repeats=1)
results = bp.evaluate(pipeline=pipe, dataset=dataset, cv=cv, ...)
```

- 5-fold CV with 1 repeat
- `get_best_model()` selects the fold with highest validation ROC-AUC
- That single fold's model is used for everything downstream (Haufe, predictions)

### `brainsig` (current)

```python
# Nested CV via scikit-learn
LogisticRegressionCV(
    cv=StratifiedKFold(n_splits=5),  # inner CV: hyperparameter tuning
    penalty="elasticnet",
    solver="saga",
    Cs=[0.001, 0.01, 0.1, 1, 10],
    l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
)
# Outer CV: StratifiedKFold(n_splits=5) for performance evaluation
```

- True nested CV: inner loop tunes hyperparameters, outer loop evaluates performance
- Final model is fit on all training data (not a single fold's model)
- **Methodologically more rigorous** — best-fold selection inflates apparent performance

**Decision:** `brainsig` retains its nested CV approach for performance evaluation
(more rigorous). However, Haufe feature maps require a concrete fitted estimator
with a specific set of predictions. `brainsig` will expose `get_best_cv_model(metric)`
to select the best outer-fold estimator (by ROC-AUC), and `compute_haufe_features()`
will use that estimator's training fold to compute `y_pred`. This mirrors
`brain_signature`'s approach for the Haufe step only — see Section 4.

**Impact:** Small for large ABCD samples (N > 3,000 per timepoint) for performance
metrics. Haufe maps may differ slightly due to the CV strategy difference; correlation
is expected to be > 0.95.

---

## 3. Preprocessing Changes to `brainsig`

This is the largest source of divergence.

### 3a. Feature Scaling

| | `brain_signature` (reference) | `brainsig` (current → target) |
|---|---|---|
| Scaler | `RobustScaler` (robust to outliers) | `StandardScaler` → **`RobustScaler`** |
| Scope | Float/numeric brain features only | All numeric features |

`RobustScaler` scales using the median and IQR rather than mean and variance. With
neuroimaging data containing outlier subjects, this produces meaningfully different
model weights.

**Fix in `brainsig`:** Replace `StandardScaler()` with `RobustScaler()` in
`NeuralSignatureDataset._build_default_preprocessor()` and
`Dataset._build_default_preprocessor()`.

---

### 3b. Covariate Residualization — **Largest Gap**

`brain_signature` uses BPt's `LinearResidualizer` to remove covariate effects
(age, sex, site, scanner, motion) from brain features *before* model fitting:

```python
# brain_signature/model.py lines 296–298
resid = LinearResidualizer(to_resid_df=ds["covariates"], fit_intercept=True)
resid_tr = bp.Scaler(resid, scope=list(scopes.keys()))
```

The effect: the ElasticNet model sees only the covariate-residualized brain
activations. Covariates are never model predictors — they are projected out.

**`brainsig` has no residualization at all.** Without this step, models fitted with
`brainsig` will partially learn covariate effects rather than pure condition-related
activations, producing different (and less clean) neural signature maps.

**Fix in `brainsig`:** Add `src/brainsig/residualizer.py` with a pure-sklearn
`LinearResidualizer`. Apply it at the **`NeuralSignatureDataset` level** on the raw
DataFrame, before any sklearn pipeline. Drop covariate columns from the output so the
downstream `ColumnTransformer` never sees them.

```python
# src/brainsig/residualizer.py
class LinearResidualizer(BaseEstimator, TransformerMixin):
    """Fit OLS on covariates; residualize features by subtracting predictions."""
    def __init__(self, covariate_cols: list[str]):
        self.covariate_cols = covariate_cols

    def fit(self, X: pd.DataFrame, y=None):
        C = X[self.covariate_cols].to_numpy()
        F = X.drop(columns=self.covariate_cols).to_numpy()
        self.reg_ = LinearRegression(fit_intercept=True).fit(C, F)
        self.feature_cols_ = [c for c in X.columns if c not in self.covariate_cols]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        C = X[self.covariate_cols].to_numpy()
        F = X.drop(columns=self.covariate_cols).to_numpy()
        residuals = F - self.reg_.predict(C)
        return pd.DataFrame(residuals, index=X.index, columns=self.feature_cols_)
```

`transform()` returns a DataFrame with covariate columns removed, so the
`ColumnTransformer` in `NeuralSignatureDataset._build_default_preprocessor()`
only ever sees brain feature columns — no restructuring of the existing pipeline
is needed.

Add `covariate_cols: list[str] | None = None` to `NeuralSignatureDataset.__init__()`.
When provided, fit a `LinearResidualizer` on the training split and apply it to both
splits before passing data to the `ColumnTransformer`.

---

### 3c. OneHotEncoder Drop Strategy

| | `brain_signature` (reference) | `brainsig` (current → target) |
|---|---|---|
| `drop` parameter | `"if_binary"` | `"first"` → **`"if_binary"`** |

`drop="if_binary"` drops one column only when a feature has **exactly 2** categories.
For features with more than 2 categories (e.g., site), it retains all `k` dummy
columns. `drop="first"` always drops one column regardless of cardinality, producing
`k-1` dummies.

The practical difference is small when elastic net regularization handles any
resulting collinearity, but matching `brain_signature`'s convention is required for
reproducibility.

**Fix in `brainsig`:** Change `drop="first"` → `drop="if_binary"` in both
`Dataset._build_default_preprocessor()` and
`NeuralSignatureDataset._build_default_preprocessor()`.

---

## 4. Haufe Transformation — **Not Implemented in `brainsig`**

`brain_signature` applies the Haufe transformation to convert ElasticNet coefficients
into interpretable activation maps that account for feature correlations:

```python
# brain_signature/model.py lines 511–534
def haufe_transform(X, y_pred, chunk_size=10000):
    n = len(X)
    X_mean = X.mean(axis=0)
    y_centered = y_pred - y_pred.mean()
    if chunk_size >= n:
        return ((X - X_mean).T @ y_centered) / n
    else:
        result = np.zeros(X.shape[1])
        for i in range(0, X.shape[1], chunk_size):
            end = min(i + chunk_size, X.shape[1])
            result[i:end] = (X[:, i:end] - X_mean[i:end]).T @ y_centered
        return result / n

# Uses class-0 probability to match abcd_tools sign convention
y_pred = best_model.predict_proba(X)[:, 0]
haufe_features = haufe_transform(np.asarray(X), y_pred)
```

**Sign convention:** `[:, 0]` (probability of condition 0) is used as the Haufe
target, which inverts the sign relative to standard coefficients. This matches
the original `abcd_tools` implementation. Signature scores (predictions) separately
use `[:, 1]`, so there is an intentional sign flip between Haufe maps and scores.

**Which X and model to use:** `brain_signature` uses the best-fold estimator and
that fold's training data. In `brainsig`, `compute_haufe_features()` must:
1. Call `get_best_cv_model(metric="roc_auc")` to retrieve the best outer-fold
   estimator from the cross-validation results.
2. Use that estimator's training fold indices to recover the corresponding `X_train`
   and `y_train` slices.
3. Call `estimator.predict_proba(X_train)[:, 0]` to obtain `y_pred`.
4. Pass `X_train` (preprocessed, as a numpy array) and `y_pred` to `_haufe_transform()`.

The chunked implementation is required: ABCD brain feature matrices are high-dimensional
(~400 ROIs × N subjects) and the chunk-wise covariance avoids large intermediate arrays.

**Fix in `brainsig`:** Add the following to `NeuralSignature` in `model.py`:
- `get_best_cv_model(metric: str = "roc_auc") -> estimator` — selects best outer-fold
  estimator from `cv_results_`; depends on `cross_validate()` having been called first.
- `_haufe_transform(X, y_pred, chunk_size=10000) -> np.ndarray` — static method,
  exact port of the above.
- `compute_haufe_features(dataset: NeuralSignatureDataset) -> pd.Series` — orchestrates
  steps 1–4 above; returns a Series indexed by feature name.

These three methods must be implemented together as a unit (Priority 2a).

---

## 5. Neural Signature Score Computation

Both projects compute the signature score as the difference in predicted probabilities
between the two conditions for each subject. The sign convention is aligned:

| | Formula |
|---|---|
| `brain_signature` | `P(cond1=1 \| brain_cond1) - P(cond1=1 \| brain_cond0)` using `[:, 1]` |
| `brainsig` | `P(y=1 \| condition1_data) - P(y=1 \| condition0_data)` using `[:, 1]` |

**No change needed** — this is already consistent.

---

## 6. ICC (Intraclass Correlation) — **Not Implemented in `brainsig`**

`brain_signature` quantifies test-retest reliability of neural signatures across runs
and timepoints using `pyrelimri`:

```python
from pyrelimri.icc import sumsq_icc

icc, lower, upper, *_ = sumsq_icc(
    ns_df, sub_var='src_subject_id', sess_var='session', value_var='value'
)
```

**Two modes:**
- **Run mode:** ICC between run1 and run2 within the same timepoint
- **Timepoint mode:** ICC across longitudinal waves

**Dependency on Section 7 (Run Scopes):** `compute_run_icc()` requires run-level
predictions, which means `NeuralSignatureDataset` must expose run masks (see Section 7).
Implement run scope support (Section 7) before implementing `reliability.py`.

**Fix in `brainsig`:** Add `src/brainsig/reliability.py` with `compute_run_icc()` and
`compute_timepoint_icc()`. Add `pyrelimri>=2.2.3` as optional dependency in
`pyproject.toml` under a `[reliability]` extra.

---

## 7. Run Scopes

`brain_signature` explicitly scopes brain features by run (run1 / run2 / both) to
support test-retest reliability analysis:

```python
scope_keys = {"both": "_all_", "run1": "_run1_", "run2": "_run2_"}
```

`brainsig` has no concept of run scopes — all features are treated uniformly.

This is not needed for the core neural signature results (which use the `"both"` scope),
but is a prerequisite for `compute_run_icc()` in Section 6.

**Fix in `brainsig`:** Add `run_col: str | None = None` to `NeuralSignatureDataset`
to mark which observations belong to which run. When provided, expose `run1_mask`
and `run2_mask` boolean index arrays on the dataset object so `reliability.py`
can partition predictions by run.

---

## 8. Summary of Required Changes to `brainsig`

### Priority 1 — Critical (affects model outputs)

| File | Change |
|------|--------|
| `src/brainsig/residualizer.py` | **New:** `LinearResidualizer` sklearn transformer |
| `src/brainsig/neural_dataset.py` | `StandardScaler` → `RobustScaler`; `drop="if_binary"`; add `covariate_cols` param; apply `LinearResidualizer` on DataFrame before `ColumnTransformer` |
| `src/brainsig/dataset.py` | `StandardScaler` → `RobustScaler`; `drop="if_binary"` |

### Priority 2a — Required for Haufe (implement as a unit)

| File | Change |
|------|--------|
| `src/brainsig/model.py` | Add `get_best_cv_model(metric)` to retrieve best outer-fold estimator from `cv_results_` |
| `src/brainsig/model.py` | Add `_haufe_transform(X, y_pred, chunk_size)` static method |
| `src/brainsig/model.py` | Add `compute_haufe_features(dataset)` to orchestrate best-fold selection → Haufe computation |

### Priority 3 — Reliability analysis (implement in order: 7 then 6)

| File | Change |
|------|--------|
| `src/brainsig/neural_dataset.py` | Add `run_col` param; expose `run1_mask` / `run2_mask` |
| `src/brainsig/reliability.py` | **New:** `compute_run_icc()`, `compute_timepoint_icc()` |
| `pyproject.toml` | Add `reliability = ["pyrelimri>=2.2.3"]` optional dep group |

---

## 9. What Stays the Same

- ElasticNet logistic regression with identical hyperparameter space
- Binary classification of condition1 vs condition0
- Subject-level train/test splitting (80/20, `random_state=42`)
- Neural signature score as `P(y=1|cond1) - P(y=1|cond0)`
- Missing data handling (drop high-missingness columns, drop subjects with any NaN)
- Model serialization via joblib

---

## 10. Verification Strategy

After implementing the Priority 1 and Priority 2a changes, reproduce the N-Back
`2b-0b` contrast at baseline as a reference case:

1. Load the same ABCD data used by `brain_signature` (processed parquet files)
2. Construct `NeuralSignatureDataset` with the same covariate columns that
   `brain_signature` residualizes (age, sex, site, scanner, MID/SST/N-Back motion)
3. Fit `NeuralSignature` and call `compute_haufe_features()`
4. Compare output to the saved neural signature CSV written by `brain_signature`
   (`models/neuralsig/` directory)
5. Expected: Pearson *r* > 0.95 for Haufe maps; *r* > 0.90 for per-subject
   signature scores (residual difference due to CV strategy)
