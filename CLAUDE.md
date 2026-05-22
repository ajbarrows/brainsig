# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**brainsig** is a Python package for computing neural signatures from task-based fMRI data. It discriminates between two task conditions using elastic net regularized logistic regression and computes per-subject signature scores as the difference in predicted probabilities.

## Development Commands

This project uses **Hatch** for environment management and **UV** as the installer (see `pyproject.toml` for all environment definitions).

### Testing

```bash
hatch run +py=3.10 test:run                                              # run full test suite
hatch run +py=3.10 test:run tests/unit/test_model.py                     # run a single test file
hatch run +py=3.10 test:run -k "test_fit"                                # run tests matching a name
hatch run +py=3.10 test:run --cov=brainsig --cov-report=term-missing     # with coverage
```

### Linting and Formatting

```bash
hatch run style:check      # lint (ruff) + docstring check (pydoclint)
hatch run style:format     # auto-format with ruff
hatch run style:code       # ruff only
hatch run style:docstrings # pydoclint only
```

### Documentation

```bash
hatch run docs:build   # build HTML docs (Sphinx)
hatch run docs:serve   # live-reload docs server (watches src/brainsig)
```

### Other

```bash
hatch run build:check   # build wheel/sdist and verify with twine
hatch run audit:check   # dependency security audit with pip-audit
```

## Architecture

The package has three modules in `src/brainsig/`, used in sequence:

### 1. `dataset.py` — `Dataset`
Generic tabular data preprocessor. Drops columns exceeding a missing-data threshold, scales numeric features, one-hot encodes categoricals, and splits into train/test sets. Builds a scikit-learn `ColumnTransformer` pipeline internally; exposes `X_train`, `X_test`, `y_train`, `y_test` as arrays.

### 2. `neural_dataset.py` — `NeuralSignatureDataset`
Wraps two DataFrames (condition 1 = positive class, condition 0 = negative class) for paired-condition fMRI analysis. Delegates preprocessing to `Dataset` but enforces subject-level train/test splits to prevent data leakage — subjects in the test set are held out from both conditions together.

### 3. `model.py` — `ElasticNetClassifier` + `NeuralSignature`
- **`ElasticNetClassifier`**: Multi-target wrapper around scikit-learn `LogisticRegressionCV` with elastic net penalty. Supports nested cross-validation (inner CV for Cs/l1_ratios hyperparameter search, outer CV for evaluation).
- **`NeuralSignature`**: Subclasses `ElasticNetClassifier` for the binary fMRI use case. Computes signature scores as `P(condition=1 | fMRI_condition1) - P(condition=1 | fMRI_condition0)`. Adds `get_roc_curve()`, save/load via `joblib`, and defaults `scoring='roc_auc'`.

### Typical workflow

```python
ds = NeuralSignatureDataset(df_condition1, df_condition0, target_col="subject_id")
sig = NeuralSignature()
sig.fit(ds)
scores = sig.compute_signature_scores(ds)
cv_results = sig.cross_validate(ds)
```

## Code Style

- **Ruff** with all rules enabled; line length 88. Per-file ignores are defined in `pyproject.toml` for `__init__.py`, tests, and scikit-learn convention overrides.
- **Docstrings**: NumPy style, validated by `pydoclint`. Parameters, types, and return sections are required.
- Pre-commit hooks enforce formatting before commits (`.pre-commit-config.yaml`).
