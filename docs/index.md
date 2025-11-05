# brainsig

## Overview

A package for computing "neural signatures" from fMRI data.

## Features

- **Dataset Management**: Automated preprocessing of fMRI data with missing data handling
- **Elastic Net Classification**: Binary classification with nested cross-validation
- **Neural Signature Computation**: Compute signature scores from task-based fMRI data

## Quick Start

```python
from brainsig import Dataset, NeuralSignature
import pandas as pd

# Load your fMRI data
df = pd.read_csv("fmri_data.csv")

# Create dataset with automatic preprocessing
dataset = Dataset(df, target="condition", test_size=0.2, random_state=42)

# Fit neural signature model
neural_sig = NeuralSignature(random_state=42)
neural_sig.fit(dataset)

# Compute signature scores
scores = neural_sig.compute_signature_scores(condition1_data, condition0_data)
```

## API Reference

The API documentation is automatically generated from the source code docstrings. Navigate to the "brainsig" section in the sidebar to explore the available classes and functions.

## Copyright

- Copyright © 2025 Tony Barrows.
- Free software distributed under the MIT License.
