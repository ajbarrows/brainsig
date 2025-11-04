"""
A module that creates the dataset object.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Dataset:
   def __init__(self, df: pd.DataFrame, target: str, missing_threshold=0.5, preprocessor=None, 
                test_size=0.2, random_state=None, verbose=True):
       
       self.original_df = df.copy()
       self.target = target
       
       # Clean missing data
       cleaned_df, self.dropped_summary = self._drop_missing_data(
           df, missing_threshold, verbose
       )
       
       # Create sklearn dataset
       self._create_sklearn_dataset(
           cleaned_df, target, preprocessor, test_size, random_state
       )
   
   def _drop_missing_data(self, df, missing_threshold, verbose):
       original_shape = df.shape
       dropped_summary = {
           'all_missing_cols': [],
           'high_missing_cols': [],
           'rows_dropped': 0
       }
       
       # Drop completely missing columns
       all_missing = df.columns[df.isnull().all()]
       df_clean = df.drop(columns=all_missing)
       dropped_summary['all_missing_cols'] = list(all_missing)
       
       # Drop columns above threshold
       missing_pct = df_clean.isnull().mean()
       high_missing = missing_pct[missing_pct > missing_threshold].index
       df_clean = df_clean.drop(columns=high_missing)
       dropped_summary['high_missing_cols'] = list(high_missing)
       
       # Drop rows with missing values
       rows_before = len(df_clean)
       df_clean = df_clean.dropna()
       dropped_summary['rows_dropped'] = rows_before - len(df_clean)
       
       if verbose:
           print(f"Original shape: {original_shape}")
           print(f"Final shape: {df_clean.shape}")
           if dropped_summary['all_missing_cols']:
               print(f"All-missing columns dropped: {dropped_summary['all_missing_cols']}")
           if dropped_summary['high_missing_cols']:
               print(f"High-missing columns dropped: {dropped_summary['high_missing_cols']}")
           print(f"Rows dropped: {dropped_summary['rows_dropped']}")
       
       return df_clean, dropped_summary

   def _create_sklearn_dataset(self, df, target, preprocessor, test_size, random_state):
        # Handle targets
        if isinstance(target, str):
            targets = [target]
        else:
            targets = list(target)
            
        X = df.drop(columns=targets)
        y = df[targets]

        # Extract target labels for categorical targets (preserve pd.Categorical order)
        self.target_labels = {}
        for col in targets:
            if df[col].dtype.name == 'category':
                # Use the explicit categories you set with pd.Categorical
                self.target_labels[col] = df[col].cat.categories.tolist()
            elif df[col].dtype == 'object':
                self.target_labels[col] = df[col].unique().tolist()
            else:
                self.target_labels[col] = None

        # Default preprocessor
        if preprocessor is None:
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(exclude=[np.number]).columns
            
            # For categorical features, preserve the pd.Categorical order
      # Preserve categorical order for all categorical columns
        categorical_categories = []
        for col in categorical_features:
            if X[col].dtype.name == 'category':
                categorical_categories.append(X[col].cat.categories.tolist())
            else:
                categorical_categories.append(sorted(X[col].unique()))

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                                categories=categorical_categories, handle_unknown='ignore'), categorical_features)
        ])

        self.preprocessor = preprocessor
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # Train/test split
        if test_size > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y.values, test_size=test_size, random_state=random_state
            )
        else:
            self.X = X_processed
            self.y = y.values