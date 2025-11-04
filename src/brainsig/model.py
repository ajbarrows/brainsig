"""
A module to fit elastic net logistic regression, the model behind the neural signature.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split

from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV

class ElasticNetClassifier():
    def __init__(self, inner_folds: int=5, outer_folds: int=5,
                 inner_scoring='roc_auc_ovr',
                 outer_scoring = {
                     'acc': 'accuracy',
                     'f1': 'f1_macro',
                     'auc': 'roc_auc_ovr',
                 },
                 Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                 max_iter=1000,
                 n_jobs=-1,
                 random_state=42):
        self.inner_scoring=inner_scoring
        self.outer_scoring=outer_scoring
        self.inner_folds=inner_folds
        self.outer_folds=outer_folds
        self.Cs=Cs
        self.l1_ratios=l1_ratios
        self.max_iter=max_iter
        self.n_jobs=n_jobs
        self.random_state=random_state
        self.inner_cv=self.build_cv_scheme(self.inner_folds, self.random_state)
        self.outer_cv=self.build_cv_scheme(self.outer_folds, self.random_state + 50)
        self.dataset = None
        
        # Initialize containers for multiple targets
        self.models = {}
        self.cv_results = {}
        self.target_names = []

    def define_model(self, random_state=42):
        return LogisticRegressionCV(
            cv=self.inner_cv,
            penalty='elasticnet',
            solver='saga',
            scoring=self.inner_scoring,
            Cs=self.Cs,
            l1_ratios=self.l1_ratios,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=random_state
        )

    def build_cv_scheme(self, n_splits=5, random_state=42):
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def fit_model(self, dataset, keep_dataset=True) -> None:
        """Fit models for each target."""
        self.target_names = list(dataset.target_labels.keys())
        n_targets = len(self.target_names)

        if keep_dataset:
            self.dataset = dataset
        
        for i, target_name in enumerate(self.target_names):
            y_target = dataset.y_train[:, i] if n_targets > 1 else dataset.y_train.ravel()
            
            model = self.define_model(self.random_state + 100)
            model.fit(dataset.X_train, y_target)
            self.models[target_name] = model

    def predict(self, dataset):
        """Make predictions on test set for all targets."""
        self.predictions = {}
        
        for target_name, model in self.models.items():
            y_pred = model.predict(dataset.X_test)
            y_pred_proba = model.predict_proba(dataset.X_test) if hasattr(model, 'predict_proba') else None
            
            self.predictions[target_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_true': dataset.y_test[:, list(dataset.target_labels.keys()).index(target_name)] 
                            if len(dataset.target_labels) > 1 else dataset.y_test.ravel()
            }
        
        return self.predictions

    def cross_validate(self, dataset) -> dict:
        """Cross-validate each target separately."""
        self.target_names = list(dataset.target_labels.keys())
        n_targets = len(self.target_names)
        
        for i, target_name in enumerate(self.target_names):
            y_target = dataset.y_train[:, i] if n_targets > 1 else dataset.y_train.ravel()
            
            model = self.define_model()
            cv_result = cross_validate(
                model,
                dataset.X_train, y_target,
                cv=self.outer_cv,
                scoring=self.outer_scoring,
                return_indices=True,
                return_estimator=True,
                return_train_score=True
            )
            self.cv_results[target_name] = cv_result
        
        return self.cv_results

    
    def get_cv_coefs(self, dataset, exponentiate=False):
        """Get coefficients for all targets with proper labeling."""
        all_coefs = pd.DataFrame()
        
        # Clean feature names more carefully
        clean_feature_names = []
        for name in dataset.feature_names:
            if name.startswith('num__'):
                clean_feature_names.append(name.replace('num__', ''))
            elif name.startswith('cat__'):
                clean_feature_names.append(name.replace('cat__', ''))
            else:
                clean_feature_names.append(name)
        
        for target_name, cv_result in self.cv_results.items():
            models = cv_result['estimator']
            target_labels = dataset.target_labels[target_name]
            
            for i, model in enumerate(models):
                coef_matrix = model.coef_
                
                for class_idx, class_label in enumerate(target_labels):
                    coef_row = coef_matrix[class_idx] if coef_matrix.ndim > 1 else coef_matrix
                    
                    coef_df = pd.DataFrame([coef_row], columns=clean_feature_names)
                    coef_df = coef_df.assign(
                        cv_fold=i,
                        target_variable=target_name,
                        target=class_label
                    ).set_index(['cv_fold', 'target_variable', 'target'])
                    
                    all_coefs = pd.concat([all_coefs, coef_df])
        
        if exponentiate:
            numeric_cols = all_coefs.select_dtypes(include=[np.number]).columns
            all_coefs[numeric_cols] = np.exp(all_coefs[numeric_cols])
        
        return all_coefs
    
    def get_model_scores(self, dataset=None) -> pd.DataFrame:
        """Get all available model scores: CV scores and/or fitted model scores."""
        all_scores = pd.DataFrame()
        
        # Add CV scores if available
        if hasattr(self, 'cv_results') and self.cv_results:
            for target_name, cv_result in self.cv_results.items():
                # Get the class labels for this target from dataset
                target_labels = None
                if dataset and hasattr(dataset, 'target_labels'):
                    target_labels = dataset.target_labels.get(target_name)
                
                for k, v in cv_result.items():
                    splits = k.split('_')
                    if splits[0] in ['train', 'test']:
                        # If this is a multi-class problem, scores are per class
                        if target_labels and len(target_labels) > 2:  # Multi-class
                            for i, class_label in enumerate(target_labels):
                                cv_scores = pd.DataFrame({'value': v[:, i] if v.ndim > 1 else v}).assign(
                                    partition=splits[0],
                                    metric=splits[1],
                                    target=target_name,
                                    score_type='cv',
                                    **{'target_label': class_label}
                                ).reset_index(names=['cv_fold'])
                                all_scores = pd.concat([all_scores, cv_scores])
                        else:  # Binary or unknown classification
                            cv_scores = pd.DataFrame({'value': v}).assign(
                                partition=splits[0],
                                metric=splits[1],
                                target=target_name,
                                score_type='cv',
                                **{'target_label': None}
                            ).reset_index(names=['cv_fold'])
                            all_scores = pd.concat([all_scores, cv_scores])
            
            # Add fitted model scores if available (rest remains the same)
            if hasattr(self, 'predictions') and self.predictions:
                
                for target_name, pred_data in self.predictions.items():
                    y_true, y_pred = pred_data['y_true'], pred_data['y_pred']
                    y_pred_proba = pred_data['y_pred_proba']
                    
                    unique_classes = np.unique(y_true)
                    f1_per_class = f1_score(y_true, y_pred, average=None)
                    
                    if y_pred_proba is not None:
                        auc_per_class = roc_auc_score(y_true, y_pred_proba, 
                                                    multi_class='ovr', average=None)
                    
                    acc_score = accuracy_score(y_true, y_pred)
                    
                    for i, class_label in enumerate(unique_classes):
                        class_scores = {
                            'acc': acc_score,
                            'f1': f1_per_class[i]
                        }
                        
                        if y_pred_proba is not None:
                            class_scores['auc'] = auc_per_class[i]
                        
                        for metric, score in class_scores.items():
                            fitted_scores = pd.DataFrame({
                                'cv_fold': [0],
                                'value': [score],
                                'partition': ['test_holdout'],
                                'metric': [metric],
                                'target': [target_name],
                                'target_label': [class_label],
                                'score_type': ['fitted']
                            })
                            all_scores = pd.concat([all_scores, fitted_scores])
            
            return all_scores
