"""
Random Forest Model Implementation
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


class RandomForestModel:
    """Random Forest classifier wrapper with hyperparameter tuning support."""
    
    def __init__(self, n_estimators=300, max_depth=15, min_samples_split=2, 
                 min_samples_leaf=2, max_features='sqrt', random_state=42):
        """
        Initialize Random Forest model.
        
        Parameters:
        -----------
        n_estimators : int, default=200
            Number of trees in the forest
        max_depth : int or None, default=15
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        max_features : str or int, default='sqrt'
            Number of features for best split
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        
    def build_model(self):
        """Build and return the Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight='balanced',  # Helps with imbalanced datasets
            n_jobs=-1
        )
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the model on the given data.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        """
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions on the given data."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Returns:
        --------
        dict : Dictionary containing accuracy, precision, recall, f1, and confusion matrix
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
             'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC-AUC calculation with multi-class support
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_test, y_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
        except ValueError:
            metrics['roc_auc'] = None
        
        return metrics
    
    def get_feature_importance(self):
        """Return feature importances from the trained model."""
        if self.model is not None:
            return self.model.feature_importances_
        return None
    
    @staticmethod
    def get_param_grid():
        """Return hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    @staticmethod
    def get_param_info():
        """Return information about hyperparameters for UI."""
        return {
            'n_estimators': {
                'type': 'int',
                'min': 10,
                'max': 500,
                'default': 300,
                'description': 'Number of trees in the forest'
            },
            'max_depth': {
                'type': 'int',
                'min': 1,
                'max': 50,
                'default': 15,
                'description': 'Maximum depth of trees'
            },
            'min_samples_split': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 2,
                'description': 'Minimum samples to split a node'
            },
            'min_samples_leaf': {
                'type': 'int',
                'min': 1,
                'max': 20,
                'default': 2,
                'description': 'Minimum samples at a leaf node'
            },
            'max_features': {
                'type': 'select',
                'options': ['sqrt', 'log2'],
                'default': 'sqrt',
                'description': 'Number of features for best split'
            }
        }

