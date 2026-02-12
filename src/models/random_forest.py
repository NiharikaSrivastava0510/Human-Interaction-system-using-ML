"""
Random Forest model for activity classification.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from .base import BaseActivityModel


class RandomForestActivityModel(BaseActivityModel):
    """Random Forest classifier for activity recognition."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        **kwargs
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Fit Random Forest to training data.
        
        Args:
            X_train: Training features of shape (N, num_features)
            y_train: Training labels of shape (N,)
            **kwargs: Additional fit parameters (ignored)
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix of shape (N, num_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importances")
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """Save model to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
