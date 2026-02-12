"""
Gaussian Mixture Model for unsupervised activity classification.
"""

import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from .base import BaseActivityModel


class GMMActivityModel(BaseActivityModel):
    """Gaussian Mixture Model for activity classification."""
    
    def __init__(self, n_components: int = 7, **kwargs):
        """
        Initialize GMM model.
        
        Args:
            n_components: Number of mixture components
            **kwargs: Additional parameters for GaussianMixture
        """
        super().__init__()
        self.n_components = n_components
        self.gmm_params = kwargs
        self.model = GaussianMixture(n_components=n_components, **kwargs)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None, **kwargs):
        """
        Fit GMM to training data (unsupervised).
        
        Args:
            X_train: Training features of shape (N, num_features)
            y_train: Not used (unsupervised), kept for API consistency
            **kwargs: Additional fit parameters
        """
        self.model.fit(X_train)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get cluster membership probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix of shape (N, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Save model to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
