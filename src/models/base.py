"""
Abstract base class for activity classification models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseActivityModel(ABC):
    """Abstract base class for activity classification models."""
    
    def __init__(self, num_classes: int = 7):
        """
        Initialize base model.
        
        Args:
            num_classes: Number of activity classes
        """
        self.num_classes = num_classes
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Fit the model to training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Model-specific parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities of shape (N, num_classes)
        """
        raise NotImplementedError("predict_proba not implemented for this model")
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model from file."""
        pass
