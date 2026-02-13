"""
Random Forest model for activity classification.

Matches the notebook's Random Forest pipeline:
- Baseline: 100 estimators, no max_depth limit
- Fine-tuned: 200 estimators, max_depth=20, min_samples_leaf=2
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from .base import BaseActivityModel

from src.utils.config import (
    RF_BASELINE_ESTIMATORS, RF_BASELINE_MAX_DEPTH,
    RF_FINETUNED_ESTIMATORS, RF_FINETUNED_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF, RF_RANDOM_STATE,
)


class RandomForestActivityModel(BaseActivityModel):
    """Random Forest classifier for activity recognition."""

    def __init__(
        self,
        n_estimators: int = RF_BASELINE_ESTIMATORS,
        max_depth: int = RF_BASELINE_MAX_DEPTH,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_leaf: Minimum samples required at a leaf node
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
            **kwargs
        )

    @classmethod
    def create_finetuned(cls):
        """
        Create a fine-tuned Random Forest matching the notebook's parameters.

        Returns:
            RandomForestActivityModel with fine-tuned hyperparameters
        """
        return cls(
            n_estimators=RF_FINETUNED_ESTIMATORS,
            max_depth=RF_FINETUNED_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
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
