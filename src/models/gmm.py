"""
Gaussian Mixture Model for unsupervised activity classification.

Matches the notebook's GMM pipeline:
- Baseline: n_components = number of unique classes, n_init=5
- Fine-tuned: BIC-based optimal k selection from [8, 10, 12, 16, 20]
- Cluster-to-label mapping using mode of true labels per cluster
"""

import numpy as np
import pickle
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .base import BaseActivityModel

from src.utils.config import (
    GMM_BASELINE_K, GMM_FINETUNED_K, GMM_COVARIANCE_TYPE,
    GMM_N_INIT, GMM_N_INIT_FINETUNE, GMM_RANDOM_STATE,
    GMM_BIC_SEARCH_RANGE,
)


class GMMActivityModel(BaseActivityModel):
    """Gaussian Mixture Model for activity classification."""

    def __init__(self, n_components: int = GMM_BASELINE_K, **kwargs):
        """
        Initialize GMM model.

        Args:
            n_components: Number of mixture components
            **kwargs: Additional parameters for GaussianMixture
        """
        super().__init__()
        self.n_components = n_components
        self.gmm_params = kwargs
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=kwargs.get('covariance_type', GMM_COVARIANCE_TYPE),
            random_state=kwargs.get('random_state', GMM_RANDOM_STATE),
            n_init=kwargs.get('n_init', GMM_N_INIT),
        )
        self.cluster_label_mapping = {}
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None, **kwargs):
        """
        Fit GMM to training data (unsupervised) and create cluster-to-label mapping.

        Args:
            X_train: Training features of shape (N, num_features)
            y_train: True labels for creating cluster-to-label mapping
            **kwargs: Additional fit parameters
        """
        self.model.fit(X_train)
        self.is_fitted = True

        if y_train is not None:
            train_clusters = self.model.predict(X_train)
            self.cluster_label_mapping = self._map_clusters_to_labels(
                train_clusters, y_train
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments.

        Args:
            X: Feature matrix

        Returns:
            Cluster assignments (raw cluster IDs if no mapping exists)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_mapped(self, X: np.ndarray) -> np.ndarray:
        """
        Predict activity labels using cluster-to-label mapping.

        Matches the notebook's approach of mapping GMM clusters to true labels
        using the mode of true labels within each cluster.

        Args:
            X: Feature matrix

        Returns:
            Mapped activity labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if not self.cluster_label_mapping:
            raise ValueError("No cluster-to-label mapping exists. "
                             "Call fit() with y_train to create mapping.")

        clusters = self.model.predict(X)
        return np.array([
            self.cluster_label_mapping.get(c, 0) for c in clusters
        ])

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

    def get_bic(self, X: np.ndarray) -> float:
        """
        Get Bayesian Information Criterion for the fitted model.

        Args:
            X: Feature matrix used for BIC calculation

        Returns:
            BIC score (lower is better)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing BIC")
        return self.model.bic(X)

    @staticmethod
    def find_optimal_k(
        X_train: np.ndarray,
        y_train: np.ndarray = None,
        k_range: list = None,
    ):
        """
        Find optimal number of components using BIC.

        Matches the notebook's fine-tuning approach that tests different
        cluster counts and selects the one with lowest BIC.

        Args:
            X_train: Training features (already scaled)
            y_train: True labels for mapping (optional)
            k_range: List of k values to try

        Returns:
            Fitted GMMActivityModel with optimal k
        """
        if k_range is None:
            k_range = GMM_BIC_SEARCH_RANGE

        best_model = None
        best_bic = np.inf

        for n_c in k_range:
            gmm = GMMActivityModel(
                n_components=n_c,
                covariance_type=GMM_COVARIANCE_TYPE,
                random_state=GMM_RANDOM_STATE,
                n_init=GMM_N_INIT_FINETUNE,
            )
            gmm.fit(X_train, y_train)
            bic = gmm.get_bic(X_train)
            print(f"   k={n_c}: BIC={bic:.0f}")

            if bic < best_bic:
                best_bic = bic
                best_model = gmm

        print(f"Selected Optimal k={best_model.n_components}")
        return best_model

    @staticmethod
    def _map_clusters_to_labels(clusters, true_labels):
        """
        Map cluster assignments to true labels using mode.

        Args:
            clusters: Cluster assignments
            true_labels: Ground truth labels

        Returns:
            Dictionary mapping cluster ID to most common true label
        """
        mapping = {}
        for i in np.unique(clusters):
            indices = np.where(clusters == i)
            if len(indices[0]) > 0:
                mode_label = stats.mode(true_labels[indices], keepdims=False)[0]
                mapping[i] = mode_label
        return mapping

    def save(self, filepath: str):
        """Save model and mapping to pickle file."""
        save_data = {
            'model': self.model,
            'mapping': self.cluster_label_mapping,
            'n_components': self.n_components,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str):
        """Load model and mapping from pickle file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        if isinstance(save_data, dict):
            self.model = save_data['model']
            self.cluster_label_mapping = save_data.get('mapping', {})
            self.n_components = save_data.get('n_components', self.model.n_components)
        else:
            self.model = save_data
        self.is_fitted = True
