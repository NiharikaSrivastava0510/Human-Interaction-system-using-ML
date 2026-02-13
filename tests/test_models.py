"""
Tests for model modules.
"""

import unittest
import numpy as np
from src.models.gmm import GMMActivityModel
from src.models.random_forest import RandomForestActivityModel


class TestModels(unittest.TestCase):
    """Test cases for models."""

    def setUp(self):
        """Create sample test data."""
        self.X_train = np.random.randn(200, 16)
        self.y_train = np.random.randint(1, 10, 200)  # Labels 1-9 matching notebook
        self.X_test = np.random.randn(50, 16)

    def test_gmm_fit_predict(self):
        """Test GMM fitting and prediction."""
        model = GMMActivityModel(n_components=7)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        assert y_pred.shape[0] == self.X_test.shape[0]

    def test_gmm_predict_mapped(self):
        """Test GMM cluster-to-label mapping."""
        model = GMMActivityModel(n_components=7)
        model.fit(self.X_train, self.y_train)

        y_mapped = model.predict_mapped(self.X_test)
        assert y_mapped.shape[0] == self.X_test.shape[0]

    def test_gmm_bic(self):
        """Test GMM BIC score computation."""
        model = GMMActivityModel(n_components=7)
        model.fit(self.X_train)

        bic = model.get_bic(self.X_train)
        assert isinstance(bic, float)

    def test_random_forest_fit_predict(self):
        """Test Random Forest fitting and prediction."""
        model = RandomForestActivityModel(n_estimators=10)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        assert y_pred.shape[0] == self.X_test.shape[0]

    def test_random_forest_finetuned(self):
        """Test fine-tuned RF creation."""
        model = RandomForestActivityModel.create_finetuned()
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        assert y_pred.shape[0] == self.X_test.shape[0]

    def test_random_forest_feature_importance(self):
        """Test RF feature importance."""
        model = RandomForestActivityModel(n_estimators=10)
        model.fit(self.X_train, self.y_train)

        importances = model.get_feature_importance()
        assert importances.shape[0] == self.X_train.shape[1]


if __name__ == '__main__':
    unittest.main()
