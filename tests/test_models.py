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
        self.y_train = np.random.randint(0, 7, 200)
        self.X_test = np.random.randn(50, 16)
    
    def test_gmm_fit_predict(self):
        """Test GMM fitting and prediction."""
        model = GMMActivityModel(n_components=7)
        model.fit(self.X_train)
        
        y_pred = model.predict(self.X_test)
        assert y_pred.shape[0] == self.X_test.shape[0]
    
    def test_random_forest_fit_predict(self):
        """Test Random Forest fitting and prediction."""
        model = RandomForestActivityModel(n_estimators=10)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        assert y_pred.shape[0] == self.X_test.shape[0]


if __name__ == '__main__':
    unittest.main()
