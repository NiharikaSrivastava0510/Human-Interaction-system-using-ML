"""
Tests for feature extraction module.
"""

import unittest
import numpy as np
from src.features.enmo import calculate_enmo
from src.features.statistical import extract_statistical_features


class TestFeatures(unittest.TestCase):
    """Test cases for feature extraction."""
    
    def setUp(self):
        """Create sample test data."""
        self.window = np.random.randn(100, 6)
        self.acceleration = np.random.randn(100, 3)
    
    def test_enmo_calculation(self):
        """Test ENMO calculation."""
        enmo = calculate_enmo(self.acceleration)
        
        assert enmo.shape[0] == self.acceleration.shape[0]
        assert (enmo >= 0).all()  # ENMO should be non-negative
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = extract_statistical_features(self.window)
        
        assert isinstance(features, dict)
        assert 'back_x_mean' in features
        assert 'total_energy' in features


if __name__ == '__main__':
    unittest.main()
