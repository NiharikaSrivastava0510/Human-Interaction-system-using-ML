"""
Tests for feature extraction module.
"""

import unittest
import numpy as np
from src.features.enmo import calculate_enmo, calculate_enmo_from_window
from src.features.builder import extract_features_enmo, extract_refined_features
from src.features.statistical import extract_statistical_features


class TestFeatures(unittest.TestCase):
    """Test cases for feature extraction."""

    def setUp(self):
        """Create sample test data."""
        self.window = np.random.randn(100, 6)
        self.windows = np.random.randn(10, 100, 6)  # 10 windows

    def test_enmo_calculation(self):
        """Test ENMO calculation with 3 separate axis arrays."""
        x, y, z = self.window[:, 0], self.window[:, 1], self.window[:, 2]
        enmo = calculate_enmo(x, y, z)

        assert enmo.shape[0] == self.window.shape[0]
        assert (enmo >= 0).all()  # ENMO should be non-negative

    def test_enmo_from_window(self):
        """Test ENMO calculation from full window."""
        back_enmo, thigh_enmo = calculate_enmo_from_window(self.window)

        assert back_enmo.shape[0] == 100
        assert thigh_enmo.shape[0] == 100
        assert (back_enmo >= 0).all()
        assert (thigh_enmo >= 0).all()

    def test_extract_features_enmo(self):
        """Test 10-feature ENMO baseline extraction."""
        features = extract_features_enmo(self.windows)

        assert features.shape == (10, 10)  # 10 windows, 10 features each

    def test_extract_refined_features(self):
        """Test 16-feature refined extraction."""
        features = extract_refined_features(self.windows)

        assert features.shape == (10, 16)  # 10 windows, 16 features each

    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = extract_statistical_features(self.window)

        assert isinstance(features, dict)
        assert 'back_x_mean' in features
        assert 'total_energy' in features


if __name__ == '__main__':
    unittest.main()
