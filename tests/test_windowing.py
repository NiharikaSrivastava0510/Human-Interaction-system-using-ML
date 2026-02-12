"""
Tests for windowing module.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.windowing import create_sliding_windows, validate_windows


class TestWindowing(unittest.TestCase):
    """Test cases for sliding window creation."""
    
    def setUp(self):
        """Create sample test data."""
        self.df = pd.DataFrame({
            'back_x': np.random.randn(1000),
            'back_y': np.random.randn(1000),
            'back_z': np.random.randn(1000),
            'thigh_x': np.random.randn(1000),
            'thigh_y': np.random.randn(1000),
            'thigh_z': np.random.randn(1000),
            'label': np.random.choice([0, 1, 2], 1000),
        })
    
    def test_create_windows(self):
        """Test window creation."""
        X, y = create_sliding_windows(self.df, window_size=100, overlap=50)
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 100
        assert X.shape[2] == 6
    
    def test_validate_windows(self):
        """Test window validation."""
        X, y = create_sliding_windows(self.df, window_size=100, overlap=50)
        assert validate_windows(X, y)


if __name__ == '__main__':
    unittest.main()
