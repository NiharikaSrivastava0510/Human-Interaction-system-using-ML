"""
Tests for data cleaning module.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.cleaner import clean_dataset, get_class_distribution


class TestCleaner(unittest.TestCase):
    """Test cases for data cleaning."""
    
    def setUp(self):
        """Create sample test data."""
        self.df = pd.DataFrame({
            'back_x': np.random.randn(1000),
            'back_y': np.random.randn(1000),
            'back_z': np.random.randn(1000),
            'thigh_x': np.random.randn(1000),
            'thigh_y': np.random.randn(1000),
            'thigh_z': np.random.randn(1000),
            'label': np.random.choice([0, 1, 2, 3, 4, 5, 13, 14], 1000),
        })
    
    def test_drop_cycling(self):
        """Test that cycling labels are dropped."""
        df_clean = clean_dataset(self.df, drop_cycling=True)
        cycling_labels = [13, 14, 130, 140]
        assert not any(df_clean['label'].isin(cycling_labels))
    
    def test_class_distribution(self):
        """Test class distribution calculation."""
        dist = get_class_distribution(self.df)
        assert isinstance(dist, dict)
        assert 'counts' in dist
        assert 'total' in dist


if __name__ == '__main__':
    unittest.main()
