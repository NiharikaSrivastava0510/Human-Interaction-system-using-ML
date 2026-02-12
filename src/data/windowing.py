"""
Sliding window segmentation for temporal feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 100,
    overlap: int = 50,
    sensor_cols: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time-series sensor data.
    
    Args:
        df: Input DataFrame with sensor readings and labels
        window_size: Number of samples per window
        overlap: Number of overlapping samples between windows
        sensor_cols: List of sensor column names
        
    Returns:
        Tuple of (X, y) where X is (N, window_size, 6) and y is (N,)
    """
    if sensor_cols is None:
        sensor_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    X_list = []
    y_list = []
    
    step_size = window_size - overlap
    
    for idx in range(0, len(df) - window_size, step_size):
        window = df.iloc[idx:idx + window_size]
        
        # Skip if window contains NaN
        if window.isnull().any().any():
            continue
        
        # Extract sensor readings
        X_window = window[sensor_cols].values  # Shape: (window_size, 6)
        
        # Use most common label in window
        y_window = window['label'].mode()[0]
        
        X_list.append(X_window)
        y_list.append(y_window)
    
    X = np.array(X_list)  # Shape: (N, window_size, 6)
    y = np.array(y_list)  # Shape: (N,)
    
    return X, y


def validate_windows(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Validate window data format and integrity.
    
    Args:
        X: Feature windows
        y: Labels
        
    Returns:
        True if validation passes
    """
    assert len(X) == len(y), "X and y must have same length"
    assert X.ndim == 3, "X must be 3D array (N, window_size, channels)"
    assert X.shape[2] == 6, "Must have 6 sensor channels"
    assert y.min() >= 0, "Labels must be non-negative"
    
    return True


def save_windows(X: np.ndarray, y: np.ndarray, filepath: str):
    """
    Save windows to compressed NumPy format.
    
    Args:
        X: Feature windows
        y: Labels
        filepath: Output file path
    """
    np.savez_compressed(filepath, X=X, y=y)


def load_windows(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load windows from compressed NumPy format.
    
    Args:
        filepath: Input file path
        
    Returns:
        Tuple of (X, y)
    """
    data = np.load(filepath)
    return data['X'], data['y']
