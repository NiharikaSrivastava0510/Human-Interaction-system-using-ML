"""
Statistical feature extraction from sensor windows.
"""

import numpy as np
from typing import Dict, List


def extract_statistical_features(
    window: np.ndarray,
    sensor_cols: List[str] = None
) -> Dict[str, float]:
    """
    Extract statistical features from a sensor window.
    
    Features calculated:
    - Mean, standard deviation, min, max for each axis
    - RMS (Root Mean Square) for each axis
    - Energy (sum of squares) for each axis
    - Total energy across all axes
    
    Args:
        window: Sensor window of shape (window_size, num_axes)
        sensor_cols: Names of sensor columns for feature naming
        
    Returns:
        Dictionary of extracted features
    """
    if sensor_cols is None:
        sensor_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    features = {}
    
    for i, col in enumerate(sensor_cols):
        axis_data = window[:, i]
        
        features[f'{col}_mean'] = float(np.mean(axis_data))
        features[f'{col}_std'] = float(np.std(axis_data))
        features[f'{col}_min'] = float(np.min(axis_data))
        features[f'{col}_max'] = float(np.max(axis_data))
        features[f'{col}_median'] = float(np.median(axis_data))
        features[f'{col}_rms'] = float(np.sqrt(np.mean(axis_data**2)))
        features[f'{col}_energy'] = float(np.sum(axis_data**2))
    
    # Total energy across all axes
    features['total_energy'] = float(np.sum(window**2))
    
    return features


def extract_enmo_statistics(
    window: np.ndarray,
    feature_name: str = 'enmo'
) -> Dict[str, float]:
    """
    Extract statistical features from ENMO values.
    
    Args:
        window: ENMO window of shape (window_size,)
        feature_name: Name prefix for features
        
    Returns:
        Dictionary of ENMO statistics
    """
    return {
        f'{feature_name}_mean': float(np.mean(window)),
        f'{feature_name}_std': float(np.std(window)),
        f'{feature_name}_min': float(np.min(window)),
        f'{feature_name}_max': float(np.max(window)),
        f'{feature_name}_median': float(np.median(window)),
        f'{feature_name}_rms': float(np.sqrt(np.mean(window**2))),
    }


def extract_correlation_features(
    window: np.ndarray,
    sensor_pairs: List[tuple] = None
) -> Dict[str, float]:
    """
    Extract correlation features between sensor axes.
    
    Args:
        window: Sensor window of shape (window_size, num_axes)
        sensor_pairs: List of axis index pairs to correlate
        
    Returns:
        Dictionary of correlation features
    """
    if sensor_pairs is None:
        # All pairs including same axis
        sensor_pairs = [(i, j) for i in range(window.shape[1]) 
                       for j in range(i, window.shape[1])]
    
    features = {}
    for i, j in sensor_pairs:
        corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
        features[f'corr_{i}_{j}'] = float(corr) if not np.isnan(corr) else 0.0
    
    return features
