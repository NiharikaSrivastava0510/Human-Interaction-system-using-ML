"""
Feature pipeline: build feature matrices from windows.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from .statistical import extract_statistical_features, extract_enmo_statistics
from .enmo import calculate_enmo


def build_feature_matrix(
    X_windows: np.ndarray,
    feature_type: str = 'statistical',
    include_enmo: bool = True
) -> np.ndarray:
    """
    Build feature matrix from windowed data.
    
    Args:
        X_windows: Windowed sensor data of shape (N, window_size, 6)
        feature_type: Type of features ('statistical', 'raw', 'combined')
        include_enmo: Whether to include ENMO features
        
    Returns:
        Feature matrix of shape (N, num_features)
    """
    num_windows = X_windows.shape[0]
    features_list = []
    
    sensor_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    for window_idx in range(num_windows):
        window = X_windows[window_idx]  # Shape: (window_size, 6)
        feature_dict = {}
        
        if feature_type in ['statistical', 'combined']:
            # Extract standard statistical features
            feature_dict.update(extract_statistical_features(window, sensor_cols))
        
        if include_enmo:
            # Calculate ENMO from acceleration
            back_accel = window[:, :3]  # back_x, back_y, back_z
            thigh_accel = window[:, 3:]  # thigh_x, thigh_y, thigh_z
            
            back_enmo = calculate_enmo(back_accel)
            thigh_enmo = calculate_enmo(thigh_accel)
            
            feature_dict.update(extract_enmo_statistics(back_enmo, 'back_enmo'))
            feature_dict.update(extract_enmo_statistics(thigh_enmo, 'thigh_enmo'))
        
        features_list.append(feature_dict)
    
    feature_df = pd.DataFrame(features_list)
    return feature_df.values


def get_feature_names(
    feature_type: str = 'statistical',
    include_enmo: bool = True
) -> List[str]:
    """
    Get list of feature names for a given configuration.
    
    Args:
        feature_type: Type of features
        include_enmo: Whether ENMO is included
        
    Returns:
        List of feature names
    """
    sensor_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    names = []
    
    if feature_type in ['statistical', 'combined']:
        for col in sensor_cols:
            names.extend([
                f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max',
                f'{col}_median', f'{col}_rms', f'{col}_energy'
            ])
        names.append('total_energy')
    
    if include_enmo:
        for prefix in ['back_enmo', 'thigh_enmo']:
            names.extend([
                f'{prefix}_mean', f'{prefix}_std', f'{prefix}_min',
                f'{prefix}_max', f'{prefix}_median', f'{prefix}_rms'
            ])
    
    return names
