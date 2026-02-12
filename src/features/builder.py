"""
Feature pipeline: build feature matrices from windows.

Provides two feature extraction functions matching the notebook:
1. extract_features_enmo() - 10-feature ENMO baseline
2. extract_refined_features() - 16-feature set (raw means + stds + ENMO)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from .statistical import extract_statistical_features, extract_enmo_statistics
from .enmo import calculate_enmo, calculate_enmo_from_window


def extract_features_enmo(windows: np.ndarray) -> np.ndarray:
    """
    Extract 10-feature ENMO baseline from windowed data.

    Matches the notebook's extract_features_enmo function.
    Features per window: [back_enmo_mean, back_enmo_max, back_enmo_median,
    back_enmo_std, back_enmo_energy, thigh_enmo_mean, thigh_enmo_max,
    thigh_enmo_median, thigh_enmo_std, thigh_enmo_energy]

    Args:
        windows: Array of shape (N, window_size, 6)

    Returns:
        Feature matrix of shape (N, 10)
    """
    feature_list = []
    for w in windows:
        b_enmo, t_enmo = calculate_enmo_from_window(w)
        b_feats = [
            np.mean(b_enmo), np.max(b_enmo), np.median(b_enmo),
            np.std(b_enmo), np.sum(b_enmo**2)
        ]
        t_feats = [
            np.mean(t_enmo), np.max(t_enmo), np.median(t_enmo),
            np.std(t_enmo), np.sum(t_enmo**2)
        ]
        feature_list.append(b_feats + t_feats)
    return np.array(feature_list)


def extract_refined_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract 16-feature refined set from windowed data.

    Matches the notebook's extract_refined_features function.
    Features: 6 raw means (posture/orientation) + 6 raw stds (intensity)
    + 4 ENMO features (pure motion).

    Args:
        windows: Array of shape (N, window_size, 6)

    Returns:
        Feature matrix of shape (N, 16)
    """
    features = []
    for w in windows:
        raw_means = np.mean(w, axis=0)       # 6 features: posture/orientation
        raw_stds = np.std(w, axis=0)          # 6 features: general intensity

        mag_b = np.sqrt(np.sum(w[:, 0:3]**2, axis=1))
        mag_t = np.sqrt(np.sum(w[:, 3:6]**2, axis=1))
        enmo_b = np.maximum(0, mag_b - 1)
        enmo_t = np.maximum(0, mag_t - 1)
        enmo_feats = [np.mean(enmo_b), np.max(enmo_b),
                      np.mean(enmo_t), np.max(enmo_t)]  # 4 features

        features.append(np.concatenate([raw_means, raw_stds, enmo_feats]))
    return np.array(features)


# Alias for backward compatibility
extract_optimized_features = extract_refined_features


def build_feature_matrix(
    X_windows: np.ndarray,
    feature_type: str = 'statistical',
    include_enmo: bool = True
) -> np.ndarray:
    """
    Build feature matrix from windowed data.

    Args:
        X_windows: Windowed sensor data of shape (N, window_size, 6)
        feature_type: Type of features ('statistical', 'raw', 'combined',
                      'enmo_baseline', 'refined')
        include_enmo: Whether to include ENMO features

    Returns:
        Feature matrix of shape (N, num_features)
    """
    if feature_type == 'enmo_baseline':
        return extract_features_enmo(X_windows)
    elif feature_type == 'refined':
        return extract_refined_features(X_windows)

    num_windows = X_windows.shape[0]
    features_list = []

    sensor_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

    for window_idx in range(num_windows):
        window = X_windows[window_idx]
        feature_dict = {}

        if feature_type in ['statistical', 'combined']:
            feature_dict.update(extract_statistical_features(window, sensor_cols))

        if include_enmo:
            back_enmo = calculate_enmo(window[:, 0], window[:, 1], window[:, 2])
            thigh_enmo = calculate_enmo(window[:, 3], window[:, 4], window[:, 5])

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
    if feature_type == 'enmo_baseline':
        return [
            'enmo_b_mean', 'enmo_b_max', 'enmo_b_median', 'enmo_b_std', 'enmo_b_energy',
            'enmo_t_mean', 'enmo_t_max', 'enmo_t_median', 'enmo_t_std', 'enmo_t_energy',
        ]
    elif feature_type == 'refined':
        return [
            'bx_mean', 'by_mean', 'bz_mean', 'tx_mean', 'ty_mean', 'tz_mean',
            'bx_std', 'by_std', 'bz_std', 'tx_std', 'ty_std', 'tz_std',
            'enmo_b_mean', 'enmo_b_max', 'enmo_t_mean', 'enmo_t_max',
        ]

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
