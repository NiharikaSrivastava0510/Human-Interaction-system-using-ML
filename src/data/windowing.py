"""
Sliding window segmentation for temporal feature extraction.

Provides two windowing approaches matching the notebook:
1. get_windows() - returns (X, y) arrays from raw sensor data
2. get_windows_with_subjects() - returns (X, y, groups) for GroupKFold CV
3. create_sliding_windows() - feature-based windowing with statistical extraction
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

from src.utils.config import WINDOW_SIZE, STEP_SIZE, SENSOR_AXES


def get_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    sensor_cols: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows of raw sensor data with mode-based labelling.

    Matches the notebook's get_windows() function used by all 3 models.

    Args:
        df: Input DataFrame with sensor columns and 'label'
        window_size: Number of samples per window (default: 100 = 2 seconds)
        step_size: Step between windows (default: 50 = 1 second overlap)
        sensor_cols: Sensor column names

    Returns:
        Tuple of (X, y) where:
        - X has shape (N, window_size, 6)
        - y has shape (N,) with mode label per window
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_AXES

    data = df[sensor_cols].values
    labels = df['label'].values

    X, y = [], []
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i + window_size])
        y.append(stats.mode(labels[i:i + window_size], keepdims=False)[0])

    return np.array(X), np.array(y)


def get_windows_with_subjects(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    sensor_cols: list = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sliding windows with subject group IDs for GroupKFold CV.

    Matches the notebook's get_windows_with_subjects() function used for
    cross-validation to prevent data leakage across subjects.

    Args:
        df: Input DataFrame with sensor columns, 'label', and 'subject_id'
        window_size: Number of samples per window
        step_size: Step between windows
        sensor_cols: Sensor column names

    Returns:
        Tuple of (X, y, groups) where:
        - X has shape (N, window_size, 6)
        - y has shape (N,) with mode label per window
        - groups has shape (N,) with subject_id per window
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_AXES

    data = df[sensor_cols].values
    labels = df['label'].values
    subjects = df['subject_id'].values

    X, y, groups = [], [], []
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i + window_size])
        y.append(stats.mode(labels[i:i + window_size], keepdims=False)[0])
        groups.append(subjects[i])

    return np.array(X), np.array(y), np.array(groups)


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    overlap: int = STEP_SIZE,
    sensor_cols: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time-series sensor data (DataFrame-based).

    Args:
        df: Input DataFrame with sensor readings and labels
        window_size: Number of samples per window
        overlap: Number of overlapping samples between windows
        sensor_cols: List of sensor column names

    Returns:
        Tuple of (X, y) where X is (N, window_size, 6) and y is (N,)
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_AXES

    X_list = []
    y_list = []

    step_size = window_size - overlap

    for idx in range(0, len(df) - window_size, step_size):
        window = df.iloc[idx:idx + window_size]

        if window.isnull().any().any():
            continue

        X_window = window[sensor_cols].values
        y_window = window['label'].mode()[0]

        X_list.append(X_window)
        y_list.append(y_window)

    X = np.array(X_list)
    y = np.array(y_list)

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
    """Save windows to compressed NumPy format."""
    np.savez_compressed(filepath, X=X, y=y)


def load_windows(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load windows from compressed NumPy format."""
    data = np.load(filepath)
    return data['X'], data['y']
