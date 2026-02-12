"""
ENMO (Euclidean Norm Minus One) calculation.

ENMO removes the static gravity component from accelerometer data,
isolating the dynamic movement signal.

Formula: ENMO = max(0, sqrt(x² + y² + z²) - 1)

Note: While useful for capturing movement intensity, ENMO removes
gravity orientation information that is critical for distinguishing
static postures (Sitting vs Standing). The fine-tuned feature set
supplements ENMO with raw axis means to recover this information.
"""

import numpy as np


def calculate_enmo(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate ENMO from 3-axis accelerometer data.

    Args:
        x, y, z: Acceleration values for each axis (in g)

    Returns:
        ENMO values (clipped at 0)
    """
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    enmo = np.maximum(0, magnitude - 1)
    return enmo


def calculate_enmo_from_window(window: np.ndarray):
    """
    Calculate ENMO for back and thigh sensors from a single window.

    Matches the notebook's calculate_enmo function that takes a full
    window array of shape (window_size, 6).

    Args:
        window: Array of shape (window_size, 6) with columns
                [back_x, back_y, back_z, thigh_x, thigh_y, thigh_z]

    Returns:
        Tuple of (back_enmo, thigh_enmo) arrays
    """
    back_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
    thigh_mag = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
    back_enmo = np.maximum(0, back_mag - 1)
    thigh_enmo = np.maximum(0, thigh_mag - 1)
    return back_enmo, thigh_enmo


def compute_enmo_for_window(window: np.ndarray) -> dict:
    """
    Compute ENMO-based features for a single window.

    Args:
        window: Array of shape (window_size, 6) with columns
                [back_x, back_y, back_z, thigh_x, thigh_y, thigh_z]

    Returns:
        Dictionary with ENMO features for back and thigh sensors
    """
    back_enmo, thigh_enmo = calculate_enmo_from_window(window)

    return {
        # Back sensor ENMO features
        "enmo_b_mean": np.mean(back_enmo),
        "enmo_b_max": np.max(back_enmo),
        "enmo_b_median": np.median(back_enmo),
        "enmo_b_std": np.std(back_enmo),
        "enmo_b_energy": np.sum(back_enmo**2),
        # Thigh sensor ENMO features
        "enmo_t_mean": np.mean(thigh_enmo),
        "enmo_t_max": np.max(thigh_enmo),
        "enmo_t_median": np.median(thigh_enmo),
        "enmo_t_std": np.std(thigh_enmo),
        "enmo_t_energy": np.sum(thigh_enmo**2),
    }
