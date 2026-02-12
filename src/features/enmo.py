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
import pandas as pd


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


def compute_enmo_for_window(window: np.ndarray) -> dict:
    """
    Compute ENMO-based features for a single window.

    Args:
        window: Array of shape (window_size, 6) with columns
                [back_x, back_y, back_z, thigh_x, thigh_y, thigh_z]

    Returns:
        Dictionary with ENMO features for back and thigh sensors
    """
    back_enmo = calculate_enmo(window[:, 0], window[:, 1], window[:, 2])
    thigh_enmo = calculate_enmo(window[:, 3], window[:, 4], window[:, 5])

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
