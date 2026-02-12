"""
Sensor quality analysis: detect frozen sensors and time gaps.

Matches the notebook's S007 quality analysis:
- Rolling std on back_x with window=100
- Frozen sensor detection for label==1 (Walking) with std < 0.02
- Time gap detection with threshold > 0.015s
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from src.utils.config import (
    FROZEN_SENSOR_STD_THRESHOLD,
    FROZEN_SENSOR_ROLLING_WINDOW,
    TIME_GAP_THRESHOLD,
)


def detect_frozen_sensors(
    df: pd.DataFrame,
    std_threshold: float = FROZEN_SENSOR_STD_THRESHOLD,
    window_size: int = FROZEN_SENSOR_ROLLING_WINDOW,
    sensor_col: str = 'back_x',
    target_label: int = 1,
) -> np.ndarray:
    """
    Detect frozen sensor segments using rolling standard deviation.

    Matches notebook logic: looks for label=Walking (1) AND rolling std < threshold
    on the back_x sensor axis.

    Args:
        df: Input DataFrame with sensor columns and 'label' column
        std_threshold: Standard deviation threshold (in g) for frozen detection
        window_size: Window size for rolling std calculation
        sensor_col: Sensor column to check for frozen signal
        target_label: Activity label to check for frozen sensor mismatch

    Returns:
        Boolean array indicating frozen segments (True = frozen)
    """
    rolling_std = df[sensor_col].rolling(window=window_size, min_periods=1).std()
    frozen_mask = (df['label'] == target_label) & (rolling_std < std_threshold)
    return frozen_mask.values


def clean_frozen_sensors(
    df: pd.DataFrame,
    std_threshold: float = FROZEN_SENSOR_STD_THRESHOLD,
    window_size: int = FROZEN_SENSOR_ROLLING_WINDOW,
    sensor_col: str = 'back_x',
    target_label: int = 1,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove frozen sensor segments from dataframe.

    Args:
        df: Input DataFrame
        std_threshold: Std threshold for frozen detection
        window_size: Rolling window size
        sensor_col: Sensor column to check
        target_label: Activity label to check against

    Returns:
        Tuple of (cleaned DataFrame, number of rows dropped)
    """
    frozen_mask = detect_frozen_sensors(
        df, std_threshold, window_size, sensor_col, target_label
    )
    initial_size = len(df)
    df_cleaned = df[~frozen_mask].copy()
    dropped_count = initial_size - len(df_cleaned)
    return df_cleaned, dropped_count


def detect_time_gaps(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    gap_threshold: float = TIME_GAP_THRESHOLD,
) -> pd.Series:
    """
    Detect time gaps in sensor readings.

    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        gap_threshold: Minimum gap duration to flag (seconds)

    Returns:
        Series of time differences where gaps exceed threshold
    """
    if timestamp_col not in df.columns:
        return pd.Series(dtype=float)

    df_ts = df.copy()
    df_ts[timestamp_col] = pd.to_datetime(df_ts[timestamp_col])
    time_diffs = df_ts[timestamp_col].diff().dt.total_seconds()
    gaps = time_diffs[time_diffs > gap_threshold]
    return gaps


def get_subject_quality_report(
    df: pd.DataFrame,
    subject_id: str = 'S007',
) -> dict:
    """
    Generate quality report for a specific subject.

    Args:
        df: Full training DataFrame with 'subject_id' column
        subject_id: Subject to analyse

    Returns:
        Dictionary with quality metrics for the subject
    """
    df_subject = df[df['subject_id'] == subject_id].copy()

    if df_subject.empty:
        return {'error': f'Subject {subject_id} not found'}

    frozen_mask = detect_frozen_sensors(df_subject)
    gaps = detect_time_gaps(df_subject)

    label_10_count = len(df_subject[df_subject['label'] == 10])

    return {
        'subject_id': subject_id,
        'total_rows': len(df_subject),
        'label_10_count': label_10_count,
        'frozen_segments': int(frozen_mask.sum()),
        'frozen_percentage': float(frozen_mask.sum() / len(df_subject) * 100),
        'time_gaps': len(gaps),
        'max_gap_duration': float(gaps.max()) if len(gaps) > 0 else 0,
    }


def get_sensor_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive sensor quality report.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with quality metrics
    """
    frozen_mask = detect_frozen_sensors(df)
    gaps = detect_time_gaps(df)

    return {
        'total_rows': len(df),
        'frozen_segments': int(frozen_mask.sum()),
        'frozen_percentage': float(frozen_mask.sum() / len(df) * 100),
        'time_gaps': len(gaps),
        'max_gap_duration': float(gaps.max()) if len(gaps) > 0 else 0,
    }
