"""
Sensor quality analysis: detect frozen sensors and time gaps.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def detect_frozen_sensors(
    df: pd.DataFrame,
    std_threshold: float = 0.02,
    window_size: int = 10
) -> np.ndarray:
    """
    Detect frozen sensor segments using rolling standard deviation.
    
    Args:
        df: Input DataFrame with sensor columns
        std_threshold: Standard deviation threshold (in g) for frozen detection
        window_size: Window size for rolling calculation
        
    Returns:
        Boolean array indicating frozen segments
    """
    sensor_cols = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    frozen = np.zeros(len(df), dtype=bool)
    
    for col in sensor_cols:
        rolling_std = df[col].rolling(window=window_size, center=True).std()
        frozen |= rolling_std < std_threshold
    
    return frozen


def detect_time_gaps(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    expected_interval: float = 0.02  # 50 Hz sampling
) -> List[Tuple[int, float]]:
    """
    Detect time gaps in sensor readings.
    
    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        expected_interval: Expected time between samples (seconds)
        
    Returns:
        List of (index, gap_duration) tuples for gaps exceeding threshold
    """
    if timestamp_col not in df.columns:
        return []
    
    df['timestamp'] = pd.to_datetime(df[timestamp_col])
    time_diffs = df[timestamp_col].diff().dt.total_seconds()
    
    gap_threshold = expected_interval * 2  # Allow 2x expected interval
    gaps = [(idx, gap) for idx, gap in enumerate(time_diffs) if gap > gap_threshold]
    
    return gaps


def get_sensor_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive sensor quality report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    frozen = detect_frozen_sensors(df)
    gaps = detect_time_gaps(df)
    
    return {
        'total_rows': len(df),
        'frozen_segments': int(frozen.sum()),
        'frozen_percentage': float(frozen.sum() / len(df) * 100),
        'time_gaps': len(gaps),
        'max_gap_duration': max([g[1] for g in gaps]) if gaps else 0,
    }
