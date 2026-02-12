"""
Data cleaning: remove cycling, merge stairs, handle NaNs and invalid data.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def clean_dataset(
    df: pd.DataFrame,
    drop_cycling: bool = True,
    merge_stairs: bool = True,
    remove_unknown: bool = True
) -> pd.DataFrame:
    """
    Apply cleaning operations to raw activity data.
    
    Args:
        df: Raw input DataFrame
        drop_cycling: Whether to drop cycling activities (labels 13, 14, 130, 140)
        merge_stairs: Whether to merge ascending (4) and descending (5) stairs into class 9
        remove_unknown: Whether to remove unknown labels (label 10)
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Drop cycling if requested
    if drop_cycling:
        cycling_labels = [13, 14, 130, 140]
        initial_len = len(df_clean)
        df_clean = df_clean[~df_clean['label'].isin(cycling_labels)]
        print(f"Dropped {initial_len - len(df_clean)} cycling records")
    
    # Merge stairs if requested
    if merge_stairs:
        df_clean.loc[df_clean['label'] == 5, 'label'] = 4
        df_clean.loc[df_clean['label'] == 4, 'label'] = 9
    
    # Remove unknown labels
    if remove_unknown:
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean['label'] != 10]
        print(f"Dropped {initial_len - len(df_clean)} unknown label records")
    
    return df_clean.reset_index(drop=True)


def handle_nan_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle NaN values in sensor readings.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'forward_fill' for forward fill
        
    Returns:
        DataFrame with NaNs handled
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'forward_fill':
        return df.fillna(method='ffill')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_class_distribution(df: pd.DataFrame) -> dict:
    """
    Get class distribution statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with class counts and percentages
    """
    counts = df['label'].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(2)
    
    return {
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict(),
        'total': len(df)
    }
