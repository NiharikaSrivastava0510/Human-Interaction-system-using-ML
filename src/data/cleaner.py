"""
Data cleaning: remove cycling, merge stairs, handle NaNs and invalid data.

Matches the notebook pipeline:
1. Drop cycling labels (13, 14, 130, 140)
2. Merge stairs ascending (4) + descending (5) into label 9
3. Remove unknown labels (label 10)
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.utils.config import (
    CYCLING_LABELS, STAIRS_LABELS, MERGED_STAIRS_LABEL,
    UNKNOWN_LABEL, ACTIVITY_LABEL_MAPPING
)


def drop_cycling_activity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all cycling activity data (labels 13, 14, 130, 140).

    Args:
        df: Input DataFrame with 'label' column

    Returns:
        DataFrame with cycling rows removed
    """
    df_filtered = df[~df['label'].isin(CYCLING_LABELS)].copy()
    return df_filtered


def merge_stairs_activity_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stairs ascending (4) and descending (5) into a single 'stairs' class (9).

    Args:
        df: Input DataFrame with 'label' column

    Returns:
        DataFrame with stairs labels merged to 9
    """
    stairs_check = df['label'].isin(STAIRS_LABELS)
    df.loc[stairs_check, 'label'] = MERGED_STAIRS_LABEL
    return df


def clean_dataset(
    df: pd.DataFrame,
    drop_cycling: bool = True,
    merge_stairs: bool = True,
    remove_unknown: bool = True
) -> pd.DataFrame:
    """
    Apply all cleaning operations to raw activity data.

    Args:
        df: Raw input DataFrame
        drop_cycling: Whether to drop cycling activities (labels 13, 14, 130, 140)
        merge_stairs: Whether to merge ascending (4) and descending (5) stairs into class 9
        remove_unknown: Whether to remove unknown labels (label 10)

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    if drop_cycling:
        initial_len = len(df_clean)
        df_clean = drop_cycling_activity_data(df_clean)
        print(f"Dropped {initial_len - len(df_clean)} cycling records")

    if merge_stairs:
        df_clean = merge_stairs_activity_label(df_clean)

    if remove_unknown:
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean['label'] != UNKNOWN_LABEL]
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


def analyse_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse class balance of the dataset.

    Args:
        df: Input DataFrame with 'label' column

    Returns:
        DataFrame with Label, Activity, Raw Count, Percentage columns
    """
    balance = df['label'].value_counts().reset_index()
    balance.columns = ['Label', 'Raw Count']
    total_count = balance['Raw Count'].sum()
    balance['Percentage'] = (balance['Raw Count'] / total_count) * 100
    balance['Activity'] = balance['Label'].map(ACTIVITY_LABEL_MAPPING)
    balance = balance[['Label', 'Activity', 'Raw Count', 'Percentage']].sort_values(
        by='Raw Count', ascending=False
    ).reset_index(drop=True)
    return balance


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
