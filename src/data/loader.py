"""
CSV loading and initial data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple


def load_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a directory.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Concatenated DataFrame with all data points
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)


def validate_columns(df: pd.DataFrame, expected_columns: list = None) -> bool:
    """
    Validate that dataframe contains expected columns.
    
    Args:
        df: Input DataFrame
        expected_columns: List of expected column names
        
    Returns:
        True if validation passes
    """
    if expected_columns is None:
        expected_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']
    
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    return True


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_labels': df['label'].nunique() if 'label' in df.columns else None,
    }
    return stats
