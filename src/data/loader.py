"""
CSV loading and initial data validation.

Handles inconsistent columns in files like S015 and S023 which have
extra 'index' and 'unnamed' columns. Extracts subject_id from filename.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional

from src.utils.config import EXPECTED_COLUMNS


def load_har_sensor_data(
    directory_path: Union[str, Path],
    base_name: str = "Dataset"
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all CSV files from a directory with column cleaning.

    Handles inconsistent columns (e.g. extra 'index' or 'unnamed' columns
    in S015 and S023), drops rows with invalid timestamps, and adds
    subject_id extracted from the filename.

    Args:
        directory_path: Path to directory containing CSV files
        base_name: Label for logging (e.g. 'Training Set', 'Test Set')

    Returns:
        Concatenated DataFrame with columns matching EXPECTED_COLUMNS + subject_id.
        Returns None if no files loaded successfully.
    """
    pattern = os.path.join(str(directory_path), "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return None

    har_sensor_dfs = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"Successfully loaded: {os.path.basename(file)}")

            # Clean inconsistent columns
            cols_to_drop = [
                col for col in df.columns
                if col not in EXPECTED_COLUMNS and 'unnamed' in col.lower()
            ]
            if 'index' in df.columns:
                cols_to_drop.append('index')

            # Handle files where first column is not expected and timestamp is second
            if df.columns[0] not in EXPECTED_COLUMNS and df.columns[1] == 'timestamp':
                df = df.iloc[:, 1:]

            df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
            df = df[EXPECTED_COLUMNS]

            # Add subject_id from filename
            subject_id = os.path.basename(file).split('.')[0]
            df['subject_id'] = subject_id

            # Parse timestamps and drop invalid rows
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)

            har_sensor_dfs.append(df)

        except Exception as e:
            print(f"Failed to load {os.path.basename(file)}: {e}")

    if har_sensor_dfs:
        final_dataset = pd.concat(har_sensor_dfs, ignore_index=True)
        print(f"\nSuccessfully loaded and cleaned data for {base_name}. "
              f"Total rows: {len(final_dataset)}")
        return final_dataset
    else:
        print("No CSV files were loaded successfully")
        return None


# Keep backward compatibility alias
load_dataset = load_har_sensor_data


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
        expected_columns = EXPECTED_COLUMNS

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
        'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
        'missing_values': df.isnull().sum().to_dict(),
        'unique_labels': int(df['label'].nunique()) if 'label' in df.columns else None,
        'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else None,
    }
    return stats
