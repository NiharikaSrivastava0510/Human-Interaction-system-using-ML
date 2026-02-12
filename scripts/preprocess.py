"""
End-to-end data preprocessing pipeline.

Matches the notebook pipeline (Sections 1 & 2):
1. Load raw CSV files with column cleaning and subject_id extraction
2. Clean data: drop cycling, merge stairs (4+5 → 9), remove label 10
3. Sensor quality analysis (frozen sensors, time gaps)
4. Sliding window segmentation (2s windows, 50% overlap)
5. Save processed windows as .npz files

Usage:
    python scripts/preprocess.py --train-dir data/raw/train --test-dir data/raw/test --output-dir data/processed/
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, CYCLING_LABELS, UNKNOWN_LABEL,
    WINDOW_SIZE, STEP_SIZE, SENSOR_AXES, ACTIVITY_LABELS
)
from src.data.loader import load_har_sensor_data
from src.data.cleaner import clean_dataset, analyse_class_balance
from src.data.quality import get_sensor_quality_report, get_subject_quality_report
from src.data.windowing import get_windows, get_windows_with_subjects, save_windows


def main(train_dir: Path, test_dir: Path, output_dir: Path):
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("Human Activity Recognition — Data Preprocessing")
    print("=" * 60)

    # ── Step 1: Load raw data ────────────────────────────────
    print("\n[1/5] Loading raw CSV files...")
    train_df = load_har_sensor_data(train_dir, base_name="Training Set")
    test_df = load_har_sensor_data(test_dir, base_name="Test Set")

    if train_df is None or test_df is None:
        print("ERROR: Failed to load data. Check directory paths.")
        return

    print(f"  Training set: {len(train_df):,} rows")
    print(f"  Test set:     {len(test_df):,} rows")

    # ── Step 2: Clean data ───────────────────────────────────
    print("\n[2/5] Cleaning data...")
    print(f"  - Dropping cycling labels: {CYCLING_LABELS}")
    print(f"  - Removing unknown label: {UNKNOWN_LABEL}")
    print("  - Merging stairs ascending (4) + descending (5) → 9")

    train_clean = clean_dataset(train_df)
    test_clean = clean_dataset(test_df)

    print(f"\n  Training set after cleaning: {len(train_clean):,} rows")
    print(f"  Test set after cleaning:     {len(test_clean):,} rows")

    # Class balance analysis
    print("\n  Class balance (Training):")
    balance = analyse_class_balance(train_clean)
    print(balance.to_string(index=False))

    # ── Step 3: Quality checks ───────────────────────────────
    print("\n[3/5] Running sensor quality analysis...")

    # Overall quality report
    train_quality = get_sensor_quality_report(train_clean)
    print(f"  Frozen segments: {train_quality['frozen_segments']:,} "
          f"({train_quality['frozen_percentage']:.2f}%)")
    print(f"  Time gaps detected: {train_quality['time_gaps']}")

    # Subject-level report (S007 as per notebook)
    if 'subject_id' in train_clean.columns:
        s007_report = get_subject_quality_report(train_clean, 'S007')
        if 'error' not in s007_report:
            print(f"\n  S007 Quality Report:")
            print(f"    Total rows: {s007_report['total_rows']:,}")
            print(f"    Label 10 count: {s007_report['label_10_count']}")
            print(f"    Frozen segments: {s007_report['frozen_segments']}")

    # ── Step 4: Sliding window segmentation ──────────────────
    print(f"\n[4/5] Applying sliding window segmentation...")
    print(f"  - Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE / 50:.1f}s)")
    print(f"  - Step size: {STEP_SIZE} samples ({STEP_SIZE / 50:.1f}s overlap)")

    X_train, y_train = get_windows(train_clean)
    X_test, y_test = get_windows(test_clean)

    print(f"  Training windows: {X_train.shape[0]:,}  (shape: {X_train.shape})")
    print(f"  Test windows:     {X_test.shape[0]:,}  (shape: {X_test.shape})")

    # Also get windows with subjects for CV
    if 'subject_id' in train_clean.columns:
        X_train_s, y_train_s, groups_train = get_windows_with_subjects(train_clean)
        print(f"  Subject groups: {len(np.unique(groups_train))} unique subjects")

    # ── Step 5: Save processed data ──────────────────────────
    print("\n[5/5] Saving processed data...")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_windows.npz"
    test_path = output_dir / "test_windows.npz"

    save_windows(X_train, y_train, str(train_path))
    save_windows(X_test, y_test, str(test_path))

    print(f"  Saved: {train_path}")
    print(f"  Saved: {test_path}")

    # Save with subject groups if available
    if 'subject_id' in train_clean.columns:
        cv_path = output_dir / "train_windows_with_subjects.npz"
        np.savez_compressed(str(cv_path), X=X_train_s, y=y_train_s, groups=groups_train)
        print(f"  Saved: {cv_path}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"  Final activity classes: {list(ACTIVITY_LABELS.values())}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HAR dataset")
    parser.add_argument("--train-dir", type=Path, default=DATA_RAW / "train",
                        help="Path to training CSV files")
    parser.add_argument("--test-dir", type=Path, default=DATA_RAW / "test",
                        help="Path to test CSV files")
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED)
    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.output_dir)
