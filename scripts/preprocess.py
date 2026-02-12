"""
End-to-end data preprocessing pipeline.

Usage:
    python scripts/preprocess.py --data-dir data/raw/ --output-dir data/processed/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, CYCLING_LABELS, UNKNOWN_LABEL,
    WINDOW_SIZE, STEP_SIZE, SENSOR_AXES
)


def main(data_dir: Path, output_dir: Path):
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("Human Activity Recognition — Data Preprocessing")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/5] Loading raw CSV files...")
    # 1 Implement in src/data/loader.py
    train_df = load_dataset(data_dir)
    test_df = load_dataset(data_dir / "test")

    # Step 2: Clean data
    print("[2/5] Cleaning data...")
    print(f"  - Dropping cycling labels: {CYCLING_LABELS}")
    print(f"  - Removing unknown label: {UNKNOWN_LABEL}")
    print("  - Merging stairs ascending + descending")
    # 2 Implement in src/data/cleaner.py

    # Step 3: Quality checks
    print("[3/5] Running sensor quality analysis...")
    print(f"  - Frozen sensor threshold: < 0.02g std")
    print(f"  - Time gap threshold: > 0.015s")
    # 3 Implement in src/data/quality.py

    # Step 4: Sliding window segmentation
    print("[4/5] Applying sliding window segmentation...")
    print(f"  - Window size: {WINDOW_SIZE} samples")
    print(f"  - Step size: {STEP_SIZE} samples")
    # 4 Implement in src/data/windowing.py

    # Step 5: Save processed data
    print("[5/5] Saving processed data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 5 Save as .npz files
    # np.savez(output_dir / "train_windows.npz", X=X_train, y=y_train)
    # np.savez(output_dir / "test_windows.npz", X=X_test, y=y_test)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HAR dataset")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
