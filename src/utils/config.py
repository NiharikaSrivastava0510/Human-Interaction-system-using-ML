"""
Configuration for Human Activity Recognition pipeline.
All hyperparameters, file paths, and constants in one place.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─── Dataset Constants ───────────────────────────────────
SAMPLING_RATE = 50  # Hz (downsampled from 100 Hz)
SENSOR_AXES = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

# Activity labels
ACTIVITY_LABELS = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    6: "Stairs",       # Merged: Ascending (4) + Descending (5)
    7: "Sitting",
    8: "Standing",
    9: "Lying Down",
}
NUM_CLASSES = len(ACTIVITY_LABELS)

# Labels to drop
CYCLING_LABELS = [13, 14, 130, 140]
UNKNOWN_LABEL = 10

# ─── Sliding Window ──────────────────────────────────────
WINDOW_SIZE_SEC = 2.0
OVERLAP_SEC = 1.0
WINDOW_SIZE = int(WINDOW_SIZE_SEC * SAMPLING_RATE)    # 100 samples
STEP_SIZE = int((WINDOW_SIZE_SEC - OVERLAP_SEC) * SAMPLING_RATE)  # 50 samples

# ─── Feature Engineering ─────────────────────────────────
# 10-feature baseline: ENMO-based stats for back + thigh
BASELINE_FEATURES = [
    "enmo_b_mean", "enmo_b_max", "enmo_b_median", "enmo_b_std", "enmo_b_energy",
    "enmo_t_mean", "enmo_t_max", "enmo_t_median", "enmo_t_std", "enmo_t_energy",
]

# 16-feature fine-tuned: raw axis means + stds + ENMO mean/max
FINETUNED_FEATURES = [
    "bx_mean", "by_mean", "bz_mean", "tx_mean", "ty_mean", "tz_mean",
    "bx_std", "by_std", "bz_std", "tx_std", "ty_std", "tz_std",
    "enmo_b_mean", "enmo_b_max", "enmo_t_mean", "enmo_t_max",
]

# ─── GMM Hyperparameters ─────────────────────────────────
GMM_BASELINE_K = 7
GMM_FINETUNED_K = 20
GMM_COVARIANCE_TYPE = "full"
GMM_MAX_ITER = 200
GMM_N_INIT = 5
GMM_RANDOM_STATE = 42

# ─── CNN Hyperparameters ─────────────────────────────────
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT_RATE = 0.4

# Fine-tuned CNN architecture
CNN_FILTERS = [64, 128, 256]       # Filters per conv block
CNN_KERNEL_SIZES = [3, 3, 3]       # Kernel size per block
CNN_POOL_SIZES = [2, 2, 2]         # MaxPool size per block

# ─── Random Forest Hyperparameters ───────────────────────
RF_BASELINE_ESTIMATORS = 100
RF_BASELINE_MAX_DEPTH = None

RF_FINETUNED_ESTIMATORS = 200
RF_FINETUNED_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = 42

# ─── Cross-Validation ────────────────────────────────────
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42

# ─── Data Quality Thresholds ─────────────────────────────
FROZEN_SENSOR_STD_THRESHOLD = 0.02  # g — below this = "frozen" sensor
TIME_GAP_THRESHOLD = 0.015          # seconds — gaps above this are flagged
