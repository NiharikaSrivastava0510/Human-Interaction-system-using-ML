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
SAMPLING_RATE = 50  # Hz (detected from data via mode of time diffs)
SENSOR_AXES = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
EXPECTED_COLUMNS = ['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']

# Initial activity label mapping (all labels in raw data)
ACTIVITY_LABEL_MAPPING = {
    1: 'Walking', 2: 'Running', 3: 'Shuffling', 4: 'Stairs (Ascending)',
    5: 'Stairs (Descending)', 6: 'Standing', 7: 'Sitting', 8: 'Lying',
    13: 'Cycling (Sit)', 14: 'Cycling (Stand)', 130: 'Cycling (Sit, inactive)',
    140: 'Cycling (Stand, inactive)'
}

# Final activity labels after cleaning (merged stairs = 9)
ACTIVITY_LABELS = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    6: "Standing",
    7: "Sitting",
    8: "Lying",
    9: "Stairs",  # Merged: Ascending (4) + Descending (5)
}
NUM_CLASSES = len(ACTIVITY_LABELS)

# Labels to drop
CYCLING_LABELS = [13, 14, 130, 140]
STAIRS_LABELS = [4, 5]
MERGED_STAIRS_LABEL = 9
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
GMM_FINETUNED_K = 16
GMM_COVARIANCE_TYPE = "full"
GMM_MAX_ITER = 200
GMM_N_INIT = 5
GMM_N_INIT_FINETUNE = 3
GMM_RANDOM_STATE = 42
GMM_BIC_SEARCH_RANGE = [8, 10, 12, 16, 20]

# ─── CNN Hyperparameters ─────────────────────────────────
CNN_BASELINE_EPOCHS = 10
CNN_FINETUNED_EPOCHS = 50
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT_RATE = 0.3

# Baseline CNN architecture
CNN_BASELINE_FILTERS = 64
CNN_BASELINE_KERNEL_SIZE = 3
CNN_BASELINE_DROPOUT = 0.5

# Fine-tuned MLP architecture (operates on 16-feature set)
CNN_FINETUNED_HIDDEN_1 = 128
CNN_FINETUNED_HIDDEN_2 = 64
CNN_FINETUNED_DROPOUT = 0.3

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
FROZEN_SENSOR_ROLLING_WINDOW = 100  # samples for rolling std
TIME_GAP_THRESHOLD = 0.015          # seconds — gaps above this are flagged
