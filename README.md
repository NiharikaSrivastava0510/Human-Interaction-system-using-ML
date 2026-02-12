# Human-Interaction-system-using-ML


[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning pipeline for classifying human activities from wearable accelerometer data. This project compares three distinct approaches — **Gaussian Mixture Models** (unsupervised), **Convolutional Neural Networks** (deep learning), and **Random Forests** (ensemble) — to classify 7 activity types from 6.6M+ sensor readings.

**Best Result: 92.9% test accuracy** with a fine-tuned 3-layer CNN using batch normalisation, dropout regularisation, and global average pooling.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Gaussian Mixture Model](#2-gaussian-mixture-model-unsupervised)
  - [Convolutional Neural Network](#3-convolutional-neural-network-supervised)
  - [Random Forest](#4-random-forest-supervised)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

This project tackles the challenge of recognising human activities from raw accelerometer signals captured by body-worn sensors (back and thigh). The pipeline covers the full ML lifecycle: data cleaning, feature engineering, model development, hyperparameter tuning, cross-validation, and performance analysis.

**Activities Classified (7 classes):**
Walking, Sitting, Standing, Lying Down, Shuffling, Running, Stairs (Ascending + Descending merged)

**Key Challenges Addressed:**
- Severe class imbalance (47% of data was NaN/unlabelled)
- Sensor quality issues (frozen sensors, time gaps up to 17.69 seconds)
- Subject-dependent variation requiring Group K-Fold cross-validation to prevent data leakage

---

## Dataset

| Split | Raw Data Points | After Cleaning | Feature Windows |
|-------|---------------:|---------------:|----------------:|
| Training | 5,568,946 | 5,070,330 | 101,405 |
| Test | 1,122,375 | 1,105,427 | 22,107 |
| **Total** | **6,691,321** | **6,175,757** | **123,512** |

- **Sensors:** 6-axis accelerometer (back\_x, back\_y, back\_z, thigh\_x, thigh\_y, thigh\_z)
- **Sampling Rate:** 50 Hz (after downsampling from 100 Hz)
- **Sliding Window:** 2-second windows with 1-second overlap (100 samples per window)

---

## Project Structure

```
human-activity-recognition/
│
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
│
├── data/
│   ├── raw/                        # Original CSV files (not tracked in git)
│   │   ├── *.csv
│   │   └── test/
│   ├── processed/                  # Cleaned and windowed data
│   │   ├── train_windows.npz
│   │   └── test_windows.npz
│   └── README.md                   # Data source and download instructions
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb          # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb       # ENMO calculation, sliding windows
│   ├── 03_gmm_pipeline.ipynb             # GMM baseline and fine-tuning
│   ├── 04_cnn_pipeline.ipynb             # CNN baseline and fine-tuning
│   ├── 05_random_forest_pipeline.ipynb   # Random Forest baseline and fine-tuning
│   └── 06_model_comparison.ipynb         # Cross-model performance analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # CSV loading and initial validation
│   │   ├── cleaner.py              # Drop cycling, merge stairs, handle NaNs
│   │   ├── quality.py              # Sensor freeze detection, time gap analysis
│   │   └── windowing.py            # Sliding window segmentation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── enmo.py                 # ENMO (Euclidean Norm Minus One) calculation
│   │   ├── statistical.py          # Mean, max, median, std, energy features
│   │   └── builder.py              # Feature pipeline (10-feature and 16-feature sets)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gmm.py                  # Gaussian Mixture Model pipeline
│   │   ├── cnn.py                  # CNN architecture (baseline + fine-tuned)
│   │   ├── random_forest.py        # Random Forest pipeline
│   │   └── base.py                 # Abstract base class for models
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Accuracy, precision, recall, F1, ARI
│   │   ├── cross_validation.py     # Group K-Fold CV implementation
│   │   └── error_analysis.py       # Confusion matrix, error categorisation
│   │
│   ├── visualisation/
│   │   ├── __init__.py
│   │   ├── eda_plots.py            # Class distribution, correlation heatmaps
│   │   ├── sensor_plots.py         # Waveform comparison, time-series plots
│   │   ├── model_plots.py          # Confusion matrices, performance bar charts
│   │   └── feature_importance.py   # Random Forest feature importance plots
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Hyperparameters, file paths, constants
│       └── logger.py               # Logging configuration
│
├── models/                         # Saved model weights and artifacts
│   ├── gmm/
│   │   ├── gmm_baseline.pkl
│   │   └── gmm_finetuned.pkl
│   ├── cnn/
│   │   ├── cnn_baseline.h5
│   │   └── cnn_finetuned.h5
│   └── random_forest/
│       ├── rf_baseline.pkl
│       └── rf_finetuned.pkl
│
├── results/
│   ├── figures/                    # Generated plots and visualisations
│   │   ├── class_balance.png
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrices/
│   │   ├── performance_comparison.png
│   │   └── feature_importance.png
│   └── metrics/                    # Saved evaluation metrics
│       ├── gmm_results.json
│       ├── cnn_results.json
│       └── rf_results.json
│
├── scripts/
│   ├── train_gmm.py               # CLI script to train GMM
│   ├── train_cnn.py               # CLI script to train CNN
│   ├── train_rf.py                # CLI script to train Random Forest
│   ├── evaluate_all.py            # Run all models and generate comparison
│   └── preprocess.py              # End-to-end data preprocessing
│
└── tests/
    ├── __init__.py
    ├── test_cleaner.py
    ├── test_windowing.py
    ├── test_features.py
    └── test_models.py
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/NiharikaSrivastava0510/human-activity-recognition.git
cd human-activity-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Usage

### Quick Start

```bash
# Step 1: Preprocess raw data
python scripts/preprocess.py --data-dir data/raw/train --output-dir data/processed/

# Step 2: Train models
python scripts/train_gmm.py --features 16 --clusters 20
python scripts/train_cnn.py --epochs 50 --batch-size 64
python scripts/train_rf.py --n-estimators 200 --max-depth 20

# Step 3: Evaluate and compare
python scripts/evaluate_all.py --output results/
```

### Using Notebooks

Launch Jupyter and work through the notebooks in order:

```bash
jupyter notebook notebooks/
```

### Programmatic Usage

```python
from src.data.loader import load_dataset
from src.data.cleaner import clean_dataset
from src.data.windowing import create_sliding_windows
from src.features.builder import build_features
from src.models.cnn import FineTunedCNN

# Load and preprocess
train_raw = load_dataset("data/raw/train")
train_clean = clean_dataset(train_raw, drop_cycling=True, merge_stairs=True)
X_train, y_train = create_sliding_windows(train_clean, window_size=100, overlap=50)

# Train CNN
model = FineTunedCNN(num_classes=7)
model.fit(X_train, y_train, epochs=50, batch_size=64)
```

---

## Pipeline Details

### 1. Data Preprocessing

**Cleaning steps:**
- Converted timestamps to proper datetime format
- Dropped all cycling data (labels 13, 14, 130, 140) — removed 515,564 rows
- Merged Stairs Ascending (4) and Stairs Descending (5) into a single Stairs class (9)
- Removed unknown labels (Label 10) and frozen sensor segments (Std < 0.02g)
- Detected and handled 40 time gaps exceeding the 0.015s threshold

**Sliding Window Segmentation:**
- Window size: 2 seconds (100 samples at 50 Hz)
- Overlap: 1 second (50% overlap)
- Step size: 50 samples

### 2. Gaussian Mixture Model (Unsupervised)

| Configuration | Features | Clusters | ARI | Accuracy |
|--------------|----------|----------|-----|----------|
| Baseline | 10 (ENMO-based) | 7 | 0.4196 | 63.8% |
| Fine-tuned | 16 (raw axes + ENMO) | 20 | 0.3195 | 79.2% |
| **Final (CV)** | **16** | **20** | — | **80.3%** |

Key improvement: Adding raw axis means (6 features) resolved the Sitting vs. Standing confusion that ENMO-only features couldn't capture, since ENMO removes gravity information.

### 3. Convolutional Neural Network (Supervised)

**Fine-tuned Architecture:**
```
Input (100, 6) → [Conv1D→BN→ReLU→MaxPool] ×3 → GlobalAvgPool → Dense(7, softmax)
```

| Configuration | Architecture | Dropout | Accuracy |
|--------------|-------------|---------|----------|
| Baseline | 1-layer Conv1D | None | 88.2% |
| Fine-tuned | 3-layer Conv1D + BN + GAP | 0.3–0.5 | 92.3% |
| **Final (CV)** | **3-layer + Group K-Fold** | **0.3–0.5** | **92.9%** |

Cross-validation (5-fold, grouped by subject): Mean accuracy 93.12% ± 0.8%

### 4. Random Forest (Supervised)

| Configuration | Features | Estimators | Max Depth | Accuracy |
|--------------|----------|-----------|-----------|----------|
| Baseline | 10 | 100 | None | 66.4% |
| Fine-tuned | 16 | 200 | 20 | 89.7% |
| **Final (CV)** | **16** | **200** | **20** | **89.7%** |

Top features by importance: `tz_mean` (0.22), `bx_mean` (0.12), `bx_std` (0.08)

---

## Results

### Comparative Performance (Test Set)

| Model | Test Accuracy | CV Mean | Training Accuracy | Key Strength |
|-------|:------------:|:-------:|:-----------------:|-------------|
| GMM | 79.2% | 82.2% | 86.1% | No labels needed |
| **CNN** | **92.9%** | **93.1%** | **98.9%** | Best overall accuracy |
| Random Forest | 89.7% | 93.1% | 98.9% | Interpretable features |

### Confusion Matrix Insights

- **CNN excels at:** Sitting vs. Standing (gravity orientation), Walking vs. Stairs (temporal patterns)
- **Random Forest struggles with:** Shuffling vs. Walking (similar statistical features)
- **GMM conflates:** Walking and Stairs into a single cluster due to overlapping mean/std distributions

---

## Key Findings

1. **Raw axis features are essential** — ENMO alone removes gravity, making static postures (Sitting, Standing) indistinguishable. Adding raw means resolved this.

2. **CNNs learn temporal patterns that handcrafted features miss** — The 3-layer CNN automatically captures gait cycle differences between Walking and Stairs that statistical features cannot represent.

3. **Group K-Fold CV is critical for HAR** — Standard K-Fold leads to data leakage since windows from the same subject appear in both train and validation sets. Group K-Fold splits by subject ID.

4. **Batch normalisation + Global Average Pooling** significantly reduced overfitting in the CNN (training-test gap dropped from ~10% to ~6%).

---

## Future Work

- **CNN-LSTM hybrid** for capturing longer temporal dependencies across windows
- **Transfer learning** to adapt the model to new sensor placements or sampling rates
- **Real-time inference** pipeline with model quantisation for edge deployment
- **Data augmentation** techniques (time warping, rotation) to improve minority class performance
- **Attention mechanisms** to visualise which temporal segments drive classification decisions

---

## Author

**Niharika Srivastava**
MSc Artificial Intelligence, University of Southampton (2025–2026)

- [LinkedIn](https://www.linkedin.com/in/niharika-srivastava048361167)
- [Email](mailto:niharika051095@gmail.com)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was completed as part of the Machine Learning Technologies module at the University of Southampton.

