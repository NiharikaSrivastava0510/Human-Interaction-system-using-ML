# Human-Interaction-system-using-ML


[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning pipeline for classifying human activities from wearable accelerometer data. This project compares three distinct approaches — **Gaussian Mixture Models** (unsupervised), **Convolutional Neural Networks** (deep learning), and **Random Forests** (ensemble) — to classify 7 activity types from 6.6M+ sensor readings.

**🏆 Best Result: 92.9% test accuracy** with a fine-tuned 3-layer CNN using batch normalisation, dropout regularisation, and global average pooling. All three models have been trained, evaluated, and saved.

**Model Status:**
- ✅ **CNN**: 92.9% accuracy (best performer)
- ✅ **Random Forest**: 89.46% accuracy (most interpretable)
- ✅ **GMM**: 43.1% accuracy (unsupervised baseline)

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
│   │   ├── train/
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

| Configuration | Features | Clusters | ARI | Test Accuracy |
|--------------|----------|----------|-----|----------|
| Baseline | 10 (ENMO-based) | 7 | 0.4196 | 63.8% |
| Fine-tuned | 16 (raw axes + ENMO) | 20 | 0.4162 | **43.1%** |

**Status:** Model trained and saved at `models/gmm/gmm_finetuned.pkl`
**Note:** Lower test accuracy indicates challenging unsupervised clustering of overlapping activity distributions. GMM performs well at discovering natural clusters but struggles with label mapping in imbalanced multiclass scenarios.

Key attempt: Adding raw axis means (6 features) aimed to resolve the Sitting vs. Standing confusion that ENMO-only features couldn't capture, since ENMO removes gravity information.

### 3. Convolutional Neural Network (Supervised)

**Fine-tuned Architecture:**
```
Input (100, 6) → [Conv1D→BN→ReLU→MaxPool] ×3 → GlobalAvgPool → Dense(7, softmax)
```

| Configuration | Architecture | Dropout | Features | Test Accuracy |
|--------------|-------------|---------|----------|----------|
| Baseline | 1-layer Conv1D | None | Raw sensor data | 88.2% |
| Fine-tuned | 3-layer Conv1D + BN + GAP | 0.3–0.5 | Raw sensor data | **92.9%** |

**Status:** Model trained and saved at `models/cnn/cnn_finetuned.h5`
**Performance:** Best-performing model with excellent generalization
- Test accuracy: **92.9%**
- Learns temporal patterns automatically from raw 6-axis sensor data
- Batch normalization significantly reduces overfitting (6% train-test gap)

### 4. Random Forest (Supervised)

| Configuration | Features | Estimators | Max Depth | Test Accuracy |
|--------------|----------|-----------|-----------|----------|
| Baseline | 10 | 100 | None | 66.4% |
| Fine-tuned | 16 | 200 | 20 | **89.46%** |

**Status:** Model trained and saved at `models/random_forest/rf_finetuned.pkl`
**Performance:** Strong baseline with excellent interpretability
- Test accuracy: **89.46%**
- Model parameters: 200 estimators, max_depth=20, min_samples_leaf=2
- Hyperparameter tuning significantly improved performance (+23%)

**Top 10 Feature Importance:**
1. `tz_mean` (0.2162) — Thigh Z-axis mean (primary discriminator)
2. `tx_mean` (0.1775) — Thigh X-axis mean
3. `bx_mean` (0.0960) — Back X-axis mean
4. `tz_std` (0.0922) — Thigh Z-axis std dev
5. `bx_std` (0.0572) — Back X-axis std dev
6. `by_std` (0.0562) — Back Y-axis std dev
7. `enmo_b_max` (0.0488) — Back ENMO max
8. `tx_std` (0.0422) — Thigh X-axis std dev
9. `enmo_b_mean` (0.0358) — Back ENMO mean
10. `ty_std` (0.0353) — Thigh Y-axis std dev

---

## Results

### Model Performance Summary

| Model | Approach | Test Accuracy | Status | Best For |
|-------|----------|:-------------:|:------:|----------|
| **CNN** | Deep Learning (Supervised) | **92.9%** | ✅ Trained & Saved | Overall best, temporal patterns |
| **Random Forest** | Ensemble (Supervised) | **89.46%** | ✅ Trained & Saved | Interpretability, feature importance |
| **GMM** | Unsupervised Clustering | **43.1%** | ✅ Trained & Saved | Baseline, no labels needed |

### Test Set Performance Breakdown

**CNN (Best Performer)**
- Accuracy: 92.9%
- Learns temporal patterns from raw sensor signals
- Effective at distinguishing all 7 activity classes
- Strong on Walking vs. Stairs distinction

**Random Forest**
- Accuracy: 89.46%
- Interpretable decisions with explicit feature contributions
- Top feature: Thigh Z-axis mean (`tz_mean`, 21.6% importance)
- Struggles with rare classes (Shuffling: 61% recall)

**GMM (Unsupervised Baseline)**
- Test Accuracy: 43.1%
- Discovers 20 natural clusters without labels
- Primary limitation: Unsupervised clustering doesn't align well with activity labels
- Strong clustering metrics (ARI: 0.416, NMI: 0.458)
- Performs well at Sitting (100% precision) but misclassifies others

---

## Key Findings

1. **CNN is the clear winner for supervised learning** — With 92.9% test accuracy, the 3-layer CNN with batch normalisation and dropout significantly outperforms stationary feature-based approaches. It learns temporal patterns from raw sensor waveforms that handcrafted features miss.

2. **Random Forest provides strong interpretability** — At 89.46% accuracy, Random Forest is a practical alternative that reveals which features matter most. The thigh Z-axis mean (`tz_mean`: 21.6%) is the strongest single discriminator, reflecting gravity's role in distinguishing postures from dynamic activities.

3. **Unsupervised clustering (GMM) has fundamental limitations** — The 43.1% accuracy reveals that 20-cluster GMM doesn't map cleanly to activity labels. However, unsupervised metrics (ARI: 0.416, NMI: 0.458) show it discovers meaningful structure without any label information.

4. **Feature selection impacts all models:**
   - **10-feature set (ENMO-only):** Limited by loss of gravity information; poor at distinguishing static postures
   - **16-feature set (raw axes + ENMO):** Adds mean/std of each axis; enables CMM and RF to capture both static and dynamic content

5. **Data preprocessing is critical** — Cleaning 47% NaN data, removing 515K cycling rows, merging stair classes, and detecting sensor freezes/time gaps were essential preprocessing steps that improved all downstream model performance.

6. **Class imbalance remains challenging** — Running (Sitting: 38% → Class 7), Sitting (32%), and Shuffling (3%) creates inherent imbalance. CNN handles this better through end-to-end learning; Random Forest shows 61% recall on rare Shuffling class vs. 100% on dominant classes.

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

