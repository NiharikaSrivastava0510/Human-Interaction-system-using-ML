"""
Script to train Gaussian Mixture Model for activity classification.

Matches the notebook's GMM pipeline:
- 3.1.a: Baseline GMM with 10 ENMO features
- 3.1.b: Fine-tuned GMM with BIC-based k selection on 16 features
- 3.1.c: Final model with GroupKFold cross-validation
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import get_windows, get_windows_with_subjects, load_windows
from src.features.builder import extract_features_enmo, extract_refined_features
from src.models.gmm import GMMActivityModel
from src.evaluation.metrics import calculate_metrics, calculate_clustering_metrics
from src.utils.config import GMM_FINETUNED_K, CV_N_SPLITS


def train_gmm(
    data_path: str = None,
    output_path: str = None,
    n_components: int = None,
    mode: str = 'finetuned',
    train_df=None,
    test_df=None,
):
    """
    Train GMM model matching the notebook pipeline.

    Args:
        data_path: Path to processed .npz data file (alternative to DataFrames)
        output_path: Path to save model
        n_components: Number of mixture components (auto-detected if None)
        mode: 'baseline' (10 ENMO features) or 'finetuned' (16 refined features)
        train_df: Training DataFrame (alternative to data_path)
        test_df: Test DataFrame (alternative to data_path)
    """
    # Load data
    if train_df is not None:
        X_train_raw, y_train = get_windows(train_df)
        X_test_raw, y_test = get_windows(test_df) if test_df is not None else (None, None)
    else:
        X_train_raw, y_train = load_windows(data_path)
        X_test_raw, y_test = None, None

    # Extract features based on mode
    if mode == 'baseline':
        print("Extracting 10 ENMO baseline features...")
        X_train_feat = extract_features_enmo(X_train_raw)
        if X_test_raw is not None:
            X_test_feat = extract_features_enmo(X_test_raw)
        if n_components is None:
            n_components = len(np.unique(y_train))
    else:
        print("Extracting 16 refined features...")
        X_train_feat = extract_refined_features(X_train_raw)
        if X_test_raw is not None:
            X_test_feat = extract_refined_features(X_test_raw)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    if X_test_raw is not None:
        X_test_scaled = scaler.transform(X_test_feat)

    # Train model
    if mode == 'baseline':
        print(f"Training GMM with k={n_components} components...")
        model = GMMActivityModel(n_components=n_components)
        model.fit(X_train_scaled, y_train)
    else:
        print("Finding optimal k via BIC search...")
        model = GMMActivityModel.find_optimal_k(X_train_scaled, y_train)

    # Evaluate
    y_pred_mapped = model.predict_mapped(X_train_scaled)
    train_acc = accuracy_score(y_train, y_pred_mapped)
    train_ari = adjusted_rand_score(y_train, model.predict(X_train_scaled))
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training ARI: {train_ari:.4f}")

    if X_test_raw is not None:
        y_test_pred = model.predict_mapped(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_ari = adjusted_rand_score(y_test, model.predict(X_test_scaled))
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ARI: {test_ari:.4f}")
        print(classification_report(y_test, y_test_pred, zero_division=0))

    # Save model
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        print(f"Model saved to {output_path}")

    return model


def train_gmm_with_cv(train_df, test_df=None):
    """
    Train GMM with GroupKFold cross-validation matching notebook section 3.1.c.
    """
    X_train_raw, y_train, groups_train = get_windows_with_subjects(train_df)
    X_train_feat = extract_refined_features(X_train_raw)

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=CV_N_SPLITS)

    train_scores, val_scores = [], []
    fold = 1

    for train_idx, val_idx in gkf.split(X_train_feat, y_train, groups=groups_train):
        X_tr, X_val = X_train_feat[train_idx], X_train_feat[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler_cv = StandardScaler()
        X_tr_scaled = scaler_cv.fit_transform(X_tr)
        X_val_scaled = scaler_cv.transform(X_val)

        gmm_cv = GMMActivityModel(n_components=GMM_FINETUNED_K)
        gmm_cv.fit(X_tr_scaled, y_tr)

        y_tr_pred = gmm_cv.predict_mapped(X_tr_scaled)
        tr_acc = accuracy_score(y_tr, y_tr_pred)
        train_scores.append(tr_acc)

        y_val_pred = gmm_cv.predict_mapped(X_val_scaled)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_scores.append(val_acc)

        print(f"Fold {fold}: Train Acc = {tr_acc:.4f} | Val Acc = {val_acc:.4f}")
        fold += 1

    print(f"Average Training Accuracy: {np.mean(train_scores):.4f} +/- {np.std(train_scores):.4f}")
    print(f"Average CV Accuracy:       {np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f}")

    return train_scores, val_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GMM for activity classification')
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save model')
    parser.add_argument('--n-components', type=int, default=None, help='Number of components')
    parser.add_argument('--mode', type=str, default='finetuned',
                        choices=['baseline', 'finetuned'], help='Training mode')

    args = parser.parse_args()

    train_gmm(
        data_path=args.data_path,
        output_path=args.output_path,
        n_components=args.n_components,
        mode=args.mode,
    )
