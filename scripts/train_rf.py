"""
Script to train Random Forest model for activity classification.

Matches the notebook's Random Forest pipeline:
- 3.3.a: Baseline RF (100 trees) with 10 ENMO features + NN baseline
- 3.3.b: Fine-tuned RF (200 trees, max_depth=20) with 16 optimized features
- 3.3.c/d: GroupKFold CV and error analysis
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import get_windows, get_windows_with_subjects, load_windows
from src.features.builder import (
    extract_features_enmo, extract_refined_features, extract_optimized_features
)
from src.models.random_forest import RandomForestActivityModel
from src.models.cnn import get_baseline_nn
from src.utils.config import (
    RF_FINETUNED_ESTIMATORS, RF_FINETUNED_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
    CNN_BASELINE_EPOCHS, CNN_BATCH_SIZE, CV_N_SPLITS
)


def train_rf_baseline(train_df, test_df):
    """
    Train baseline RF with 10 ENMO features (notebook section 3.3.a).
    """
    X_train_raw, y_train_orig = get_windows(train_df)
    X_test_raw, y_test_orig = get_windows(test_df)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_orig)
    y_test = le.transform(y_test_orig)
    num_classes = len(np.unique(y_train))

    # RF baseline with ENMO features
    X_train_feat = extract_features_enmo(X_train_raw)
    X_test_feat = extract_features_enmo(X_test_raw)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_feat, y_train)
    y_pred_rf = rf.predict(X_test_feat)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

    # NN baseline for comparison
    scaler = StandardScaler()
    N_train, T, F = X_train_raw.shape
    N_test = X_test_raw.shape[0]
    X_train_cnn = scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(N_train, T, F)
    X_test_cnn = scaler.transform(X_test_raw.reshape(-1, F)).reshape(N_test, T, F)

    model = get_baseline_nn(input_shape=(T, F), num_classes=num_classes)
    model.fit(X_train_cnn, y_train, epochs=CNN_BASELINE_EPOCHS,
              batch_size=CNN_BATCH_SIZE, validation_data=(X_test_cnn, y_test), verbose=0)
    _, acc_cnn = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"NN Baseline Accuracy: {acc_cnn:.4f}")

    print(f"\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf,
                                target_names=le.classes_.astype(str), zero_division=0))

    return rf, le


def train_rf_finetuned(train_df, test_df):
    """
    Train fine-tuned RF with 16 optimized features (notebook section 3.3.b).
    """
    X_train_raw, y_train = get_windows(train_df)
    X_test_raw, y_test = get_windows(test_df)

    X_train_feat = extract_optimized_features(X_train_raw)
    X_test_feat = extract_optimized_features(X_test_raw)

    rf_tuned = RandomForestActivityModel.create_finetuned()
    rf_tuned.fit(X_train_feat, y_train)

    y_pred = rf_tuned.predict(X_test_feat)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance
    importances = rf_tuned.get_feature_importance()
    feat_names = (
        [f'{axis}_mean' for axis in ['bx', 'by', 'bz', 'tx', 'ty', 'tz']] +
        [f'{axis}_std' for axis in ['bx', 'by', 'bz', 'tx', 'ty', 'tz']] +
        ['enmo_b_mean', 'enmo_b_max', 'enmo_t_mean', 'enmo_t_max']
    )
    print("\nFeature Importance:")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return rf_tuned


def train_rf_with_cv(train_df, test_df=None):
    """
    Train with GroupKFold CV and error analysis (notebook section 3.3.c/d).
    """
    X_train_raw, y_train, groups_train = get_windows_with_subjects(train_df)
    X_train_feat = extract_refined_features(X_train_raw)

    rf = RandomForestClassifier(
        n_estimators=RF_FINETUNED_ESTIMATORS,
        max_depth=RF_FINETUNED_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1, random_state=42,
    )

    rf.fit(X_train_feat, y_train)
    train_pred = rf.predict(X_train_feat)
    train_acc = accuracy_score(y_train, train_pred)

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    cv_scores = []
    fold = 1

    for train_idx, val_idx in gkf.split(X_train_feat, y_train, groups=groups_train):
        X_tr_fold, X_val_fold = X_train_feat[train_idx], X_train_feat[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        rf_fold = RandomForestClassifier(
            n_estimators=RF_FINETUNED_ESTIMATORS,
            max_depth=RF_FINETUNED_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=-1, random_state=42,
        )
        rf_fold.fit(X_tr_fold, y_tr_fold)
        score = rf_fold.score(X_val_fold, y_val_fold)
        cv_scores.append(score)
        print(f"Fold {fold}: Accuracy = {score:.4f} "
              f"(Hidden Subjects: {np.unique(groups_train[val_idx])})")
        fold += 1

    cv_acc_mean = np.mean(cv_scores)
    print(f"\n1. Training Accuracy:       {train_acc:.4f}")
    print(f"2. Cross-Validation Mean:   {cv_acc_mean:.4f}")

    if test_df is not None:
        X_test_raw, y_test, _ = get_windows_with_subjects(test_df)
        X_test_feat = extract_refined_features(X_test_raw)
        test_pred = rf.predict(X_test_feat)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"3. Generalization (Test):   {test_acc:.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred, zero_division=0))


def train_rf(
    data_path: str,
    output_path: str,
    n_estimators: int = RF_FINETUNED_ESTIMATORS,
    max_depth: int = RF_FINETUNED_MAX_DEPTH,
    features: str = 'refined',
):
    """
    Train Random Forest model from .npz data file.
    """
    X_windows, y = load_windows(data_path)

    if features == 'enmo_baseline':
        X_features = extract_features_enmo(X_windows)
    elif features == 'refined':
        X_features = extract_refined_features(X_windows)
    else:
        from src.features.builder import build_feature_matrix
        X_features = build_feature_matrix(X_windows, feature_type=features, include_enmo=True)

    model = RandomForestActivityModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    )
    model.fit(X_features, y)

    y_pred = model.predict(X_features)
    acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {acc:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        print(f"Model saved to {output_path}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Random Forest for activity classification')
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save model')
    parser.add_argument('--n-estimators', type=int, default=RF_FINETUNED_ESTIMATORS)
    parser.add_argument('--max-depth', type=int, default=RF_FINETUNED_MAX_DEPTH)
    parser.add_argument('--features', type=str, default='refined',
                        choices=['enmo_baseline', 'refined', 'statistical'])

    args = parser.parse_args()

    train_rf(
        data_path=args.data_path,
        output_path=args.output_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        features=args.features,
    )
