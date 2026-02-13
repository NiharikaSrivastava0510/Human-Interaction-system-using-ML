"""
Script to train CNN / Neural Network for activity classification.

Matches the notebook's CNN pipeline:
- 3.2.a: Baseline CNN on raw scaled sensor data (10 epochs)
- 3.2.b: Fine-tuned feature NN (MLP on 16 extracted features, 50 epochs)
- 3.2.c: Cross-validation and error analysis with Random Forest on features
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
from src.models.cnn import (
    build_baseline_cnn, build_finetuned_feature_nn, get_training_callbacks
)
from src.utils.config import (
    CNN_BASELINE_EPOCHS, CNN_FINETUNED_EPOCHS, CNN_BATCH_SIZE, NUM_CLASSES
)


def train_cnn_baseline(train_df, test_df, num_classes=None):
    """
    Train baseline CNN on raw scaled sensor data (notebook section 3.2.a).
    """
    X_train_raw, y_train_orig = get_windows(train_df)
    X_test_raw, y_test_orig = get_windows(test_df)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_orig)
    y_test = le.transform(y_test_orig)
    if num_classes is None:
        num_classes = len(np.unique(y_train))

    # Also train RF baseline for comparison
    X_train_feat = extract_features_enmo(X_train_raw)
    X_test_feat = extract_features_enmo(X_test_raw)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_feat, y_train)
    y_pred_rf = rf.predict(X_test_feat)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Feature Baseline (RF + 10 ENMO Features) Accuracy: {acc_rf:.4f}")

    # Scale raw data for CNN
    scaler = StandardScaler()
    N_train, T, F = X_train_raw.shape
    N_test = X_test_raw.shape[0]
    X_train_cnn = scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(N_train, T, F)
    X_test_cnn = scaler.transform(X_test_raw.reshape(-1, F)).reshape(N_test, T, F)

    # Build and train CNN
    model = build_baseline_cnn(input_shape=(T, F), num_classes=num_classes)
    history = model.fit(
        X_train_cnn, y_train,
        epochs=CNN_BASELINE_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_data=(X_test_cnn, y_test),
        verbose=1,
    )

    loss, acc_cnn = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"CNN Baseline Accuracy: {acc_cnn:.4f}")

    y_pred_cnn = np.argmax(model.predict(X_test_cnn), axis=1)
    print("\nCNN Classification Report:")
    print(classification_report(y_test, y_pred_cnn,
                                target_names=le.classes_.astype(str), zero_division=0))

    return model, le


def train_cnn_finetuned(train_df, test_df, num_classes=None):
    """
    Train fine-tuned feature NN on 16 extracted features (notebook section 3.2.b).
    """
    X_train_raw, y_train_orig = get_windows(train_df)
    X_test_raw, y_test_orig = get_windows(test_df)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_orig)
    y_test = le.transform(y_test_orig)
    if num_classes is None:
        num_classes = len(np.unique(y_train))

    # Extract 16 optimized features
    X_train_feat = extract_optimized_features(X_train_raw)
    X_test_feat = extract_optimized_features(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # Build and train fine-tuned feature NN
    model = build_finetuned_feature_nn(input_dim=16, num_classes=num_classes)
    early_stop = get_training_callbacks()

    history = model.fit(
        X_train_scaled, y_train,
        epochs=CNN_FINETUNED_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_data=(X_test_scaled, y_test),
        callbacks=early_stop,
        verbose=1,
    )

    _, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Fine-Tuned Feature NN Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_.astype(str), zero_division=0))

    return model, le


def train_cnn_with_cv(train_df, test_df=None):
    """
    Train with cross-validation and error analysis (notebook section 3.2.c/d).

    Uses Random Forest on 16 refined features with GroupKFold CV.
    """
    X_train_raw, y_train, groups_train = get_windows_with_subjects(train_df)
    X_train_feat = extract_refined_features(X_train_raw)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X_train_feat, y_train)
    train_pred = rf.predict(X_train_feat)
    train_acc = accuracy_score(y_train, train_pred)

    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    fold = 1

    for train_idx, val_idx in gkf.split(X_train_feat, y_train, groups=groups_train):
        X_tr_fold = X_train_feat[train_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        rf_fold = RandomForestClassifier(n_estimators=200, max_depth=20,
                                         min_samples_leaf=2, n_jobs=-1, random_state=42)
        rf_fold.fit(X_tr_fold, y_tr_fold)
        score = rf_fold.score(X_train_feat[val_idx], y_val_fold)
        cv_scores.append(score)
        print(f"Fold {fold}: Accuracy = {score:.4f}")
        fold += 1

    print(f"Training Accuracy:       {train_acc:.4f}")
    print(f"Cross-Validation Mean:   {np.mean(cv_scores):.4f}")

    if test_df is not None:
        X_test_raw, y_test, _ = get_windows_with_subjects(test_df)
        X_test_feat = extract_refined_features(X_test_raw)
        test_pred = rf.predict(X_test_feat)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"Generalization (Test):   {test_acc:.4f}")
        print(classification_report(y_test, test_pred, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN for activity classification')
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=CNN_FINETUNED_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=CNN_BATCH_SIZE, help='Batch size')
    parser.add_argument('--mode', type=str, default='finetuned',
                        choices=['baseline', 'finetuned'], help='Training mode')

    args = parser.parse_args()

    X_windows, y = load_windows(args.data_path)

    if args.mode == 'baseline':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        num_classes = len(np.unique(y_enc))

        scaler = StandardScaler()
        N, T, F = X_windows.shape
        X_scaled = scaler.fit_transform(X_windows.reshape(-1, F)).reshape(N, T, F)

        model = build_baseline_cnn((T, F), num_classes)
        model.fit(X_scaled, y_enc, epochs=args.epochs,
                  batch_size=args.batch_size, verbose=1)
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        num_classes = len(np.unique(y_enc))

        X_feat = extract_optimized_features(X_windows)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feat)

        model = build_finetuned_feature_nn(16, num_classes)
        model.fit(X_scaled, y_enc, epochs=args.epochs,
                  batch_size=args.batch_size, verbose=1)

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output_path)
        print(f"Model saved to {args.output_path}")
