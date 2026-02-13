"""
Script to evaluate the trained Random Forest model on the test set.
"""

import argparse
import json
import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import load_windows
from src.features.builder import extract_refined_features
from src.models.random_forest import RandomForestActivityModel
from src.utils.config import ACTIVITY_LABELS


def evaluate_rf(
    rf_model_path: str,
    test_data_path: str,
    output_dir: str = 'results',
):
    """
    Evaluate pre-trained Random Forest model on test set.
    """
    print("=" * 60)
    print("Random Forest Model Evaluation on Test Set")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    X_windows, y_test_orig = load_windows(test_data_path)
    print(f"Test windows shape: {X_windows.shape}")
    print(f"Test labels shape: {y_test_orig.shape}")
    
    # Extract refined features
    print("Extracting refined features...")
    X_test_feat = extract_refined_features(X_windows)
    print(f"Refined features shape: {X_test_feat.shape}")
    
    # Load model
    print(f"\nLoading Random Forest model from {rf_model_path}...")
    rf_model = RandomForestActivityModel()
    rf_model.load(rf_model_path)
    print(f"Model loaded successfully")
    
    # Get model details
    print(f"Number of estimators: {rf_model.model.n_estimators}")
    print(f"Max depth: {rf_model.model.max_depth}")
    print(f"Min samples leaf: {rf_model.model.min_samples_leaf}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = rf_model.predict(X_test_feat)
    y_pred_proba = rf_model.predict_proba(X_test_feat)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test_orig, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    report = classification_report(y_test_orig, y_pred, zero_division=0, output_dict=False)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred)
    
    # Feature importance
    feature_importance = rf_model.get_feature_importance()
    feat_names = (
        [f'{axis}_mean' for axis in ['bx', 'by', 'bz', 'tx', 'ty', 'tz']] +
        [f'{axis}_std' for axis in ['bx', 'by', 'bz', 'tx', 'ty', 'tz']] +
        ['enmo_b_mean', 'enmo_b_max', 'enmo_t_mean', 'enmo_t_max']
    )
    
    print("\n" + "-" * 60)
    print("Top 10 Feature Importance:")
    print("-" * 60)
    for name, imp in sorted(zip(feat_names, feature_importance), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {imp:.4f}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'test_samples': int(len(y_test_orig)),
        'confusion_matrix': cm.tolist(),
        'model_params': {
            'n_estimators': int(rf_model.model.n_estimators),
            'max_depth': rf_model.model.max_depth,
            'min_samples_leaf': int(rf_model.model.min_samples_leaf),
        },
        'feature_importance': {name: float(imp) for name, imp in zip(feat_names, feature_importance)}
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'rf_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save confusion matrix visualization
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(y_test_orig)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Random Forest Confusion Matrix - Test Set')
    plt.tight_layout()
    cm_path = output_dir / 'rf_confusion_matrix.png'
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    # Save feature importance visualization
    plt.figure(figsize=(10, 8))
    sorted_indices = np.argsort(feature_importance)[::-1][:15]
    plt.barh(range(len(sorted_indices)), feature_importance[sorted_indices])
    plt.yticks(range(len(sorted_indices)), [feat_names[i] for i in sorted_indices])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importance - Random Forest')
    plt.tight_layout()
    fi_path = output_dir / 'rf_feature_importance.png'
    plt.savefig(fi_path, dpi=100, bbox_inches='tight')
    print(f"Feature importance plot saved to {fi_path}")
    plt.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Random Forest model on test set')
    parser.add_argument('--rf-model', type=str, required=True, help='Path to Random Forest model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    evaluate_rf(
        rf_model_path=args.rf_model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
    )
