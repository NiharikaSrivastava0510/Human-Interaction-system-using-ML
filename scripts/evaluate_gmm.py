"""
Script to evaluate the trained GMM model on the test set.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    adjusted_rand_score, normalized_mutual_info_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import load_windows
from src.features.builder import extract_refined_features
from src.models.gmm import GMMActivityModel
from src.utils.config import ACTIVITY_LABELS


def evaluate_gmm(
    gmm_model_path: str,
    test_data_path: str,
    output_dir: str = 'results',
):
    """
    Evaluate pre-trained GMM model on test set.
    """
    print("=" * 60)
    print("GMM Model Evaluation on Test Set")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    X_windows, y_test = load_windows(test_data_path)
    print(f"Test windows shape: {X_windows.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Extract refined features
    print("Extracting refined features...")
    X_refined = extract_refined_features(X_windows)
    print(f"Refined features shape: {X_refined.shape}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_refined)
    
    # Load model
    print(f"Loading GMM model from {gmm_model_path}...")
    gmm_model = GMMActivityModel()
    gmm_model.load(gmm_model_path)
    print(f"Model loaded. Number of components: {gmm_model.n_components}")
    
    # Make predictions
    print("Making predictions...")
    y_pred_gmm = gmm_model.predict_mapped(X_test_scaled)
    y_pred_raw = gmm_model.predict(X_test_scaled)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test, y_pred_gmm)
    ari = adjusted_rand_score(y_test, y_pred_raw)
    nmi = normalized_mutual_info_score(y_test, y_pred_raw)
    
    print(f"\nAccuracy:                    {accuracy:.4f}")
    print(f"Adjusted Rand Index (ARI):   {ari:.4f}")
    print(f"Normalized Mutual Info:      {nmi:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred_gmm, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_gmm)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'ari': float(ari),
        'nmi': float(nmi),
        'n_components': int(gmm_model.n_components),
        'test_samples': int(len(y_test)),
        'confusion_matrix': cm.tolist(),
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'gmm_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('GMM Confusion Matrix - Test Set')
    cm_path = output_dir / 'gmm_confusion_matrix.png'
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GMM model on test set')
    parser.add_argument('--gmm-model', type=str, required=True, help='Path to GMM model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    evaluate_gmm(
        gmm_model_path=args.gmm_model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
    )
