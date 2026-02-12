"""
Script to evaluate and compare all three models.

Matches the notebook's evaluation approach:
- GMM: Cluster-to-label mapping, ARI, accuracy
- CNN: Baseline CNN + fine-tuned feature NN
- Random Forest: Baseline + fine-tuned with feature importance
- All models: GroupKFold cross-validation, confusion matrices, error analysis
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    adjusted_rand_score, normalized_mutual_info_score
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import get_windows, get_windows_with_subjects, load_windows
from src.features.builder import (
    extract_features_enmo, extract_refined_features, extract_optimized_features
)
from src.models.gmm import GMMActivityModel
from src.models.random_forest import RandomForestActivityModel
from src.evaluation.metrics import calculate_metrics, calculate_clustering_metrics
from src.utils.config import ACTIVITY_LABELS


def evaluate_all_from_dataframes(train_df, test_df, output_dir='results'):
    """
    Evaluate all three models from DataFrames (matching notebook flow).
    """
    print("=" * 60)
    print("Evaluating All Models")
    print("=" * 60)

    X_train_raw, y_train = get_windows(train_df)
    X_test_raw, y_test = get_windows(test_df)

    results = {}

    # ── 1. GMM ─────────────────────────────────────────────
    print("\n--- GMM ---")
    X_train_feat_refined = extract_refined_features(X_train_raw)
    X_test_feat_refined = extract_refined_features(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat_refined)
    X_test_scaled = scaler.transform(X_test_feat_refined)

    gmm = GMMActivityModel.find_optimal_k(X_train_scaled, y_train)
    y_pred_gmm = gmm.predict_mapped(X_test_scaled)
    gmm_acc = accuracy_score(y_test, y_pred_gmm)
    gmm_ari = adjusted_rand_score(y_test, gmm.predict(X_test_scaled))
    print(f"GMM Test Accuracy: {gmm_acc:.4f}")
    print(f"GMM Test ARI:      {gmm_ari:.4f}")
    results['gmm'] = {
        'accuracy': gmm_acc,
        'ari': gmm_ari,
        'n_components': gmm.n_components,
    }

    
    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Accuracy':>10}")
    print("-" * 42)
    print(f"{'GMM (BIC-optimized)':<30} {gmm_acc:>10.4f}")


    # Save results
    output_path = Path(output_dir) / 'evaluation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {output_path}")

    return results


def evaluate_all(
    gmm_model_path: str,
    # rf_model_path: str,
    test_data_path: str,
    output_dir: str = 'results',
):
    """
    Evaluate pre-trained GMM and RF models from saved files.
    """
    print("Loading test data...")
    X_windows, y_test = load_windows(test_data_path)
    X_refined = extract_refined_features(X_windows)

    results = {}

    # GMM
    print("Evaluating GMM...")
    gmm_model = GMMActivityModel()
    gmm_model.load(gmm_model_path)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_refined)
    y_pred_gmm = gmm_model.predict_mapped(X_test_scaled)
    results['gmm'] = calculate_metrics(y_test, y_pred_gmm)
    results['gmm']['ari'] = float(adjusted_rand_score(y_test, gmm_model.predict(X_test_scaled)))
    print(f"GMM Metrics: {results['gmm']}")

   

    # Save results
    output_path = Path(output_dir) / 'evaluation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Results saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all models')
    parser.add_argument('--gmm-model', type=str, required=True, help='Path to GMM model')
    # parser.add_argument('--rf-model', type=str, required=True, help='Path to RF model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    evaluate_all(
        gmm_model_path=args.gmm_model,
        # rf_model_path=args.rf_model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
    )
