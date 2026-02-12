"""
Script to evaluate and compare all models.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from src.data.windowing import load_windows
from src.features.builder import build_feature_matrix
from src.models.gmm import GMMActivityModel
from src.models.cnn import FineTunedCNN
from src.models.random_forest import RandomForestActivityModel
from src.evaluation.metrics import calculate_metrics, get_confusion_matrix
from src.utils.logger import logger


def evaluate_all(
    gmm_model_path: str,
    cnn_model_path: str,
    rf_model_path: str,
    test_data_path: str,
    output_dir: str = 'results',
):
    """
    Evaluate all three models and generate comparison report.
    
    Args:
        gmm_model_path: Path to trained GMM model
        cnn_model_path: Path to trained CNN model
        rf_model_path: Path to trained Random Forest model
        test_data_path: Path to test data
        output_dir: Output directory for results
    """
    logger.info("Loading test data...")
    X_windows, y_test = load_windows(test_data_path)
    
    results = {}
    
    # Evaluate CNN (uses raw windows)
    logger.info("Evaluating CNN...")
    cnn_model = FineTunedCNN()
    cnn_model.load(cnn_model_path)
    y_pred_cnn = cnn_model.predict(X_windows)
    results['cnn'] = calculate_metrics(y_test, y_pred_cnn)
    logger.info(f"CNN Metrics: {results['cnn']}")
    
    # Evaluate GMM and Random Forest (use features)
    X_features = build_feature_matrix(X_windows, feature_type='statistical', include_enmo=True)
    
    logger.info("Evaluating GMM...")
    gmm_model = GMMActivityModel()
    gmm_model.load(gmm_model_path)
    y_pred_gmm = gmm_model.predict(X_features)
    results['gmm'] = calculate_metrics(y_test, y_pred_gmm)
    logger.info(f"GMM Metrics: {results['gmm']}")
    
    logger.info("Evaluating Random Forest...")
    rf_model = RandomForestActivityModel()
    rf_model.load(rf_model_path)
    y_pred_rf = rf_model.predict(X_features)
    results['rf'] = calculate_metrics(y_test, y_pred_rf)
    logger.info(f"Random Forest Metrics: {results['rf']}")
    
    # Save results
    output_path = Path(output_dir) / 'evaluation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all models')
    parser.add_argument('--gmm-model', type=str, required=True, help='Path to GMM model')
    parser.add_argument('--cnn-model', type=str, required=True, help='Path to CNN model')
    parser.add_argument('--rf-model', type=str, required=True, help='Path to Random Forest model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_all(
        gmm_model_path=args.gmm_model,
        cnn_model_path=args.cnn_model,
        rf_model_path=args.rf_model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
    )
