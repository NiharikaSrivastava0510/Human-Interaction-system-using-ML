"""
Script to train Random Forest model for activity classification.
"""

import argparse
import numpy as np
from pathlib import Path
from src.data.windowing import load_windows
from src.features.builder import build_feature_matrix
from src.models.random_forest import RandomForestActivityModel
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import logger


def train_rf(
    data_path: str,
    output_path: str,
    n_estimators: int = 200,
    max_depth: int = 20,
    features: str = 'statistical',
):
    """
    Train Random Forest model.
    
    Args:
        data_path: Path to processed data file
        output_path: Path to save model
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        features: Feature type
    """
    logger.info("Loading data...")
    X_windows, y = load_windows(data_path)
    
    logger.info("Building features...")
    X_features = build_feature_matrix(X_windows, feature_type=features, include_enmo=True)
    
    logger.info(f"Training Random Forest with {n_estimators} estimators, max depth {max_depth}...")
    model = RandomForestActivityModel(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_features, y)
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_features)
    
    metrics = calculate_metrics(y, y_pred)
    logger.info(f"Training metrics: {metrics}")
    
    logger.info(f"Saving model to {output_path}")
    model.save(output_path)
    
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Random Forest for activity classification')
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save model')
    parser.add_argument('--n-estimators', type=int, default=200, help='Number of estimators')
    parser.add_argument('--max-depth', type=int, default=20, help='Max depth')
    parser.add_argument('--features', type=str, default='statistical', help='Feature type')
    
    args = parser.parse_args()
    
    train_rf(
        data_path=args.data_path,
        output_path=args.output_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        features=args.features,
    )
