"""
Script to train Gaussian Mixture Model for activity classification.
"""

import argparse
import numpy as np
from pathlib import Path
from src.data.windowing import load_windows
from src.features.builder import build_feature_matrix
from src.models.gmm import GMMActivityModel
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import logger


def train_gmm(
    data_path: str,
    output_path: str,
    n_components: int = 7,
    features: str = 'statistical',
):
    """
    Train GMM model.
    
    Args:
        data_path: Path to processed data file
        output_path: Path to save model
        n_components: Number of mixture components
        features: Feature type ('statistical' or 'combined')
    """
    logger.info("Loading data...")
    X_windows, y = load_windows(data_path)
    
    logger.info("Building features...")
    X_features = build_feature_matrix(X_windows, feature_type=features, include_enmo=True)
    
    logger.info(f"Training GMM with {n_components} components...")
    model = GMMActivityModel(n_components=n_components)
    model.fit(X_features)
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_features)
    
    # Calculate clustering metrics
    from src.evaluation.metrics import calculate_clustering_metrics
    metrics = calculate_clustering_metrics(y, y_pred)
    
    logger.info(f"Metrics: {metrics}")
    
    # Save model
    logger.info(f"Saving model to {output_path}")
    model.save(output_path)
    
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GMM for activity classification')
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save model')
    parser.add_argument('--n-components', type=int, default=7, help='Number of components')
    parser.add_argument('--features', type=str, default='statistical', help='Feature type')
    
    args = parser.parse_args()
    
    train_gmm(
        data_path=args.data_path,
        output_path=args.output_path,
        n_components=args.n_components,
        features=args.features,
    )
