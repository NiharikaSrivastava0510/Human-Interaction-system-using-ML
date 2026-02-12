"""
Script to train Convolutional Neural Network for activity classification.
"""

import argparse
import numpy as np
from pathlib import Path
from src.data.windowing import load_windows
from src.models.cnn import FineTunedCNN
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import logger


def train_cnn(
    data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.2,
):
    """
    Train CNN model.
    
    Args:
        data_path: Path to processed data file
        output_path: Path to save model
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
    """
    logger.info("Loading data...")
    X_windows, y = load_windows(data_path)
    
    logger.info(f"Training CNN with {epochs} epochs, batch size {batch_size}...")
    model = FineTunedCNN(num_classes=7)
    
    model.fit(
        X_windows, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_windows)
    
    metrics = calculate_metrics(y, y_pred)
    logger.info(f"Training metrics: {metrics}")
    
    logger.info(f"Saving model to {output_path}")
    model.save(output_path)
    
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN for activity classification')
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    train_cnn(
        data_path=args.data_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
