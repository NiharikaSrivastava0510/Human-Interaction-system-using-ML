"""
Evaluation metrics for activity classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
)
from typing import Dict


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'weighted', 'macro', or 'micro' averaging for multi-class metrics
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    
    return metrics


def calculate_clustering_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate clustering evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
        
    Returns:
        Dictionary with clustering metrics
    """
    metrics = {
        'adjusted_rand_index': float(adjusted_rand_score(y_true, y_pred)),
        'normalized_mutual_info': float(normalized_mutual_info_score(y_true, y_pred)),
    }
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = None
) -> Dict[int, Dict[str, float]]:
    """
    Calculate per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of per-class metrics
    """
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    
    per_class = {}
    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        per_class[class_id] = {
            'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        }
    
    return per_class
