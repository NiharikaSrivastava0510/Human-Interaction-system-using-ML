"""
Error analysis and confusion matrix utilities.
"""

import numpy as np
from .metrics import get_confusion_matrix


def analyze_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None
) -> dict:
    """
    Analyze confusion matrix to identify error patterns.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        
    Returns:
        Dictionary with error analysis
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    analysis = {
        'total_samples': int(cm.sum()),
        'correct_predictions': int(np.trace(cm)),
        'accuracy': float(np.trace(cm) / cm.sum()),
        'top_errors': [],
    }
    
    # Find top confusion pairs
    errors = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                errors.append({
                    'true_class': i,
                    'pred_class': j,
                    'count': int(cm[i, j]),
                    'true_label': class_names[i],
                    'pred_label': class_names[j],
                })
    
    # Sort by frequency
    errors.sort(key=lambda x: x['count'], reverse=True)
    analysis['top_errors'] = errors[:10]  # Top 10 confusion pairs
    
    return analysis


def get_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray = None
) -> dict:
    """
    Get indices and statistics of misclassified samples.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        X: Feature matrix (optional)
        
    Returns:
        Dictionary with misclassification info
    """
    misclassified = y_true != y_pred
    misclass_indices = np.where(misclassified)[0]
    
    result = {
        'num_misclassified': int(misclassified.sum()),
        'error_rate': float(misclassified.sum() / len(y_true)),
        'misclassified_indices': misclass_indices.tolist(),
        'misclassified_true': y_true[misclassified].tolist(),
        'misclassified_pred': y_pred[misclassified].tolist(),
    }
    
    return result
