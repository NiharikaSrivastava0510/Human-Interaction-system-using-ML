"""
Error analysis and confusion matrix utilities.

Matches the notebook's error analysis sections (3.x.d):
- Confusion matrix analysis
- Error categorization (static-static, dynamic-dynamic, cross-type)
- Top confusion pairs identification
"""

import numpy as np
from .metrics import get_confusion_matrix
from src.utils.config import ACTIVITY_LABELS


# Activity categories for error analysis (matching notebook)
STATIC_ACTIVITIES = {6: 'Standing', 7: 'Sitting', 8: 'Lying'}
DYNAMIC_ACTIVITIES = {1: 'Walking', 2: 'Running', 3: 'Shuffling', 9: 'Stairs'}


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


def categorize_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: dict = None,
) -> dict:
    """
    Categorize misclassifications into error types matching the notebook.

    Categories:
    - static_static: Confusion between Standing, Sitting, Lying
    - dynamic_dynamic: Confusion between Walking, Running, Shuffling, Stairs
    - cross_type: Confusion between static and dynamic activities

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: Dict mapping label IDs to names

    Returns:
        Dictionary with error categories and counts
    """
    if label_names is None:
        label_names = ACTIVITY_LABELS

    static_labels = set(STATIC_ACTIVITIES.keys())
    dynamic_labels = set(DYNAMIC_ACTIVITIES.keys())

    errors = {
        'static_static': 0,
        'dynamic_dynamic': 0,
        'cross_type': 0,
        'total_errors': 0,
        'details': [],
    }

    misclass_mask = y_true != y_pred
    for i in np.where(misclass_mask)[0]:
        true_label = y_true[i]
        pred_label = y_pred[i]

        true_is_static = true_label in static_labels
        pred_is_static = pred_label in static_labels

        if true_is_static and pred_is_static:
            errors['static_static'] += 1
        elif not true_is_static and not pred_is_static:
            errors['dynamic_dynamic'] += 1
        else:
            errors['cross_type'] += 1

        errors['total_errors'] += 1

    # Compute percentages
    total = errors['total_errors']
    if total > 0:
        errors['static_static_pct'] = errors['static_static'] / total * 100
        errors['dynamic_dynamic_pct'] = errors['dynamic_dynamic'] / total * 100
        errors['cross_type_pct'] = errors['cross_type'] / total * 100

    return errors


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
