"""
Model evaluation visualization: confusion matrices, performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, List


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    figsize: tuple = (10, 8),
    normalize: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size
        normalize: Whether to normalize to percentages
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compare performance across multiple models.
    
    Args:
        results: Dictionary of model_name -> metrics dictionary
        metric: Which metric to compare
        figsize: Figure size
        save_path: Path to save figure
    """
    model_names = list(results.keys())
    values = [results[model][metric] for model in model_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)],
                  alpha=0.7)
    
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Model Comparison: {metric.capitalize()}')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_per_class_metrics(
    per_class_metrics: Dict[int, Dict[str, float]],
    class_names: list = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot per-class performance metrics.
    
    Args:
        per_class_metrics: Dictionary of class_id -> metrics dict
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in per_class_metrics.keys()]
    
    metrics_dict = {'precision': [], 'recall': [], 'f1': []}
    
    for class_id in sorted(per_class_metrics.keys()):
        for metric in metrics_dict.keys():
            metrics_dict[metric].append(per_class_metrics[class_id].get(metric, 0))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for idx, (metric, values) in enumerate(metrics_dict.items()):
        ax.bar(x + idx*width, values, width, label=metric.capitalize(), alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
