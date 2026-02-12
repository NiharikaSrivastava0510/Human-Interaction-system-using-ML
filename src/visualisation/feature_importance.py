"""
Feature importance visualization for tree-based models.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    top_n: int = 15,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot feature importance from tree-based models.
    
    Args:
        importances: Feature importance scores
        feature_names: List of feature names
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save figure
    """
    # Get top features
    indices = np.argsort(importances)[-top_n:]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_importances)), top_importances, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_cumulative_importance(
    importances: np.ndarray,
    feature_names: list = None,
    threshold: float = 0.95,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot cumulative feature importance.
    
    Args:
        importances: Feature importance scores
        feature_names: List of feature names
        threshold: Cumulative importance threshold
        figsize: Figure size
        save_path: Path to save figure
    """
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    cumsum = np.cumsum(sorted_importances / sorted_importances.sum())
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumsum, linewidth=2, color='steelblue')
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'{threshold:.0%} threshold')
    ax.fill_between(range(len(cumsum)), cumsum, alpha=0.3)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance')
    ax.set_title('Cumulative Feature Importance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
