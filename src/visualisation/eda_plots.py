"""
Exploratory Data Analysis visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional


def plot_class_distribution(
    y: np.ndarray,
    class_names: list = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot class distribution.
    
    Args:
        y: Label array
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in unique]
    
    ax.bar(class_names, counts, color='steelblue', alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.set_xticklabels(class_names, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_correlation_heatmap(
    X: np.ndarray,
    feature_names: list = None,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
):
    """
    Plot feature correlation heatmap.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        figsize: Figure size
        save_path: Path to save figure
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_feature_distribution(
    X: np.ndarray,
    feature_names: list = None,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot distribution of features.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        figsize: Figure size
        save_path: Path to save figure
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    n_features = min(12, X.shape[1])
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(n_features):
        axes[idx].hist(X[:, idx], bins=30, alpha=0.7, color='steelblue')
        axes[idx].set_title(feature_names[idx])
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
