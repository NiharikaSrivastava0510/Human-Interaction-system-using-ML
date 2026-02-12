"""
Sensor time-series and waveform visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_sensor_waveform(
    X_window: np.ndarray,
    label: int = None,
    sensor_names: list = None,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot raw sensor waveforms for a single window.
    
    Args:
        X_window: Single window of shape (window_size, 6)
        label: Activity label
        sensor_names: List of sensor names
        figsize: Figure size
        save_path: Path to save figure
    """
    if sensor_names is None:
        sensor_names = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for channel in range(X_window.shape[1]):
        axes[channel].plot(X_window[:, channel], linewidth=1, color='steelblue')
        axes[channel].set_title(sensor_names[channel])
        axes[channel].set_ylabel('Acceleration (g)')
        axes[channel].grid(True, alpha=0.3)
    
    fig.suptitle(f'Sensor Waveforms - Label: {label}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_activity_comparison(
    X_windows: np.ndarray,
    y_labels: np.ndarray,
    activity_pairs: list = None,
    sensor_idx: int = 0,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Compare sensor waveforms between two activities.
    
    Args:
        X_windows: Array of windows of shape (N, window_size, 6)
        y_labels: Activity labels
        activity_pairs: List of (class1, class2) tuples to compare
        sensor_idx: Which sensor to plot (0-5)
        figsize: Figure size
        save_path: Path to save figure
    """
    if activity_pairs is None:
        activity_pairs = [(0, 1)]
    
    fig, axes = plt.subplots(1, len(activity_pairs), figsize=figsize)
    if len(activity_pairs) == 1:
        axes = [axes]
    
    for idx, (class1, class2) in enumerate(activity_pairs):
        mask1 = y_labels == class1
        mask2 = y_labels == class2
        
        # Plot mean waveforms
        mean1 = X_windows[mask1, :, sensor_idx].mean(axis=0)
        mean2 = X_windows[mask2, :, sensor_idx].mean(axis=0)
        
        axes[idx].plot(mean1, label=f'Activity {class1}', linewidth=2)
        axes[idx].plot(mean2, label=f'Activity {class2}', linewidth=2)
        axes[idx].set_title(f'Activity {class1} vs {class2}')
        axes[idx].set_xlabel('Time (samples)')
        axes[idx].set_ylabel('Acceleration (g)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
