"""
Cross-validation utilities, including Group K-Fold for subject-level splits.
"""

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from typing import Generator, Tuple


class GroupKFoldSplitter:
    """Custom Group K-Fold splitter for cross-validation."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None):
        """
        Initialize Group K-Fold splitter.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle groups
            random_state: Random seed
        """
        self.splitter = GroupKFold(n_splits=n_splits)
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            groups: Group IDs (e.g., subject IDs)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        for train_idx, test_idx in self.splitter.split(X, y, groups):
            yield train_idx, test_idx


def perform_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray = None,
    n_splits: int = 5,
    use_group_kfold: bool = True
) -> dict:
    """
    Perform cross-validation and collect metrics.
    
    Args:
        model: Model with fit() and predict() methods
        X: Feature matrix
        y: Labels
        groups: Group IDs for Group K-Fold (required if use_group_kfold=True)
        n_splits: Number of folds
        use_group_kfold: Whether to use Group K-Fold
        
    Returns:
        Dictionary with cross-validation results
    """
    from .metrics import calculate_metrics
    
    if use_group_kfold:
        if groups is None:
            raise ValueError("groups parameter required for Group K-Fold")
        splitter = GroupKFoldSplitter(n_splits=n_splits)
        splits = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = splitter.split(X, y)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        metrics['fold'] = fold_idx
        fold_results.append(metrics)
    
    # Aggregate results
    results = {
        'fold_results': fold_results,
        'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
        'mean_f1': np.mean([r['f1'] for r in fold_results]),
        'std_f1': np.std([r['f1'] for r in fold_results]),
    }
    
    return results
