"""
Cross-validation utilities, including Group K-Fold for subject-level splits.

Matches the notebook's GroupKFold approach which prevents data leakage
by ensuring no subject's data appears in both train and validation splits.
"""

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Generator, Tuple

from src.utils.config import CV_N_SPLITS


class GroupKFoldSplitter:
    """Custom Group K-Fold splitter for cross-validation."""

    def __init__(self, n_splits: int = CV_N_SPLITS, shuffle: bool = False, random_state: int = None):
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
    n_splits: int = CV_N_SPLITS,
    use_group_kfold: bool = True,
    scale_features: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Perform cross-validation and collect metrics.

    Matches the notebook's CV approach for all 3 models:
    - GroupKFold to prevent subject data leakage
    - Per-fold train and validation accuracy
    - Mean and std reporting

    Args:
        model: Model with fit() and predict() methods
        X: Feature matrix
        y: Labels
        groups: Group IDs for Group K-Fold (required if use_group_kfold=True)
        n_splits: Number of folds
        use_group_kfold: Whether to use Group K-Fold
        scale_features: Whether to scale features per fold (for GMM)
        verbose: Whether to print per-fold results

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
    train_scores = []
    val_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Optionally scale features per fold (as in notebook's GMM CV)
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_test, y_pred_test)
        train_scores.append(train_acc)
        val_scores.append(val_acc)

        # Full metrics for validation
        metrics = calculate_metrics(y_test, y_pred_test)
        metrics['fold'] = fold_idx
        metrics['train_accuracy'] = train_acc
        fold_results.append(metrics)

        if verbose:
            hidden_subjects = np.unique(groups[test_idx]) if groups is not None else 'N/A'
            print(f"Fold {fold_idx + 1}: Train Acc = {train_acc:.4f} | "
                  f"Val Acc = {val_acc:.4f} "
                  f"(Hidden Subjects: {hidden_subjects})")

    # Aggregate results
    results = {
        'fold_results': fold_results,
        'mean_train_accuracy': float(np.mean(train_scores)),
        'std_train_accuracy': float(np.std(train_scores)),
        'mean_accuracy': float(np.mean(val_scores)),
        'std_accuracy': float(np.std(val_scores)),
        'mean_f1': float(np.mean([r['f1'] for r in fold_results])),
        'std_f1': float(np.std([r['f1'] for r in fold_results])),
    }

    if verbose:
        print(f"\nAverage Training Accuracy: {results['mean_train_accuracy']:.4f} "
              f"+/- {results['std_train_accuracy']:.4f}")
        print(f"Average CV Accuracy:       {results['mean_accuracy']:.4f} "
              f"+/- {results['std_accuracy']:.4f}")

    return results
