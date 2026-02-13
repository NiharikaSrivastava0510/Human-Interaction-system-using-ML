"""
Script to evaluate the trained CNN model on the test set.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import load_windows
from src.features.builder import extract_optimized_features
from src.utils.config import ACTIVITY_LABELS


def evaluate_cnn(
    cnn_model_path: str,
    train_data_path: str,
    test_data_path: str,
    output_dir: str = 'results',
):
    """
    Evaluate pre-trained CNN model on test set.
    """
    print("=" * 60)
    print("CNN Model Evaluation on Test Set")
    print("=" * 60)
    
    # Load training data to fit label encoder
    print("\nLoading training data for label encoding...")
    X_train_windows, y_train_orig = load_windows(train_data_path)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_orig)
    num_classes = len(np.unique(y_train_encoded))
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")
    
    # Load test data
    print("\nLoading test data...")
    X_test_windows, y_test_orig = load_windows(test_data_path)
    y_test_encoded = le.transform(y_test_orig)
    print(f"Test windows shape: {X_test_windows.shape}")
    print(f"Test labels shape: {y_test_encoded.shape}")
    
    # Extract optimized features
    print("Extracting optimized features...")
    X_test_feat = extract_optimized_features(X_test_windows)
    print(f"Extracted features shape: {X_test_feat.shape}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_feat)
    
    # Load model
    print(f"\nLoading CNN model from {cnn_model_path}...")
    model = keras.models.load_model(cnn_model_path)
    print(f"Model loaded successfully")
    print(f"Model summary:")
    model.summary()
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_test_scaled)
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)
    y_pred = le.inverse_transform(y_pred_encoded)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test_orig, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test_orig, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred, labels=le.classes_)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'num_classes': int(num_classes),
        'test_samples': int(len(y_test_orig)),
        'classes': list(le.classes_.astype(str)),
        'confusion_matrix': cm.tolist(),
        'model_path': str(cnn_model_path),
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'cnn_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save confusion matrix visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('CNN Confusion Matrix - Test Set')
    plt.tight_layout()
    cm_path = output_dir / 'cnn_confusion_matrix.png'
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CNN model on test set')
    parser.add_argument('--cnn-model', type=str, required=True, help='Path to CNN model')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    evaluate_cnn(
        cnn_model_path=args.cnn_model,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
    )
