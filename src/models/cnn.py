"""
Convolutional Neural Network models for Human Activity Recognition.

Matches the notebook's CNN pipeline:
- Baseline: Conv1D(64, 3) → MaxPool(2) → Flatten → Dropout(0.5) → Dense(softmax)
  Trained on raw scaled sensor windows for 10 epochs.
- Fine-tuned: 3-layer MLP with BatchNorm operating on 16-feature extracted set.
  Dense(128) → BN → Dropout(0.3) → Dense(64) → BN → Dropout(0.3) → Dense(softmax)
  Trained with EarlyStopping for up to 50 epochs.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from src.utils.config import (
    CNN_BASELINE_EPOCHS, CNN_FINETUNED_EPOCHS, CNN_BATCH_SIZE,
    CNN_LEARNING_RATE, CNN_BASELINE_FILTERS, CNN_BASELINE_KERNEL_SIZE,
    CNN_BASELINE_DROPOUT, CNN_FINETUNED_HIDDEN_1, CNN_FINETUNED_HIDDEN_2,
    CNN_FINETUNED_DROPOUT, NUM_CLASSES
)


def build_baseline_cnn(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Baseline CNN: single convolutional layer on raw sensor data.

    Matches notebook section 3.2.a:
        Conv1D(64, 3, relu) → MaxPool(2) → Flatten → Dropout(0.5) → Dense(softmax)

    Args:
        input_shape: (window_size, num_channels) e.g. (100, 6)
        num_classes: Number of activity classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Conv1D(
            filters=CNN_BASELINE_FILTERS,
            kernel_size=CNN_BASELINE_KERNEL_SIZE,
            activation='relu',
            input_shape=input_shape
        ),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(CNN_BASELINE_DROPOUT),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_finetuned_feature_nn(
    input_dim: int = 16,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    Fine-tuned feature-based MLP (not a CNN).

    Matches notebook section 3.2.b: operates on the 16-feature extracted set
    (not raw time-series), using a 3-layer MLP with BatchNormalization.

    Architecture:
        Input(16) → Dense(128, relu) → BN → Dropout(0.3)
                   → Dense(64, relu) → BN → Dropout(0.3)
                   → Dense(num_classes, softmax)

    Args:
        input_dim: Number of input features (default: 16)
        num_classes: Number of activity classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(CNN_FINETUNED_HIDDEN_1, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(CNN_FINETUNED_DROPOUT),
        layers.Dense(CNN_FINETUNED_HIDDEN_2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(CNN_FINETUNED_DROPOUT),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_baseline_nn(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    NN Baseline Architecture matching notebook section 3.1.a / 3.3.a.

    Simple Conv1D baseline used as reference for comparison.

    Args:
        input_shape: (window_size, num_channels) e.g. (100, 6)
        num_classes: Number of activity classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Conv1D(
            filters=64, kernel_size=3, activation='relu',
            input_shape=input_shape
        ),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_training_callbacks(
    model_path: str = "models/cnn/cnn_finetuned.h5",
) -> list:
    """
    Training callbacks for the fine-tuned model.

    Matches notebook: EarlyStopping on val_loss with patience=10.
    """
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
        ),
    ]
