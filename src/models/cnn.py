"""
Convolutional Neural Network models for Human Activity Recognition.

Baseline: Single Conv1D layer → MaxPool → Flatten → Dense
Fine-tuned: 3-block Conv1D with BatchNorm, Dropout, and Global Average Pooling
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.utils.config import (
    CNN_EPOCHS, CNN_BATCH_SIZE, CNN_LEARNING_RATE, CNN_DROPOUT_RATE,
    CNN_FILTERS, CNN_KERNEL_SIZES, CNN_POOL_SIZES, NUM_CLASSES
)


def build_baseline_cnn(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Baseline CNN: single convolutional layer.

    Architecture:
        Conv1D(64, 3) → MaxPool(2) → Flatten → Dense(num_classes, softmax)

    Args:
        input_shape: (window_size, num_channels) e.g. (100, 6)
        num_classes: Number of activity classes

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_finetuned_cnn(input_shape: tuple, num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Fine-tuned CNN with 3 convolutional blocks.

    Architecture:
        Block 1: Conv1D(64, 3) → BatchNorm → ReLU → MaxPool(2) → Dropout
        Block 2: Conv1D(128, 3) → BatchNorm → ReLU → MaxPool(2) → Dropout
        Block 3: Conv1D(256, 3) → BatchNorm → ReLU → MaxPool(2) → Dropout
        → GlobalAveragePooling1D → Dense(num_classes, softmax)

    Design rationale:
        - Block 1 captures simple patterns (spikes, sudden accelerations)
        - Block 2 captures medium patterns (steps, cycles)
        - Block 3 captures complex gait patterns (walking vs stair climbing)
        - BatchNorm stabilises learning and allows faster convergence
        - GAP reduces overfitting compared to Flatten
        - Dropout (0.3-0.5) forces learning of general features

    Args:
        input_shape: (window_size, num_channels) e.g. (100, 6)
        num_classes: Number of activity classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Three convolutional blocks with increasing filter count
    for i, (filters, kernel_size, pool_size) in enumerate(
        zip(CNN_FILTERS, CNN_KERNEL_SIZES, CNN_POOL_SIZES)
    ):
        x = layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(pool_size=pool_size)(x)
        x = layers.Dropout(CNN_DROPOUT_RATE if i < 2 else 0.5)(x)

    # Global Average Pooling instead of Flatten to reduce overfitting
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_training_callbacks(model_path: str = "models/cnn/cnn_finetuned.h5") -> list:
    """Standard training callbacks for CNN."""
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
