"""
Neural network model architecture for NFL spread prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional


def create_model(input_dim: int, hidden_units: list = [128, 64, 32], dropout_rate: float = 0.3) -> keras.Model:
    """
    Create a feedforward neural network for binary classification.
    
    Args:
        input_dim: Number of input features
        hidden_units: List of hidden layer sizes (default: [128, 64, 32])
        dropout_rate: Dropout rate for regularization (default: 0.3)
    
    Returns:
        Compiled Keras model
    """
    # Build model layer by layer for proper structure
    model = keras.Sequential()
    
    # Input + first hidden layer
    model.add(layers.Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dropout(dropout_rate))
    
    # Additional hidden layers
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def load_model(model_path: str) -> keras.Model:
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)


def save_model(model: keras.Model, model_path: str):
    """
    Save a model to disk.
    
    Args:
        model: Keras model to save
        model_path: Path where to save the model
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

