#!/usr/bin/env -S uv run --
"""
Training script for NFL spread prediction model.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import tensorflow as tf
from tensorflow import keras

from model_data import prepare_training_data, get_feature_count
from nfl_spread_model import create_model, save_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "nfl_spread_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
SIMPLE_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"


def train_simple_model(season: Optional[int] = None, min_week: int = 1):
    """
    Train a simple logistic regression model.
    Better for small datasets (< 100 samples).
    """
    print("NFL Spread Prediction - Simple Model (Logistic Regression)")
    print("=" * 50)
    
    # Load and prepare data
    print("\nLoading training data...")
    X, y = prepare_training_data(season=season, min_week=min_week)
    
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Home team covers: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"Visitor team covers: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
    
    if len(X) < 5:
        print("\n❌ ERROR: Not enough data to train. Need at least 5 samples.")
        return None, None
    
    # Use leave-one-out style evaluation for very small datasets
    if len(X) < 20:
        print("\n⚠️  Very small dataset - using leave-one-out cross-validation")
        test_size = max(1, len(X) // 5)
    else:
        test_size = int(len(X) * 0.2)
    
    # Split data
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    print("\nTraining logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    
    # Predictions on test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\nTest Set Diagnostics:")
    print(f"  Actual home covers: {np.sum(y_test)} / {len(y_test)}")
    print(f"  Predicted home covers: {np.sum(y_pred)} / {len(y_pred)}")
    print(f"  Average prediction probability: {np.mean(y_prob):.4f}")
    
    # Save model and scaler
    with open(SIMPLE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {SIMPLE_MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print("\nTo make predictions, run: python predict_spreads.py --simple")
    
    return model, scaler


def train_model(
    season: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    hidden_units: list = [16, 8],
    dropout_rate: float = 0.3,
    min_week: int = 1,
):
    """
    Train the NFL spread prediction model.
    
    Args:
        season: Season to train on (None = all seasons)
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        hidden_units: Hidden layer sizes
        dropout_rate: Dropout rate
        min_week: Minimum week to include (to ensure teams have history)
    """
    print("NFL Spread Prediction Model Training")
    print("=" * 50)
    
    # Load and prepare data
    print("\nLoading training data...")
    X, y = prepare_training_data(season=season, min_week=min_week)
    
    print(f"Total samples: {len(X)}")
    
    # Warn if insufficient data
    if len(X) < 50:
        print(f"\n⚠️  WARNING: Only {len(X)} samples available.")
        print("   Neural networks typically need 100+ samples to learn effectively.")
        print("   Consider:")
        print("   - Loading more historical data from previous seasons")
        print("   - Using --min-week 1 to include early-season games")
        print("   - Using a simpler model: --hidden-units 8 4")
        print()
    print(f"Features per sample: {X.shape[1]}")
    print(f"Home team covers: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"Visitor team covers: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
    
    # Split data chronologically (by index, assuming data is sorted by date)
    # First split: train+val vs test
    split_idx = int(len(X) * (1 - test_split))
    X_train_val = X[:split_idx]
    y_train_val = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Second split: train vs val
    val_size = int(len(X_train_val) * validation_split)
    X_train = X_train_val[:-val_size]
    y_train = y_train_val[:-val_size]
    X_val = X_train_val[-val_size:]
    y_val = y_train_val[-val_size:]
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")
    
    # Create model
    print(f"\nCreating model with {len(hidden_units)} hidden layers...")
    input_dim = X_train_scaled.shape[1]
    model = create_model(input_dim, hidden_units=hidden_units, dropout_rate=dropout_rate)
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining model for up to {epochs} epochs...")
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint callback
    checkpoint_path = MODEL_DIR / "checkpoint.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Check training history
    print(f"\nTraining History:")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Load best model from checkpoint
    model = keras.models.load_model(checkpoint_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test_scaled, y_test, verbose=0
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    
    # Calculate F1 score (handle division by zero)
    if test_precision + test_recall > 0:
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    else:
        test_f1 = 0.0
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Additional diagnostics
    y_pred = model.predict(X_test_scaled, verbose=0)
    y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
    
    print(f"\nTest Set Diagnostics:")
    print(f"  Actual home covers: {np.sum(y_test)} / {len(y_test)} ({100*np.mean(y_test):.1f}%)")
    print(f"  Predicted home covers: {np.sum(y_pred_binary)} / {len(y_pred_binary)} ({100*np.mean(y_pred_binary):.1f}%)")
    print(f"  Average prediction probability: {np.mean(y_pred):.4f}")
    print(f"  Prediction probability range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    # Save final model
    save_model(model, MODEL_PATH)
    
    print(f"\nTraining complete!")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    
    return model, scaler, history


def main():
    parser = argparse.ArgumentParser(
        description="Train NFL spread prediction model"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to train on (default: all seasons)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--min-week",
        type=int,
        default=1,
        help="Minimum week to include (default: 1)"
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        nargs="+",
        default=[16, 8],
        help="Hidden layer sizes (default: 16 8 for small datasets)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use logistic regression instead of neural network (better for small datasets)"
    )
    
    args = parser.parse_args()
    
    if args.simple:
        train_simple_model(season=args.season, min_week=args.min_week)
        return
    
    train_model(
        season=args.season,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout,
        min_week=args.min_week,
    )


if __name__ == "__main__":
    main()

