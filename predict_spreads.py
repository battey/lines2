#!/usr/bin/env -S uv run --
"""
Prediction script for NFL spread prediction model.
Generates predictions for upcoming games.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from tabulate import tabulate
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from model_data import get_upcoming_games, get_completed_games, extract_features_for_game
from nfl_spread_model import load_model
from db import upsert_game

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "nfl_spread_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
SIMPLE_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"


def load_model_and_scaler(simple: bool = False):
    """
    Load trained model and scaler.
    
    Args:
        simple: If True, load logistic regression model instead of neural network
    
    Returns:
        Tuple of (model, scaler, is_simple)
    """
    model_path = SIMPLE_MODEL_PATH if simple else MODEL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first using: train_model.py{' --simple' if simple else ''}"
        )
    
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Please train the model first using train_model.py"
        )
    
    print(f"Loading model from {model_path}...")
    if simple:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = load_model(str(model_path))
    
    print(f"Loading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler, simple


def predict_upcoming_games(
    season: Optional[int] = None,
    week: Optional[int] = None,
    model: Optional[object] = None,
    scaler: Optional[object] = None,
    simple: bool = False,
) -> list:
    """
    Generate predictions for upcoming games.
    
    Args:
        season: Season to predict (None = current season)
        week: Week to predict (None = highest week with incomplete games)
        model: Pre-loaded model (None = load from disk)
        scaler: Pre-loaded scaler (None = load from disk)
        simple: Use logistic regression model instead of neural network
    
    Returns:
        List of prediction dictionaries
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler, simple = load_model_and_scaler(simple=simple)
    
    # Get upcoming games
    upcoming_games = get_upcoming_games(season=season, week=week)
    
    if not upcoming_games:
        print("No upcoming games found.")
        return []
    
    # Get completed games for feature extraction
    completed_games = get_completed_games(season=season)
    
    print(f"\nFound {len(upcoming_games)} upcoming game(s)")
    print("Extracting features and generating predictions...")
    print(f"Saving predictions to database ({'logistic regression' if simple else 'deep learning'})...\n")
    
    predictions = []
    
    for game in upcoming_games:
        try:
            # Extract features
            features = extract_features_for_game(game, completed_games)
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Generate prediction (handle both model types)
            if simple:
                # Logistic regression model
                probability = model.predict_proba(features_scaled)[0][1]
            else:
                # Neural network model
                probability = model.predict(features_scaled, verbose=0)[0][0]
            
            # Determine prediction
            home_covers = probability >= 0.5
            confidence = probability if home_covers else (1 - probability)
            
            spread = game.get('spread', 0.0)
            if spread is None:
                spread = 0.0
            
            predictions.append({
                'home': game['home'],
                'visitor': game['visitor'],
                'date': game['date'],
                'spread': spread,
                'probability': probability,
                'prediction': 'Home' if home_covers else 'Visitor',
                'confidence': confidence,
                'week': game['week'],
                'season': game['season'],
            })
            
            # Save prediction to database
            # Probability represents home team's win probability (probability of home team covering spread)
            log_reg_win_prob = probability if simple else None
            dl_win_prob = probability if not simple else None
            
            upsert_game(
                home=game['home'],
                visitor=game['visitor'],
                date=game['date'],
                spread=game.get('spread'),
                home_score=game.get('home_score'),
                visitor_score=game.get('visitor_score'),
                season=game['season'],
                week=game['week'],
                home_expected_qb=game.get('home_expected_qb'),
                visitor_expected_qb=game.get('visitor_expected_qb'),
                home_actual_qb=game.get('home_actual_qb'),
                visitor_actual_qb=game.get('visitor_actual_qb'),
                log_reg_win_prob=log_reg_win_prob,
                dl_win_prob=dl_win_prob,
            )
            
        except Exception as e:
            print(f"Warning: Could not predict {game['home']} vs {game['visitor']}: {e}")
            continue
    
    return predictions


def display_predictions(predictions: list):
    """
    Display predictions in a formatted table.
    
    Args:
        predictions: List of prediction dictionaries
    """
    if not predictions:
        print("No predictions to display.")
        return
    
    # Sort by confidence (descending)
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Prepare table data
    table_data = []
    for pred in predictions:
        spread_display = f"{pred['spread']:+.1f}" if pred['spread'] is not None else "N/A"
        prob_display = f"{pred['probability']:.1%}"
        conf_display = f"{pred['confidence']:.1%}"
        
        table_data.append([
            pred['visitor'],
            "@",
            pred['home'],
            spread_display,
            pred['prediction'],
            conf_display,
            prob_display,
        ])
    
    headers = ["Visitor", "", "Home", "Spread", "Pick", "Confidence", "Prob"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Summary
    season = predictions[0]['season']
    week = predictions[0]['week']
    print(f"\nWeek {week}, Season {season}")
    print(f"Total games: {len(predictions)}")
    home_picks = sum(1 for p in predictions if p['prediction'] == 'Home')
    visitor_picks = len(predictions) - home_picks
    print(f"Home picks: {home_picks}, Visitor picks: {visitor_picks}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict NFL spread outcomes for upcoming games"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to predict (default: current season)"
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week to predict (default: highest week with incomplete games)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use logistic regression model instead of neural network"
    )
    
    args = parser.parse_args()
    
    print("NFL Spread Prediction")
    print("=" * 50)
    
    try:
        predictions = predict_upcoming_games(season=args.season, week=args.week, simple=args.simple)
        display_predictions(predictions)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo train the model, run:")
        print("  python train_model.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

