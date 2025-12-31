# Machine Learning Models for NFL Spread Prediction

This document explains the machine learning models used to predict whether the home team or visitor team will cover the spread in NFL games.

---

## Table of Contents

1. [Overview](#overview)
2. [Input Features](#input-features)
3. [Output (Target Variable)](#output-target-variable)
4. [Logistic Regression Model](#logistic-regression-model)
5. [Deep Learning Neural Network](#deep-learning-neural-network)
6. [Feature Scaling](#feature-scaling)
7. [Training Process](#training-process)
8. [Making Predictions](#making-predictions)
9. [When to Use Each Model](#when-to-use-each-model)

---

## Overview

The system predicts **whether the home team will "cover the spread"** in an NFL game. This is a **binary classification problem**:
- **1 (True)**: Home team covers the spread
- **0 (False)**: Visitor team covers the spread

### What Does "Cover the Spread" Mean?

The spread is a point handicap set by oddsmakers. For example:
- **Spread of -7**: The home team is favored by 7 points. They "cover" if they win by more than 7.
- **Spread of +3**: The home team is the underdog by 3 points. They "cover" if they lose by less than 3 OR win outright.

Mathematically, the home team covers if:
```
(home_score - visitor_score) >= -spread
```

---

## Input Features

The models use **55 input features** extracted from historical game data. These features fall into seven categories:

### 1. Current Game Spread (1 feature)

| Feature | Description |
|---------|-------------|
| `spread` | The point spread for the current game (negative = home favored) |

### 2. Home Team Season Statistics (6 features)

Statistics calculated from all of the home team's games **prior to the current week**:

| Feature | Description |
|---------|-------------|
| `win_pct` | Win percentage (0.0 to 1.0) |
| `points_for_avg` | Average points scored per game |
| `points_against_avg` | Average points allowed per game |
| `point_diff_avg` | Average point differential (positive = outscoring opponents) |
| `cover_pct` | Percentage of games where team covered the spread |
| `avg_margin` | Average scoring margin (points scored minus points allowed) |

### 3. Visitor Team Season Statistics (6 features)

The same 6 statistics as above, but calculated for the visiting team.

### 4. Home Team Recent Performance (15 features)

Performance from the **last 5 games** (3 features per game):

For each of the 5 most recent games:
| Feature | Description |
|---------|-------------|
| `margin` | Point differential in that game (positive = won) |
| `covered` | Whether the team covered the spread (1 or 0) |
| `spread` | The spread for that game from the team's perspective |

This captures momentum and recent form. If a team has played fewer than 5 games, missing games are filled with zeros.

### 5. Visitor Team Recent Performance (15 features)

The same 15 features (5 games × 3 features) for the visiting team.

### 6. Home Quarterback Statistics (6 features)

Statistics for the home team's expected/actual quarterback:

| Feature | Description |
|---------|-------------|
| `qb_games` | Number of games this QB has started this season |
| `qb_win_pct` | Win percentage when this QB plays |
| `qb_cover_pct` | Spread cover percentage when this QB plays |
| `qb_points_avg` | Average points scored when this QB plays |
| `qb_margin_avg` | Average scoring margin when this QB plays |
| `is_backup` | 1 if this QB is NOT the team's primary starter, 0 otherwise |

The `is_backup` feature is particularly important — it flags situations like "the Colts with their backup QB" which historically perform differently than with the starter.

### 7. Visitor Quarterback Statistics (6 features)

The same 6 features for the visiting team's quarterback.

### How QB Identity is Determined

- **For predictions**: Uses `expected_qb` (who is expected to start)
- **For training**: Uses `actual_qb` if available (who actually played), otherwise `expected_qb`
- **Primary QB**: The quarterback who has started the most games for that team this season
- **Backup detection**: If the current game's QB differs from the primary QB, `is_backup = 1`

### Feature Count Summary

```
1  (current spread)
+ 6  (home team stats)
+ 6  (visitor team stats)
+ 15 (home team recent: 5 games × 3 features)
+ 15 (visitor team recent: 5 games × 3 features)
+ 6  (home QB stats)
+ 6  (visitor QB stats)
────
= 55 total features
```

---

## Output (Target Variable)

The model outputs a **probability** between 0 and 1:
- Values **≥ 0.5**: Predict home team covers
- Values **< 0.5**: Predict visitor team covers

The confidence is calculated as:
- If predicting home: `confidence = probability`
- If predicting visitor: `confidence = 1 - probability`

---

## Logistic Regression Model

### What is Logistic Regression?

Logistic regression is a **linear model** for binary classification. Despite its name, it's used for classification (not regression). It works by:

1. Computing a weighted sum of all input features
2. Passing that sum through a **sigmoid function** to get a probability between 0 and 1

```
probability = sigmoid(w₁×feature₁ + w₂×feature₂ + ... + w₄₃×feature₄₃ + bias)
```

Where the sigmoid function is: `sigmoid(x) = 1 / (1 + e^(-x))`

### Implementation Details

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
```

**Key Parameters:**
- `max_iter=1000`: Maximum iterations for the optimization algorithm to converge
- `random_state=42`: Ensures reproducible results

### Advantages

- ✅ Works well with **small datasets** (under 100 samples)
- ✅ **Fast** to train and predict
- ✅ **Interpretable**: You can examine the learned weights to see which features are most important
- ✅ Less prone to **overfitting** on limited data

### Limitations

- ❌ Can only learn **linear relationships** between features and the outcome
- ❌ Cannot capture complex feature interactions without manual feature engineering

---

## Deep Learning Neural Network

### What is a Neural Network?

A neural network is a model inspired by the human brain. It consists of layers of interconnected "neurons" that can learn complex, non-linear patterns in data.

### Architecture

The network is a **feedforward neural network** (also called a Multi-Layer Perceptron):

```
Input Layer (55 features)
     ↓
Hidden Layer 1 (16 neurons) + ReLU + Dropout
     ↓
Hidden Layer 2 (8 neurons) + ReLU + Dropout
     ↓
Output Layer (1 neuron) + Sigmoid → Probability
```

### Layer-by-Layer Explanation

#### Input Layer
- Receives the 43 normalized features

#### Hidden Layers
Each hidden layer:
1. **Dense (Fully Connected)**: Every neuron connects to every neuron in the previous layer
2. **ReLU Activation**: `ReLU(x) = max(0, x)` — introduces non-linearity, allowing the network to learn complex patterns
3. **Dropout (30%)**: Randomly "turns off" 30% of neurons during training to prevent overfitting

#### Output Layer
- **1 neuron** with **sigmoid activation**: Outputs a probability between 0 and 1

### Implementation Details

```python
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(55,)),
    layers.Dropout(0.3),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| **Adam optimizer** | Adaptive learning rate optimization algorithm |
| **Learning rate 0.001** | Step size for weight updates (smaller = more careful learning) |
| **Binary crossentropy loss** | Standard loss function for binary classification |
| **Dropout 0.3** | Regularization to prevent overfitting |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Maximum number of passes through the training data |
| `batch_size` | 32 | Number of samples processed before updating weights |
| `validation_split` | 0.15 | 15% of training data used for validation |
| `test_split` | 0.15 | 15% of data held out for final testing |

### Early Stopping

Training automatically stops if validation loss doesn't improve for 10 consecutive epochs. This prevents overfitting and saves time.

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

### Advantages

- ✅ Can learn **complex, non-linear relationships**
- ✅ Automatically discovers **feature interactions**
- ✅ Can improve with **more data**

### Limitations

- ❌ Needs **more data** to train effectively (100+ samples recommended)
- ❌ Can **overfit** on small datasets
- ❌ Less interpretable (harder to understand why it makes specific predictions)
- ❌ Slower to train than logistic regression

---

## Feature Scaling

Both models use **StandardScaler** to normalize features before training:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std from training data
X_test_scaled = scaler.transform(X_test)        # Apply same transformation to test data
```

### Why Scale Features?

Different features have very different scales:
- `win_pct`: 0.0 to 1.0
- `points_for_avg`: ~10 to ~35
- `spread`: -15 to +15

Scaling transforms each feature to have:
- **Mean = 0**
- **Standard deviation = 1**

This ensures no single feature dominates the model simply because of its scale.

### Important: The Scaler Must Be Saved

The scaler learned from training data is saved to `models/scaler.pkl` and **must be used** when making predictions. Using a different scaler (or no scaler) would produce incorrect results.

---

## Training Process

### Data Preparation

1. **Load completed games** from the database (games with final scores)
2. **Filter by season**: By default, only games from the **current season** are used
3. **Filter by minimum week** (default: week 4) to ensure teams have some history
4. **Extract features** for each game using prior game data only (no data leakage)
5. **Create labels**: 1 if home covered, 0 if visitor covered

### Single-Season Training (Default Behavior)

**Important**: The model trains on only the current season's data by default. This means:

- ✅ No mixing of data from different seasons (e.g., 2025 and 2026)
- ✅ Team statistics and recent performance are consistent within the same season
- ⚠️ Early in the season, you'll have limited training data
- ⚠️ Meaningful predictions typically require at least 4-5 weeks of completed games

**Why this matters**: Team rosters, coaching strategies, and performance patterns change between seasons. Using only current-season data ensures the model learns from the most relevant information.

### Data Splitting

Data is split **chronologically** (not randomly) to simulate real-world prediction:

```
[Older Games] ─────────────────────────────────────> [Newer Games]
     │                    │                    │
     └── Training ────────┴── Validation ──────┴── Test
         (~70%)              (~15%)              (~15%)
```

This is important because:
- We can't use future data to predict past games
- The model is tested on games it hasn't seen, simulating real prediction

### Training Commands

**Logistic Regression (current season only):**
```bash
python train_model.py --simple
```

**Neural Network (current season only):**
```bash
python train_model.py --epochs 100 --hidden-units 16 8 --dropout 0.3
```

**Train on a specific season:**
```bash
python train_model.py --simple --season 2025
```

**Train on ALL historical seasons (optional):**
```bash
python train_model.py --simple --all-seasons
```

> **Note**: Using `--all-seasons` mixes data from different seasons. This provides more training data but may include patterns that are no longer relevant to the current season.

---

## Making Predictions

### Prediction Flow

```
1. Load trained model and scaler
2. Get upcoming games (games without scores)
3. Determine the season from upcoming games
4. For each game:
   a. Gather completed games from THIS SEASON ONLY
   b. Calculate team statistics from those games
   c. Extract all 43 features
   d. Scale features using the saved scaler
   e. Pass through model to get probability
   f. Predict home covers if probability ≥ 0.5
```

> **Important**: Feature extraction uses only the current season's data, consistent with how the model was trained. This ensures that team statistics reflect the current season's performance, not historical data from previous years.

### Prediction Commands

**Using Logistic Regression:**
```bash
python predict_spreads.py --simple
```

**Using Neural Network:**
```bash
python predict_spreads.py
```

### Output

Predictions are displayed in a table sorted by confidence:

```
+-----------+---+--------+--------+--------+------------+-------+
| Visitor   |   | Home   | Spread | Pick   | Confidence | Prob  |
+-----------+---+--------+--------+--------+------------+-------+
| Cowboys   | @ | Eagles | -6.5   | Home   | 72.3%      | 72.3% |
| Packers   | @ | Bears  | +3.0   | Visitor| 61.5%      | 38.5% |
+-----------+---+--------+--------+--------+------------+-------+
```

---

## When to Use Each Model

### Use Logistic Regression (`--simple`) When:

- You have **fewer than 100 training samples**
- You're **early in the season** (weeks 4-10)
- You want **fast training and predictions**
- You need **interpretable results**
- You're working with **limited computational resources**

### Use the Neural Network When:

- You have **100+ training samples** (ideally 200+)
- You're in the **second half of the season** (week 10+)
- You believe there are **complex patterns** in the data
- You have time for **longer training**
- You can tolerate a **"black box"** model

### Practical Recommendation for Single-Season Training

Since the model now trains on only the current season by default:

| Week | Available Games | Training Samples (~70%) | Recommended Model |
|------|-----------------|------------------------|-------------------|
| 1-3  | 0-48           | 0-34                   | ❌ Not enough data |
| 4-6  | ~48-96         | ~34-67                 | Logistic Regression |
| 7-10 | ~96-160        | ~67-112                | Logistic Regression |
| **11+** | **160+**    | **112+**               | **Neural Network viable** |

If you need more training data early in the season, you can use `--all-seasons` to include historical data, understanding that patterns may have changed.

---

## File Reference

| File | Purpose |
|------|---------|
| `model_data.py` | Feature extraction and data preparation |
| `nfl_spread_model.py` | Neural network architecture definition |
| `train_model.py` | Training script for both models |
| `predict_spreads.py` | Prediction script for upcoming games |
| `models/scaler.pkl` | Saved feature scaler |
| `models/logistic_model.pkl` | Saved logistic regression model |
| `models/nfl_spread_model.h5` | Saved neural network model |

