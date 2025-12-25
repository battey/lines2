"""
Feature extraction and data preparation for NFL spread prediction model.
"""

import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from db import get_connection, DB_PATH

# Number of recent weeks to use for features
RECENT_WEEKS = 5


def get_completed_games(season: Optional[int] = None, before_week: Optional[int] = None) -> List[Dict]:
    """
    Get all completed games from the database.
    
    Args:
        season: Filter by season (None = all seasons)
        before_week: Only include games from weeks before this (None = all weeks)
    
    Returns:
        List of game dictionaries with all columns
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT * FROM result 
        WHERE home_score IS NOT NULL 
          AND visitor_score IS NOT NULL
          AND spread IS NOT NULL
    """
    params = []
    
    if season is not None:
        query += " AND season = ?"
        params.append(season)
    
    if before_week is not None:
        query += " AND week < ?"
        params.append(before_week)
    
    query += " ORDER BY season, week, date"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_upcoming_games(season: Optional[int] = None, week: Optional[int] = None) -> List[Dict]:
    """
    Get upcoming games (games without scores) from the database.
    
    Args:
        season: Filter by season (None = current season)
        week: Filter by week (None = highest week with incomplete games)
    
    Returns:
        List of game dictionaries
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # If no season specified, use current season
    if season is None:
        from datetime import datetime
        now = datetime.now()
        season = now.year if now.month >= 9 else now.year - 1
    
    # If no week specified, find the highest week with incomplete games
    if week is None:
        cursor.execute("""
            SELECT MAX(week) as max_week
            FROM result
            WHERE season = ? 
              AND (home_score IS NULL OR visitor_score IS NULL)
        """, (season,))
        result = cursor.fetchone()
        if not result or result[0] is None:
            conn.close()
            return []
        week = result[0]
    
    cursor.execute("""
        SELECT * FROM result
        WHERE season = ? AND week = ?
          AND (home_score IS NULL OR visitor_score IS NULL)
        ORDER BY date
    """, (season, week))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def calculate_team_stats(team_name: str, season: int, before_week: int, completed_games: List[Dict]) -> Dict:
    """
    Calculate team statistics from completed games.
    
    Args:
        team_name: Name of the team
        season: Season year
        before_week: Only consider games before this week
        completed_games: List of all completed games (for efficiency)
    
    Returns:
        Dictionary with team statistics
    """
    # Filter games for this team in this season before the specified week
    team_games = []
    for game in completed_games:
        if (game['season'] == season and 
            game['week'] < before_week and
            (game['home'].upper() == team_name.upper() or 
             game['visitor'].upper() == team_name.upper())):
            team_games.append(game)
    
    if not team_games:
        # Return default values if no games found
        return {
            'wins': 0,
            'losses': 0,
            'win_pct': 0.5,
            'points_for_avg': 20.0,
            'points_against_avg': 20.0,
            'point_diff_avg': 0.0,
            'covers': 0,
            'no_covers': 0,
            'cover_pct': 0.5,
            'avg_margin': 0.0,
        }
    
    wins = 0
    losses = 0
    points_for = []
    points_against = []
    covers = 0
    no_covers = 0
    margins = []
    
    for game in team_games:
        is_home = game['home'].upper() == team_name.upper()
        
        if is_home:
            team_score = game['home_score']
            opp_score = game['visitor_score']
            spread = game['spread']
        else:
            team_score = game['visitor_score']
            opp_score = game['home_score']
            spread = -game['spread']  # Flip spread for visitor perspective
        
        # Win/loss
        if team_score > opp_score:
            wins += 1
        else:
            losses += 1
        
        # Points
        points_for.append(team_score)
        points_against.append(opp_score)
        margin = team_score - opp_score
        margins.append(margin)
        
        # Cover calculation
        # Team covers if: (team_score - opp_score) >= -spread
        # For home: (home_score - visitor_score) >= -spread
        # For visitor: (visitor_score - home_score) >= -(-spread) = spread
        if margin >= -spread:
            covers += 1
        else:
            no_covers += 1
    
    total_games = len(team_games)
    
    return {
        'wins': wins,
        'losses': losses,
        'win_pct': wins / total_games if total_games > 0 else 0.5,
        'points_for_avg': np.mean(points_for) if points_for else 20.0,
        'points_against_avg': np.mean(points_against) if points_against else 20.0,
        'point_diff_avg': np.mean(points_for) - np.mean(points_against) if points_for and points_against else 0.0,
        'covers': covers,
        'no_covers': no_covers,
        'cover_pct': covers / total_games if total_games > 0 else 0.5,
        'avg_margin': np.mean(margins) if margins else 0.0,
    }


def get_recent_results(team_name: str, season: int, before_week: int, completed_games: List[Dict], n_weeks: int = RECENT_WEEKS) -> List[Dict]:
    """
    Get recent game results for a team.
    
    Args:
        team_name: Name of the team
        season: Season year
        before_week: Only consider games before this week
        completed_games: List of all completed games
        n_weeks: Number of recent weeks to include
    
    Returns:
        List of recent game dictionaries with cover outcome
    """
    # Filter games for this team
    team_games = []
    for game in completed_games:
        if (game['season'] == season and 
            game['week'] < before_week and
            (game['home'].upper() == team_name.upper() or 
             game['visitor'].upper() == team_name.upper())):
            team_games.append(game)
    
    # Sort by week descending and take most recent n_weeks worth
    team_games.sort(key=lambda x: (x['week'], x['date']), reverse=True)
    
    recent_results = []
    seen_weeks = set()
    
    for game in team_games:
        if len(seen_weeks) >= n_weeks:
            break
        
        week = game['week']
        if week in seen_weeks:
            continue
        
        seen_weeks.add(week)
        
        is_home = game['home'].upper() == team_name.upper()
        
        if is_home:
            team_score = game['home_score']
            opp_score = game['visitor_score']
            spread = game['spread']
        else:
            team_score = game['visitor_score']
            opp_score = game['home_score']
            spread = -game['spread']
        
        margin = team_score - opp_score
        covered = 1 if margin >= -spread else 0
        
        recent_results.append({
            'week': week,
            'team_score': team_score,
            'opp_score': opp_score,
            'spread': spread,
            'margin': margin,
            'covered': covered,
        })
    
    # Pad with zeros if not enough games
    while len(recent_results) < n_weeks:
        recent_results.append({
            'week': 0,
            'team_score': 0,
            'opp_score': 0,
            'spread': 0.0,
            'margin': 0.0,
            'covered': 0,
        })
    
    return recent_results[:n_weeks]


def extract_features_for_game(game: Dict, completed_games: List[Dict]) -> np.ndarray:
    """
    Extract feature vector for a single game.
    
    Features:
    1. Current game spread
    2. Home team stats (win_pct, points_for_avg, points_against_avg, point_diff_avg, cover_pct, avg_margin)
    3. Visitor team stats (same as above)
    4. Home team recent results (last N weeks: margin, covered, spread)
    5. Visitor team recent results (last N weeks: margin, covered, spread)
    
    Args:
        game: Game dictionary from database
        completed_games: List of all completed games for calculating stats
    
    Returns:
        Feature vector as numpy array
    """
    season = game['season']
    week = game['week']
    home_team = game['home']
    visitor_team = game['visitor']
    spread = game.get('spread', 0.0)
    
    # Handle missing spread
    if spread is None:
        spread = 0.0
    
    # Get team statistics
    home_stats = calculate_team_stats(home_team, season, week, completed_games)
    visitor_stats = calculate_team_stats(visitor_team, season, week, completed_games)
    
    # Get recent results
    home_recent = get_recent_results(home_team, season, week, completed_games)
    visitor_recent = get_recent_results(visitor_team, season, week, completed_games)
    
    # Build feature vector
    features = []
    
    # 1. Current spread
    features.append(spread)
    
    # 2. Home team stats (6 features)
    features.extend([
        home_stats['win_pct'],
        home_stats['points_for_avg'],
        home_stats['points_against_avg'],
        home_stats['point_diff_avg'],
        home_stats['cover_pct'],
        home_stats['avg_margin'],
    ])
    
    # 3. Visitor team stats (6 features)
    features.extend([
        visitor_stats['win_pct'],
        visitor_stats['points_for_avg'],
        visitor_stats['points_against_avg'],
        visitor_stats['point_diff_avg'],
        visitor_stats['cover_pct'],
        visitor_stats['avg_margin'],
    ])
    
    # 4. Home team recent results (N weeks * 3 features: margin, covered, spread)
    for result in home_recent:
        features.extend([
            result['margin'],
            result['covered'],
            result['spread'],
        ])
    
    # 5. Visitor team recent results (N weeks * 3 features)
    for result in visitor_recent:
        features.extend([
            result['margin'],
            result['covered'],
            result['spread'],
        ])
    
    return np.array(features, dtype=np.float32)


def create_label(game: Dict) -> int:
    """
    Create label for a completed game.
    
    Label: 1 if home team covered the spread, 0 if visitor team covered.
    
    Args:
        game: Game dictionary with scores and spread
    
    Returns:
        1 if home covered, 0 if visitor covered
    """
    home_score = game['home_score']
    visitor_score = game['visitor_score']
    spread = game['spread']
    
    # Home team covers if: (home_score - visitor_score) >= -spread
    # This works because:
    # - If spread is -7 (home favored by 7), home covers if wins by 7+ or loses by <7
    # - If spread is +7 (home underdog by 7), home covers if loses by <7 or wins
    actual_margin = home_score - visitor_score
    return 1 if actual_margin >= -spread else 0


def prepare_training_data(season: Optional[int] = None, min_week: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from completed games.
    
    Args:
        season: Filter by season (None = all seasons)
        min_week: Minimum week to include (to ensure teams have some history)
    
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    # Get all completed games
    completed_games = get_completed_games(season=season)
    
    # Filter to games with enough history
    training_games = [g for g in completed_games if g['week'] >= min_week]
    
    if not training_games:
        raise ValueError("No training data available. Need completed games with week >= min_week.")
    
    features_list = []
    labels_list = []
    
    for game in training_games:
        try:
            features = extract_features_for_game(game, completed_games)
            label = create_label(game)
            features_list.append(features)
            labels_list.append(label)
        except Exception as e:
            print(f"Warning: Skipping game {game['home']} vs {game['visitor']} due to error: {e}")
            continue
    
    if not features_list:
        raise ValueError("No valid training examples after feature extraction.")
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    return X, y


def get_feature_count() -> int:
    """
    Calculate the number of features based on RECENT_WEEKS.
    
    Returns:
        Total number of features
    """
    # 1 (spread) + 6 (home stats) + 6 (visitor stats) + 
    # RECENT_WEEKS * 3 (home recent) + RECENT_WEEKS * 3 (visitor recent)
    return 1 + 6 + 6 + (RECENT_WEEKS * 3) + (RECENT_WEEKS * 3)

