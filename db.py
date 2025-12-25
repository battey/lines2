"""
Database helper functions for NFL results.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "nfl_results.db"


def get_connection():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def get_zulu_timestamp() -> str:
    """Get current UTC timestamp in Zulu format (ISO 8601 with 'Z' suffix)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def update_game_scores(home: str, visitor: str, date: str, 
                       home_score: int, visitor_score: int) -> bool:
    """
    Update a game with final scores.
    Only updates if scores are currently missing (None).
    Returns True if updated, False if game not found or scores already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Only update if scores are missing
        cursor.execute("""
            UPDATE result 
            SET home_score = ?, visitor_score = ?, updated_at = ?
            WHERE home = ? AND visitor = ? AND date = ?
            AND (home_score IS NULL OR visitor_score IS NULL)
        """, (home_score, visitor_score, get_zulu_timestamp(), home, visitor, date))
        
        conn.commit()
        updated = cursor.rowcount > 0
        return updated
    except sqlite3.Error as e:
        print(f"Error updating game scores: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_games_by_week(season: int, week: int) -> List[Dict]:
    """Get all games for a specific season and week."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM result 
        WHERE season = ? AND week = ?
        ORDER BY date
    """, (season, week))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def game_exists(home: str, visitor: str, date: str) -> bool:
    """Check if a game already exists in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM result 
        WHERE home = ? AND visitor = ? AND date = ?
    """, (home, visitor, date))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    return count > 0


def get_most_recent_expected_qb(team_name: str, season: int, before_week: int) -> Optional[str]:
    """
    Get the most recent expected quarterback for a team from previous games in the current season.
    
    Args:
        team_name: Name of the team
        season: NFL season year
        before_week: Only consider games from weeks before this
    
    Returns:
        Quarterback name, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find the most recent game where this team played and has an expected quarterback
        cursor.execute("""
            SELECT home_expected_qb, visitor_expected_qb, home, visitor
            FROM result 
            WHERE season = ? 
              AND week < ?
              AND (home_expected_qb IS NOT NULL OR visitor_expected_qb IS NOT NULL)
              AND (UPPER(home) = UPPER(?) OR UPPER(visitor) = UPPER(?))
            ORDER BY week DESC, date DESC
            LIMIT 1
        """, (season, before_week, team_name, team_name))
        
        result = cursor.fetchone()
        if not result:
            return None
        
        home_expected_qb = result[0]
        visitor_expected_qb = result[1]
        home_team = result[2]
        visitor_team = result[3]
        
        # Determine if the team we're looking for was home or away
        if home_team.upper() == team_name.upper():
            return home_expected_qb
        else:
            return visitor_expected_qb
    except sqlite3.Error as e:
        print(f"Error getting most recent expected quarterback: {e}")
        return None
    finally:
        conn.close()


def get_most_recent_actual_qb(team_name: str, season: int, before_week: int) -> Optional[str]:
    """
    Get the most recent actual quarterback for a team from previous games in the current season.
    
    Args:
        team_name: Name of the team
        season: NFL season year
        before_week: Only consider games from weeks before this
    
    Returns:
        Quarterback name, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find the most recent completed game where this team played and has an actual quarterback
        cursor.execute("""
            SELECT home_actual_qb, visitor_actual_qb, home, visitor
            FROM result 
            WHERE season = ? 
              AND week < ?
              AND home_score IS NOT NULL
              AND visitor_score IS NOT NULL
              AND (home_actual_qb IS NOT NULL OR visitor_actual_qb IS NOT NULL)
              AND (UPPER(home) = UPPER(?) OR UPPER(visitor) = UPPER(?))
            ORDER BY week DESC, date DESC
            LIMIT 1
        """, (season, before_week, team_name, team_name))
        
        result = cursor.fetchone()
        if not result:
            return None
        
        home_actual_qb = result[0]
        visitor_actual_qb = result[1]
        home_team = result[2]
        visitor_team = result[3]
        
        # Determine if the team we're looking for was home or away
        if home_team.upper() == team_name.upper():
            return home_actual_qb
        else:
            return visitor_actual_qb
    except sqlite3.Error as e:
        print(f"Error getting most recent actual quarterback: {e}")
        return None
    finally:
        conn.close()


def get_expected_qb_from_game(home: str, visitor: str, date: str, team_name: str) -> Optional[str]:
    """
    Get the expected quarterback for a specific team from a specific game record.
    
    Args:
        home: Home team name
        visitor: Visitor team name
        date: Game date
        team_name: Name of the team to get QB for
    
    Returns:
        Expected quarterback name for the specified team, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT home_expected_qb, visitor_expected_qb, home, visitor
            FROM result 
            WHERE home = ? AND visitor = ? AND date = ?
        """, (home, visitor, date))
        
        result = cursor.fetchone()
        if not result:
            return None
        
        home_expected_qb = result[0]
        visitor_expected_qb = result[1]
        home_team = result[2]
        visitor_team = result[3]
        
        # Determine if the team we're looking for was home or away
        if home_team.upper() == team_name.upper():
            return home_expected_qb
        elif visitor_team.upper() == team_name.upper():
            return visitor_expected_qb
        else:
            return None
    except sqlite3.Error as e:
        print(f"Error getting expected quarterback from game: {e}")
        return None
    finally:
        conn.close()


def upsert_game(home: str, visitor: str, date: str, spread: Optional[float],
                home_score: Optional[int], visitor_score: Optional[int],
                season: int, week: int, home_expected_qb: Optional[str] = None,
                visitor_expected_qb: Optional[str] = None, home_actual_qb: Optional[str] = None,
                visitor_actual_qb: Optional[str] = None, log_reg_win_prob: Optional[float] = None,
                dl_win_prob: Optional[float] = None) -> bool:
    """
    Insert or update a game. If game exists, only update missing values.
    Returns True if successful.
    
    Note: spread can be None if unavailable (e.g., uncertain starting QB).
    Note: expected_qb columns filled when gathering schedule, actual_qb columns filled when gathering results.
    Note: log_reg_win_prob and dl_win_prob are predictions (NULL for now).
    
    Rules:
    - If game doesn't exist, insert it
    - If game exists:
      - Only update scores if existing scores are None and new scores are provided
      - Only update spread if existing spread is None and new spread is provided
      - Only update QB columns if existing values are None and new values are provided
      - Only update prediction columns if existing values are None and new values are provided
      - Never overwrite existing values
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if game exists - match by home, visitor, season, week (not date, since date format may differ)
        # Also get existing values to check what's missing
        cursor.execute("""
            SELECT id, date, home_score, visitor_score, spread, home_expected_qb, visitor_expected_qb,
                   home_actual_qb, visitor_actual_qb, log_reg_win_prob, dl_win_prob FROM result 
            WHERE home = ? AND visitor = ? AND season = ? AND week = ?
        """, (home, visitor, season, week))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing game
            game_id, existing_date, existing_home_score, existing_visitor_score, existing_spread, existing_home_expected_qb, existing_visitor_expected_qb, existing_home_actual_qb, existing_visitor_actual_qb, existing_log_reg_win_prob, existing_dl_win_prob = existing
            
            zulu_now = get_zulu_timestamp()
            
            # Determine what needs to be updated
            # Update date if it's different (e.g., converting from Zulu to Eastern format)
            update_date = (date != existing_date)
            update_scores = (home_score is not None and visitor_score is not None and 
                           (existing_home_score is None or existing_visitor_score is None))
            update_spread = (spread is not None and existing_spread is None)
            update_home_expected_qb = (home_expected_qb is not None and existing_home_expected_qb is None)
            update_visitor_expected_qb = (visitor_expected_qb is not None and existing_visitor_expected_qb is None)
            update_home_actual_qb = (home_actual_qb is not None and existing_home_actual_qb is None)
            update_visitor_actual_qb = (visitor_actual_qb is not None and existing_visitor_actual_qb is None)
            update_log_reg_win_prob = (log_reg_win_prob is not None and existing_log_reg_win_prob is None)
            update_dl_win_prob = (dl_win_prob is not None and existing_dl_win_prob is None)
            
            # Build dynamic update query
            updates = []
            params = []
            
            if update_date:
                updates.append("date = ?")
                params.append(date)
            if update_scores:
                updates.extend(["home_score = ?", "visitor_score = ?"])
                params.extend([home_score, visitor_score])
            if update_spread:
                updates.append("spread = ?")
                params.append(spread)
            if update_home_expected_qb:
                updates.append("home_expected_qb = ?")
                params.append(home_expected_qb)
            if update_visitor_expected_qb:
                updates.append("visitor_expected_qb = ?")
                params.append(visitor_expected_qb)
            if update_home_actual_qb:
                updates.append("home_actual_qb = ?")
                params.append(home_actual_qb)
            if update_visitor_actual_qb:
                updates.append("visitor_actual_qb = ?")
                params.append(visitor_actual_qb)
            if update_log_reg_win_prob:
                updates.append("log_reg_win_prob = ?")
                params.append(log_reg_win_prob)
            if update_dl_win_prob:
                updates.append("dl_win_prob = ?")
                params.append(dl_win_prob)
            
            # Only update if there's something to update
            if updates:
                updates.append("updated_at = ?")
                params.append(zulu_now)
                params.append(game_id)
                
                query = f"UPDATE result SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
            # If nothing needs updating, do nothing (don't update updated_at either)
        else:
            # Insert new game (set both created_at and updated_at to current Zulu time)
            # Date should already be in Eastern format from normalize_date_to_eastern()
            zulu_now = get_zulu_timestamp()
            cursor.execute("""
                INSERT INTO result (home, visitor, date, spread, home_score, visitor_score, home_expected_qb, visitor_expected_qb, home_actual_qb, visitor_actual_qb, log_reg_win_prob, dl_win_prob, season, week, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (home, visitor, date, spread, home_score, visitor_score, home_expected_qb, visitor_expected_qb, home_actual_qb, visitor_actual_qb, log_reg_win_prob, dl_win_prob, season, week, zulu_now, zulu_now))
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error upserting game: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
