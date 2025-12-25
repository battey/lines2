#!/usr/bin/env -S uv run --
"""
NFL Weekly Update Script
Fetches previous week's scores and upcoming week's schedule with point spreads.
Run every Tuesday during NFL season.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests
from typing import Dict, List, Optional
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import database functions
from db import upsert_game, game_exists, update_game_scores, get_most_recent_expected_qb
from db_utils import dump_sqlite_to_file

# Configuration
DB_PATH = Path(__file__).parent / "nfl_results.db"
DB_DUMP_PATH = Path(__file__).parent / "db_dump.sql"

# API Keys (set as environment variables or in .env file)
# For The Odds API: https://the-odds-api.com/
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ESPN API endpoints (public, no key required)
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def normalize_date_to_zulu(date_str: str) -> str:
    """
    Normalize a date string to Zulu (UTC) format with 'Z' suffix.
    Handles various ISO 8601 formats from ESPN API.
    """
    if not date_str or date_str.endswith('Z'):
        return date_str
    
    try:
        # Try parsing with timezone info first
        if '+' in date_str or date_str.count('-') > 2:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            # No timezone info, assume UTC
            dt = datetime.fromisoformat(date_str)
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, AttributeError):
        # If parsing fails, return as-is (might already be correct)
        return date_str


def get_current_season() -> int:
    """Get current NFL season year."""
    now = datetime.now()
    # NFL season typically starts in September
    if now.month >= 9:
        return now.year
    else:
        return now.year - 1


def get_week_number() -> tuple[int, int]:
    """
    Calculate the previous week and upcoming week numbers.
    Uses ESPN API to get the current week, or estimates based on date.
    Returns (previous_week, upcoming_week)
    
    Logic: On Tuesday after games, ESPN may still show the completed week as "current".
    We check if games are completed to determine if we should use that week as "previous"
    and the next week as "upcoming".
    """
    now = datetime.now()
    season = get_current_season()
    
    # Try to get current week from ESPN API
    try:
        url = f"{ESPN_BASE_URL}/scoreboard"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # ESPN provides week information in the response
        week_info = data.get("week", {})
        if week_info:
            current_week = week_info.get("number", None)
            if current_week:
                # Check if games in the current week are completed
                # This helps determine if we're looking at a completed week or upcoming week
                events = data.get("events", [])
                completed_games = 0
                total_games = len(events)
                
                for event in events:
                    status = event.get("status", {})
                    status_type = status.get("type", {})
                    if status_type.get("completed", False):
                        completed_games += 1
                
                # If most games are completed (>= 80% or it's Tuesday/Wednesday), 
                # treat current week as "previous" and next week as "upcoming"
                # This handles the case where ESPN still shows week 13 as "current" on Tuesday
                is_tuesday_or_wednesday = now.weekday() in [1, 2]  # Monday=0, Tuesday=1, Wednesday=2
                mostly_completed = total_games > 0 and (completed_games / total_games) >= 0.8
                
                if mostly_completed or is_tuesday_or_wednesday:
                    # Current week is completed, so it's "previous"
                    previous_week = current_week
                    upcoming_week = min(18, current_week + 1)
                else:
                    # Current week is still in progress or upcoming
                    previous_week = max(1, current_week - 1)
                    upcoming_week = current_week
                
                return previous_week, upcoming_week
    except (requests.RequestException, KeyError, ValueError):
        pass
    
    # Fallback: Estimate week based on date
    # NFL season typically starts first week of September
    # Week 1 is usually around September 5-10
    season_start = datetime(season, 9, 5)
    
    # Calculate weeks since season start
    days_since_start = (now - season_start).days
    current_week = max(1, (days_since_start // 7) + 1)
    
    # Cap at 18 (regular season)
    current_week = min(current_week, 18)
    
    # On Tuesday/Wednesday, assume current week is completed
    is_tuesday_or_wednesday = now.weekday() in [1, 2]
    if is_tuesday_or_wednesday:
        previous_week = current_week
        upcoming_week = min(18, current_week + 1)
    else:
        previous_week = max(1, current_week - 1)
        upcoming_week = current_week
    
    return previous_week, upcoming_week


def fetch_espn_scores(season: int, week: int) -> Optional[List[Dict]]:
    """Fetch scores from ESPN API for a specific week."""
    try:
        # ESPN API uses week parameter directly
        url = f"{ESPN_BASE_URL}/scoreboard"
        params = {
            "seasontype": 2,  # Regular season (1=preseason, 2=regular, 3=postseason)
            "week": week
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            
            # Normalize date to Zulu format
            date_str = normalize_date_to_zulu(event.get("date", ""))
            
            game_data = {
                "date": date_str,
                "home_team": "",
                "away_team": "",
                "home_score": None,
                "away_score": None,
                "status": event.get("status", {}).get("type", {}).get("description", ""),
                "completed": event.get("status", {}).get("type", {}).get("completed", False)
            }
            
            # Extract team names and scores
            for competitor in competition.get("competitors", []):
                team = competitor.get("team", {})
                team_name = team.get("displayName", "") or team.get("name", "")
                
                if competitor.get("homeAway") == "home":
                    game_data["home_team"] = team_name
                    game_data["home_score"] = competitor.get("score")
                else:
                    game_data["away_team"] = team_name
                    game_data["away_score"] = competitor.get("score")
            
            games.append(game_data)
        
        return games if games else None
    except requests.RequestException as e:
        print(f"Error fetching ESPN scores: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error parsing ESPN scores data: {e}")
        return None


def fetch_espn_schedule(season: int, week: int) -> Optional[List[Dict]]:
    """Fetch schedule from ESPN API for a specific week."""
    try:
        url = f"{ESPN_BASE_URL}/scoreboard"
        params = {
            "seasontype": 2,  # Regular season
            "week": week
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            
            # Normalize date to Zulu format
            date_str = normalize_date_to_zulu(event.get("date", ""))
            
            game_data = {
                "date": date_str,
                "home_team": "",
                "away_team": "",
                "spread": None,
                "over_under": None,
                "spread_source": None
            }
            
            # Extract team names
            for competitor in competition.get("competitors", []):
                team = competitor.get("team", {})
                team_name = team.get("displayName", "") or team.get("name", "")
                
                if competitor.get("homeAway") == "home":
                    game_data["home_team"] = team_name
                else:
                    game_data["away_team"] = team_name
            
            # Try to get odds from ESPN
            odds = competition.get("odds", [])
            if odds:
                for odd in odds:
                    provider = odd.get("provider", {})
                    provider_name = provider.get("name", "")
                    
                    # Check for ESPN BET or other providers
                    if "ESPN" in provider_name.upper() or "BET" in provider_name.upper():
                        details = odd.get("details", "")
                        # Parse spread from details (format: "TEAM -3.5")
                        if details:
                            game_data["spread"] = details
                            game_data["spread_source"] = provider_name
                            break
            
            games.append(game_data)
        
        return games if games else None
    except requests.RequestException as e:
        print(f"Error fetching ESPN schedule: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error parsing ESPN schedule data: {e}")
        return None


def parse_spread(spread_str: str, home_team: str, away_team: str) -> Optional[float]:
    """
    Parse spread string and return as float with respect to home team.
    In our system: Negative = home favorite, Positive = home underdog.
    
    Handles ESPN spread strings which may include team names or show spreads
    from different perspectives. ESPN often uses abbreviations like "LAR", "SF", etc.
    """
    if not spread_str:
        return None
    
    # Try to extract number from string
    # Look for patterns like "-3.5", "+3.5", "3.5", etc.
    match = re.search(r'([+-]?\d+\.?\d*)', spread_str)
    if not match:
        return None
    
    spread_value = float(match.group(1))
    
    # Extract the team name/abbreviation from the spread string
    # ESPN format is typically "TEAM -3.5" or "TEAM +3.5"
    # Remove the number and any whitespace to get the team identifier
    team_identifier = re.sub(r'[+-]?\d+\.?\d*', '', spread_str).strip().upper()
    
    # Use the improved team matching function to check which team this refers to
    home_match = teams_match(team_identifier, home_team)
    away_match = teams_match(team_identifier, away_team)
    
    if home_match:
        # Source explicitly mentions home team
        # Source shows "Home -7" or "CAR -7" (home favored) -> store -7 (home favorite) ✓
        # Source shows "Home +10" or "CAR +10" (home underdog) -> store +10 (home underdog) ✓
        # Keep as-is: standard betting notation matches our system
        return spread_value
    elif away_match:
        # Source shows away team - flip perspective
        # Source shows "Away -10" or "LAR -10" (away favored) -> home underdog -> store +10
        # Source shows "Away +7" or "LAR +7" (away underdog) -> home favored -> store -7
        # Negate to flip from away perspective to home perspective
        return -spread_value
    else:
        # No team match found - return as-is (might be incorrect, but better than None)
        return spread_value


# Common team name mappings and abbreviations
TEAM_ABBREVIATIONS = {
    "CAR": "CAROLINA PANTHERS", "CAROLINA": "CAROLINA PANTHERS", "PANTHERS": "CAROLINA PANTHERS",
    "LAR": "LOS ANGELES RAMS", "LA RAMS": "LOS ANGELES RAMS", "RAMS": "LOS ANGELES RAMS",
    "CLE": "CLEVELAND BROWNS", "BROWNS": "CLEVELAND BROWNS",
    "SF": "SAN FRANCISCO 49ERS", "49ERS": "SAN FRANCISCO 49ERS", "SAN FRANCISCO": "SAN FRANCISCO 49ERS",
    "IND": "INDIANAPOLIS COLTS", "COLTS": "INDIANAPOLIS COLTS",
    "HOU": "HOUSTON TEXANS", "TEXANS": "HOUSTON TEXANS",
    "MIA": "MIAMI DOLPHINS", "DOLPHINS": "MIAMI DOLPHINS",
    "NO": "NEW ORLEANS SAINTS", "SAINTS": "NEW ORLEANS SAINTS", "NEW ORLEANS": "NEW ORLEANS SAINTS",
    "NYJ": "NEW YORK JETS", "JETS": "NEW YORK JETS",
    "ATL": "ATLANTA FALCONS", "FALCONS": "ATLANTA FALCONS",
    "TB": "TAMPA BAY BUCCANEERS", "BUCCANEERS": "TAMPA BAY BUCCANEERS", "TAMPA": "TAMPA BAY BUCCANEERS",
    "ARI": "ARIZONA CARDINALS", "CARDINALS": "ARIZONA CARDINALS",
    "TEN": "TENNESSEE TITANS", "TITANS": "TENNESSEE TITANS",
    "JAX": "JACKSONVILLE JAGUARS", "JAGUARS": "JACKSONVILLE JAGUARS", "JAC": "JACKSONVILLE JAGUARS",
    "SEA": "SEATTLE SEAHAWKS", "SEAHAWKS": "SEATTLE SEAHAWKS",
    "MIN": "MINNESOTA VIKINGS", "VIKINGS": "MINNESOTA VIKINGS",
    "LAC": "LOS ANGELES CHARGERS", "CHARGERS": "LOS ANGELES CHARGERS",
    "LV": "LAS VEGAS RAIDERS", "RAIDERS": "LAS VEGAS RAIDERS", "OAK": "LAS VEGAS RAIDERS",
    "PIT": "PITTSBURGH STEELERS", "STEELERS": "PITTSBURGH STEELERS",
    "BUF": "BUFFALO BILLS", "BILLS": "BUFFALO BILLS",
    "WAS": "WASHINGTON COMMANDERS", "COMMANDERS": "WASHINGTON COMMANDERS", "WFT": "WASHINGTON COMMANDERS",
    "DEN": "DENVER BRONCOS", "BRONCOS": "DENVER BRONCOS",
    "NE": "NEW ENGLAND PATRIOTS", "PATRIOTS": "NEW ENGLAND PATRIOTS", "NEW ENGLAND": "NEW ENGLAND PATRIOTS",
    "NYG": "NEW YORK GIANTS", "GIANTS": "NEW YORK GIANTS",
}


def normalize_team_name_for_matching(team_name: str) -> tuple[str, set[str]]:
    """
    Normalize team name for matching and return key identifiers.
    Returns (normalized_name, set_of_key_words)
    """
    normalized = team_name.upper().strip()
    
    # Check if it's a known abbreviation and expand it
    if normalized in TEAM_ABBREVIATIONS:
        normalized = TEAM_ABBREVIATIONS[normalized]
    
    # Extract key words (length > 2 to avoid "LA", "NY", etc.)
    words = set(w for w in normalized.split() if len(w) > 2)
    # Also add the last word (usually the team type: Panthers, Rams, etc.)
    if normalized:
        last_word = normalized.split()[-1] if len(normalized.split()[-1]) > 2 else None
        if last_word:
            words.add(last_word)
    return normalized, words


def teams_match(team1: str, team2: str) -> bool:
    """
    Check if two team names refer to the same team.
    Handles variations like "Carolina Panthers" vs "Panthers" vs "Carolina"
    """
    norm1, words1 = normalize_team_name_for_matching(team1)
    norm2, words2 = normalize_team_name_for_matching(team2)
    
    # Exact match
    if norm1 == norm2:
        return True
    
    # One contains the other
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    # Check for significant word overlap
    common_words = words1 & words2
    if len(common_words) > 0:
        # If we have at least one significant word in common, and it's not a generic word
        # Generic words to ignore: "NEW", "YORK", "LOS", "ANGELES", "SAN", "FRANCISCO"
        generic_words = {"NEW", "YORK", "LOS", "ANGELES", "SAN", "FRANCISCO", "SAINT", "SAINTS"}
        meaningful_common = common_words - generic_words
        if len(meaningful_common) > 0:
            return True
        # If only generic words match, require at least 2 matches
        if len(common_words) >= 2:
            return True
    
    return False


def fetch_odds_api_spreads(season: int, week: int) -> Optional[Dict]:
    """Fetch point spreads from The Odds API."""
    if not ODDS_API_KEY:
        return None
    
    try:
        # The Odds API endpoint
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads",
            "oddsFormat": "american"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Map odds data: key is (home_team, away_team), value is spread (with respect to home)
        spreads = {}
        for game in data:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            
            if not home_team or not away_team:
                continue
            
            # Get spread from bookmakers
            # In The Odds API: negative point = that team is favored, positive = underdog
            # In our system: negative = home favorite, positive = home underdog
            # The Odds API returns both outcomes (home and away) with opposite signs
            home_spread = None
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "spreads":
                        # Process all outcomes to find the home team's spread
                        # We need to identify which outcome belongs to the home team
                        # The Odds API returns both outcomes (home and away) with opposite signs
                        home_outcome_point = None
                        away_outcome_point = None
                        
                        for outcome in market.get("outcomes", []):
                            outcome_name = outcome.get("name", "")
                            point = outcome.get("point", None)
                            
                            if point is None:
                                continue
                            
                            # Use improved team matching function
                            is_home_outcome = teams_match(outcome_name, home_team)
                            is_away_outcome = teams_match(outcome_name, away_team)
                            
                            # If both match (shouldn't happen, but handle it), prefer the more specific match
                            if is_home_outcome and is_away_outcome:
                                # Prefer the match where the outcome name is more similar to one team name
                                home_similarity = len(set(outcome_name.upper().split()) & set(home_team.upper().split()))
                                away_similarity = len(set(outcome_name.upper().split()) & set(away_team.upper().split()))
                                if home_similarity > away_similarity:
                                    is_away_outcome = False
                                elif away_similarity > home_similarity:
                                    is_home_outcome = False
                                else:
                                    # If still tied, don't match either (ambiguous)
                                    is_home_outcome = False
                                    is_away_outcome = False
                            
                            # Store the point value for the appropriate team
                            if is_home_outcome:
                                home_outcome_point = point
                            elif is_away_outcome:
                                away_outcome_point = point
                        
                        # Determine home spread from the outcomes we found
                        # Prefer home team's outcome directly, fall back to negating away team's outcome
                        if home_outcome_point is not None:
                            # Found the home team's spread directly
                            # In The Odds API: point is relative to the named team
                            # -10.5 for home team = home favored by 10.5 -> store -10.5 ✓
                            # +10.5 for home team = home underdog by 10.5 -> store +10.5 ✓
                            home_spread = home_outcome_point
                        elif away_outcome_point is not None:
                            # Found the away team's spread, flip it for home perspective
                            # -10.5 for away team = away favored (home underdog) -> store +10.5 ✓
                            # +10.5 for away team = away underdog (home favored) -> store -10.5 ✓
                            home_spread = -away_outcome_point
                        # If we couldn't match either team but have outcomes, try to infer from signs
                        # This is a fallback - The Odds API should have team names, but handle edge cases
                        elif home_outcome_point is None and away_outcome_point is None:
                            # Collect all outcomes we saw
                            all_outcomes = []
                            for outcome in market.get("outcomes", []):
                                outcome_name = outcome.get("name", "")
                                point = outcome.get("point", None)
                                if point is not None:
                                    all_outcomes.append((outcome_name, point))
                            
                            # If we have exactly 2 outcomes with opposite signs, we can infer
                            if len(all_outcomes) == 2:
                                outcome1_name, outcome1_point = all_outcomes[0]
                                outcome2_name, outcome2_point = all_outcomes[1]
                                # They should have opposite signs
                                if (outcome1_point < 0 and outcome2_point > 0) or (outcome1_point > 0 and outcome2_point < 0):
                                    # Try one more time with very loose matching
                                    # Check if either outcome name has ANY word in common with home team
                                    home_words = set(w for w in home_team.upper().split() if len(w) > 2)
                                    outcome1_words = set(w for w in outcome1_name.upper().split() if len(w) > 2)
                                    outcome2_words = set(w for w in outcome2_name.upper().split() if len(w) > 2)
                                    
                                    outcome1_home_overlap = len(outcome1_words & home_words)
                                    outcome2_home_overlap = len(outcome2_words & home_words)
                                    
                                    if outcome1_home_overlap > outcome2_home_overlap:
                                        home_spread = outcome1_point
                                    elif outcome2_home_overlap > outcome1_home_overlap:
                                        home_spread = outcome2_point
                        
                        if home_spread is not None:
                            break
                
                if home_spread is not None:
                    break
            
            if home_spread is not None:
                key = (home_team, away_team)
                spreads[key] = home_spread
            else:
                print(f"Warning: Could not find spread for {home_team} vs {away_team}")
        
        return spreads
    except requests.RequestException as e:
        print(f"Error fetching Odds API spreads: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error parsing Odds API spreads data: {e}")
        return None


def find_matching_spread(home_team: str, away_team: str, spreads: Dict) -> Optional[float]:
    """
    Find matching spread for a game from the spreads dictionary.
    Returns spread value with respect to home team, or None if not found.
    """
    if not spreads:
        return None
    
    # Try exact match first (using tuple key from Odds API)
    key = (home_team, away_team)
    if key in spreads:
        return spreads[key]
    
    # Try fuzzy matching - check if team names are contained in keys
    home_upper = home_team.upper()
    away_upper = away_team.upper()
    
    for (spread_home, spread_away), spread_value in spreads.items():
        if isinstance(spread_value, (int, float)):
            # Check if teams match (case-insensitive, partial match)
            if (home_upper in spread_home.upper() or spread_home.upper() in home_upper) and \
               (away_upper in spread_away.upper() or spread_away.upper() in away_upper):
                return spread_value
    
    return None


def save_scores_to_db(scores: List[Dict], week: int, season: int):
    """
    Update database with previous week's scores.
    Only updates existing games - does not insert new rows.
    This is because we don't want to add games without known spreads.
    """
    updated_count = 0
    skipped_count = 0
    
    for game in scores:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        date = game.get("date", "")
        home_score = game.get("home_score")
        visitor_score = game.get("away_score")
        
        if not home or not away or not date:
            continue
        
        # Only update if we have scores
        if home_score is not None and visitor_score is not None:
            # Check if game exists - only update existing games, don't insert new ones
            if game_exists(home, away, date):
                # Update existing game with scores (spread stays as-is from when it was inserted)
                if update_game_scores(home, away, date, home_score, visitor_score):
                    if updated_count == 0:
                        print("Updating scores:")
                    updated_count += 1
                    print(f"  {home} {home_score}, {away} {visitor_score}")
            else:
                # Game doesn't exist - skip it (we don't want to add games without spreads)
                skipped_count += 1
    
    print(f"Updated {updated_count} games with scores")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} games that weren't in database (no spread available)")
    
    # Dump database to text file for Git storage
    if updated_count > 0:
        dump_sqlite_to_file(DB_PATH, DB_DUMP_PATH)


def save_schedule_to_db(schedule: List[Dict], week: int, season: int, spreads: Optional[Dict] = None, prompt_for_quarterbacks: bool = False):
    """Insert upcoming week's games into database (without scores)."""
    inserted_count = 0
    updated_count = 0
    
    for game in schedule:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        date = game.get("date", "")
        
        if not home or not away or not date:
            continue
        
        # Get spread from game data or from spreads dict
        # Prioritize The Odds API since it explicitly identifies which team each spread is for
        spread = None
        
        # First, try to get from Odds API (more reliable - explicitly identifies teams)
        if spreads:
            spread = find_matching_spread(home, away, spreads)
        
        # If no spread from Odds API, try to parse from ESPN data
        if spread is None:
            spread_str = game.get("spread", "")
            if spread_str:
                spread = parse_spread(spread_str, home, away)
        
        # Get expected quarterbacks if prompting
        home_expected_qb = None
        visitor_expected_qb = None
        if prompt_for_quarterbacks:
            # Get most recent expected quarterback for home team
            home_qb = get_most_recent_expected_qb(home, season, week)
            home_prompt = f"{home} QB"
            if home_qb:
                home_prompt += f" [{home_qb}]"
            home_prompt += ": "
            
            home_input = input(home_prompt).strip()
            home_expected_qb = home_input if home_input else home_qb
            
            # Get most recent expected quarterback for away team
            away_qb = get_most_recent_expected_qb(away, season, week)
            away_prompt = f"{away} QB"
            if away_qb:
                away_prompt += f" [{away_qb}]"
            away_prompt += ": "
            
            away_input = input(away_prompt).strip()
            visitor_expected_qb = away_input if away_input else away_qb
        
        # Check if game exists before upserting
        existed = game_exists(home, away, date)
        
        # Insert/update game without scores (actual_qb columns, win_prob_lr, win_prob_dl remain NULL)
        if upsert_game(home, away, date, spread, None, None, season, week, home_expected_qb, visitor_expected_qb, None, None, None, None):
            if inserted_count == 0 and updated_count == 0:
                print("Updating schedule:")
            spread_display = f"{spread:+.1f}" if spread is not None else "N/A"
            print(f"  {home} vs {away}, spread: {spread_display}")
            if home_expected_qb or visitor_expected_qb:
                qb_display = []
                if home_expected_qb:
                    qb_display.append(f"{home}: {home_expected_qb}")
                if visitor_expected_qb:
                    qb_display.append(f"{away}: {visitor_expected_qb}")
                print(f"    Expected QBs: {', '.join(qb_display)}")
            if existed:
                updated_count += 1
            else:
                inserted_count += 1
    
    print(f"Inserted {inserted_count} new games, updated {updated_count} existing games")
    
    # Dump database to text file for Git storage
    if inserted_count > 0 or updated_count > 0:
        dump_sqlite_to_file(DB_PATH, DB_DUMP_PATH)


def main():
    """Main function to fetch and save NFL data."""
    parser = argparse.ArgumentParser(
        description="NFL Weekly Update Script - Fetches scores and schedules with spreads"
    )
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="Only fetch upcoming week's schedule and spreads (useful before week 1 or starting mid-season)"
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only update previous week's results (useful after week 18)"
    )
    
    args = parser.parse_args()
    
    print("NFL Weekly Update Script")
    print("=" * 50)
    
    # Check if database exists
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Please run 'python db_utils.py --create' first to create the database.")
        sys.exit(1)
    
    season = get_current_season()
    previous_week, upcoming_week = get_week_number()
    
    print(f"Season: {season}")
    if not args.schedule_only:
        print(f"Previous Week: {previous_week}")
    if not args.results_only:
        print(f"Upcoming Week: {upcoming_week}")
    print()
    
    # Determine what to run based on flags
    # If both flags are set or neither is set, run both (default behavior)
    run_results = not args.schedule_only
    run_schedule = not args.results_only
    
    # Fetch previous week's scores
    if run_results:
        print("Fetching previous week's scores...")
        scores = fetch_espn_scores(season, previous_week)
        
        if scores:
            save_scores_to_db(scores, previous_week, season)
        else:
            print("Warning: Could not fetch scores. Database not updated.")
        
        if run_schedule:
            print()
    
    # Fetch upcoming week's schedule
    if run_schedule:
        print("Fetching upcoming week's schedule...")
        schedule = fetch_espn_schedule(season, upcoming_week)
        
        if schedule:
            # Try to get spreads from The Odds API
            print("Fetching point spreads...")
            if not ODDS_API_KEY:
                print("Warning: ODDS_API_KEY not set. Will only use ESPN spreads if available.")
            spreads = fetch_odds_api_spreads(season, upcoming_week)
            
            if spreads:
                print(f"Successfully fetched spreads from The Odds API for {len(spreads)} games")
            else:
                # Check if ESPN already provided spreads
                games_with_spreads = sum(1 for g in schedule if g.get("spread"))
                if games_with_spreads > 0:
                    print(f"Note: Found spreads from ESPN for {games_with_spreads} games.")
                else:
                    print("Note: Could not fetch spreads from APIs. Games saved without spreads.")
                    print("      Consider setting ODDS_API_KEY environment variable for spreads.")
            
            save_schedule_to_db(schedule, upcoming_week, season, spreads, prompt_for_quarterbacks=args.schedule_only)
        else:
            print("Warning: Could not fetch schedule. Database not updated.")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
