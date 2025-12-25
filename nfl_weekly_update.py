#!/usr/bin/env -S uv run --
"""
NFL Weekly Update Script
Fetches previous week's scores and upcoming week's schedule with point spreads.
Run every Tuesday during NFL season.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import requests
from typing import Dict, List, Optional
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import database functions
from db import (upsert_game, game_exists, update_game_scores, get_most_recent_expected_qb,
                get_most_recent_actual_qb, get_expected_qb_from_game)
from db_utils import dump_sqlite_to_file

# Configuration
DB_PATH = Path(__file__).parent / "nfl_results.db"
DB_DUMP_PATH = Path(__file__).parent / "db_dump.sql"

# API Keys (set as environment variables or in .env file)
# For The Odds API: https://the-odds-api.com/
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ESPN API endpoints (public, no key required)
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def normalize_date_to_eastern(date_str: str) -> str:
    """
    Normalize a date string to US Eastern Time (EDT/EST, automatically selected based on date).
    Handles various ISO 8601 formats from ESPN API.
    Returns ISO 8601 format with Eastern timezone offset (e.g., "2024-01-15T13:00:00-05:00" for EST
    or "2024-07-15T13:00:00-04:00" for EDT).
    """
    if not date_str:
        return date_str
    
    # If already in Eastern format (has offset like -05:00 or -04:00), return as-is
    if ('-05:00' in date_str or '-04:00' in date_str) and not date_str.endswith('Z'):
        return date_str
    
    try:
        # US Eastern timezone (automatically handles EDT/EST)
        eastern_tz = ZoneInfo("America/New_York")
        
        # Parse the date string - handle various formats
        if date_str.endswith('Z'):
            # UTC date with Z suffix
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        elif '+' in date_str or (date_str.count('-') > 2 and 'T' in date_str and ('+' in date_str or (len(date_str) > 6 and date_str[-6] in ['+', '-']))):
            # Has timezone offset already (e.g., +00:00 or -05:00)
            dt = datetime.fromisoformat(date_str)
        else:
            # No timezone info - assume UTC
            dt = datetime.fromisoformat(date_str)
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to Eastern time (automatically handles EDT/EST based on date)
        dt_eastern = dt.astimezone(eastern_tz)
        
        # Return in ISO 8601 format with timezone offset
        return dt_eastern.isoformat()
    except Exception as e:
        # If parsing fails, return as-is (might already be correct)
        print(f"Warning: Could not parse date '{date_str}': {e}")
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
            
            # Normalize date to US Eastern Time
            date_str = normalize_date_to_eastern(event.get("date", ""))
            
            game_data = {
                "date": date_str,
                "home_team": "",
                "away_team": "",
                "home_score": None,
                "away_score": None,
                "status": event.get("status", {}).get("type", {}).get("description", ""),
                "completed": event.get("status", {}).get("type", {}).get("completed", False),
                "game_id": event.get("id")  # ESPN event ID for fetching QB data
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
            
            # Normalize date to US Eastern Time
            date_str = normalize_date_to_eastern(event.get("date", ""))
            
            game_data = {
                "date": date_str,
                "home_team": "",
                "away_team": "",
                "spread": None,
                "over_under": None,
                "spread_source": None,
                "game_id": event.get("id")  # ESPN event ID for fetching QB data
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


def normalize_qb_name(qb_name: Optional[str]) -> Optional[str]:
    """
    Normalize quarterback name:
    - Replace non-alphabetic characters with spaces
    - Convert to title case (initial cap)
    - Trim leading and trailing whitespace
    
    Args:
        qb_name: Quarterback name to normalize, or None
    
    Returns:
        Normalized name, or None if input was None
    """
    if not qb_name:
        return None
    
    # Replace non-alphabetic characters (except spaces) with spaces
    normalized = re.sub(r'[^a-zA-Z\s]', ' ', qb_name)
    
    # Convert to title case (first letter of each word capitalized)
    normalized = normalized.title()
    
    # Replace multiple spaces with single space and trim
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized if normalized else None


# Team name to ESPN URL slug mapping (for scraping depth charts)
TEAM_TO_ESPN_SLUG = {
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "lv",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "wsh",
}


def fetch_expected_quarterback(team_name: str, game_id: Optional[str] = None) -> Optional[str]:
    """
    Fetch expected starting quarterback for a team from the web.
    Tries ESPN API first, then falls back to scraping ESPN depth chart page.
    
    Args:
        team_name: Name of the team
        game_id: Optional ESPN game/event ID (not currently used but reserved for API endpoint)
    
    Returns:
        Quarterback name, or None if not found
    """
    try:
        # Try scraping ESPN depth chart
        # Use exact matching only to avoid confusion between similar team names
        slug = None
        team_name_upper = team_name.upper().strip()
        
        # Try exact match first (case-insensitive)
        for team_key, team_slug in TEAM_TO_ESPN_SLUG.items():
            if team_key.upper().strip() == team_name_upper:
                slug = team_slug
                break
        
        # If no exact match, try partial matching but be very careful
        # Only match if the team name contains key unique words from the dictionary key
        if not slug:
            best_match = None
            best_score = 0
            
            for team_key, team_slug in TEAM_TO_ESPN_SLUG.items():
                key_upper = team_key.upper().strip()
                
                # Extract key identifying words (last word is usually unique: Giants, Jets, etc.)
                input_words = set(w for w in team_name_upper.split() if len(w) > 2)
                key_words = set(w for w in key_upper.split() if len(w) > 2)
                
                # For NY teams, require both "NEW YORK" AND the specific team name
                if "NEW YORK" in team_name_upper:
                    if "NEW YORK" not in key_upper:
                        continue  # Skip non-NY teams
                    # Must match the specific team identifier
                    if "JETS" in team_name_upper and "JETS" not in key_upper:
                        continue
                    if "GIANTS" in team_name_upper and "GIANTS" not in key_upper:
                        continue
                    if "JETS" in key_upper and "JETS" not in team_name_upper:
                        continue
                    if "GIANTS" in key_upper and "GIANTS" not in team_name_upper:
                        continue
                
                # Calculate match score - require last word to match (team type)
                input_last = team_name_upper.split()[-1] if team_name_upper else ""
                key_last = key_upper.split()[-1] if key_upper else ""
                
                if input_last == key_last:
                    # Last words match - this is likely the right team
                    # Calculate how many words match
                    common_words = input_words & key_words
                    # Remove generic words from scoring
                    generic = {"NEW", "YORK", "LOS", "ANGELES", "SAN", "FRANCISCO", "SAINT", "SAINTS", "VEGAS", "LAS"}
                    meaningful_common = common_words - generic
                    score = len(meaningful_common) + (2 if input_last == key_last else 0)
                    
                    if score > best_score:
                        best_score = score
                        best_match = team_slug
            
            if best_match and best_score >= 1:  # Require at least one meaningful word match
                slug = best_match
        
        if not slug:
            return None
        
        from bs4 import BeautifulSoup
        
        # Small delay to avoid rate limiting
        time.sleep(1)
        
        url = f"https://www.espn.com/nfl/team/depth/_/name/{slug}"
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # ESPN depth chart structure: Look for table rows with position data
        # Try multiple strategies to find the QB
        qb_text = None
        
        # Strategy 1: Look for table with depth chart data
        # ESPN uses tables with position in first column, players in subsequent columns
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    # Check if first cell contains "QB"
                    first_cell_text = cells[0].get_text(strip=True).upper()
                    if 'QB' in first_cell_text or 'QUARTERBACK' in first_cell_text:
                        # Get the first player (starter) from second cell
                        player_cell = cells[1]
                        # Look for player link or text
                        player_link = player_cell.find('a', href=re.compile(r'/player/', re.I))
                        if player_link:
                            qb_text = player_link.get_text(strip=True)
                            break
                        else:
                            # Try direct text
                            text = player_cell.get_text(strip=True)
                            if text and len(text) > 2:
                                qb_text = text
                                break
                if qb_text:
                    break
            if qb_text:
                break
        
        # Strategy 2: Look for div-based depth chart (newer ESPN layout)
        if not qb_text:
            # Look for sections/divs with position labels
            for element in soup.find_all(['div', 'section'], class_=re.compile(r'position|depth', re.I)):
                # Find QB label
                qb_label = element.find(string=re.compile(r'^QB\s*$', re.I))
                if not qb_label:
                    # Try case-insensitive
                    for text_node in element.find_all(string=True):
                        if re.match(r'^QB\s*$', text_node.strip(), re.I):
                            qb_label = text_node
                            break
                
                if qb_label:
                    # Find parent container and look for player name
                    container = qb_label.find_parent(['div', 'section', 'li'])
                    if container:
                        # Look for player links or names
                        player_links = container.find_all('a', href=re.compile(r'/player/', re.I))
                        if player_links:
                            qb_text = player_links[0].get_text(strip=True)
                            break
                        # Or look for span/div with player name classes
                        name_elements = container.find_all(['span', 'div'], class_=re.compile(r'name|player', re.I))
                        for name_elem in name_elements:
                            text = name_elem.get_text(strip=True)
                            if text and len(text) > 2 and not text.upper() in ['QB', 'QUARTERBACK']:
                                qb_text = text
                                break
                    if qb_text:
                        break
        
        # Strategy 3: Look for any link with /player/ that's near a QB label
        if not qb_text:
            # Find all QB labels/text
            qb_elements = soup.find_all(string=re.compile(r'\bQB\b', re.I))
            for qb_elem in qb_elements:
                # Look in nearby context (parent and siblings)
                parent = qb_elem.find_parent()
                if parent:
                    # Check same row/container
                    player_link = parent.find('a', href=re.compile(r'/player/', re.I))
                    if player_link:
                        qb_text = player_link.get_text(strip=True)
                        break
                    # Check siblings
                    if parent.parent:
                        for sibling in parent.parent.find_all('a', href=re.compile(r'/player/', re.I)):
                            qb_text = sibling.get_text(strip=True)
                            if qb_text:
                                break
                    if qb_text:
                        break
        
        if qb_text:
            return qb_text.strip()
        
        return None
    except requests.RequestException as e:
        # Network error - silently fail
        return None
    except Exception as e:
        # Other errors - silently fail
        return None


def fetch_actual_quarterback(game_id: str, team_name: str, is_home: bool) -> Optional[str]:
    """
    Fetch actual quarterback who played majority of snaps from box score.
    Tries ESPN API first, then falls back to scraping ESPN box score page.
    
    Args:
        game_id: ESPN game/event ID
        team_name: Name of the team
        is_home: True if home team, False if away team
    
    Returns:
        Quarterback name (who played majority of snaps), or None if not found
    """
    try:
        # Try ESPN API box score endpoint
        url = f"{ESPN_BASE_URL}/summary"
        params = {"event": game_id}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Look for passing statistics
            boxscore = data.get("boxscore", {})
            teams = boxscore.get("teams", [])
            
            team_index = 0 if is_home else 1
            if team_index < len(teams):
                team = teams[team_index]
                statistics = team.get("statistics", [])
                
                for stat_group in statistics:
                    if stat_group.get("name") == "passing":
                        athletes = stat_group.get("athletes", [])
                        if athletes:
                            # Find QB with most attempts or yards
                            best_qb = None
                            best_attempts = -1
                            
                            for athlete in athletes:
                                stats = athlete.get("stats", [])
                                attempts = None
                                for stat in stats:
                                    if stat.get("name") == "passingAttempts":
                                        attempts = stat.get("value", 0)
                                        break
                                
                                if attempts and attempts > best_attempts:
                                    best_attempts = attempts
                                    athlete_data = athlete.get("athlete", {})
                                    best_qb = athlete_data.get("displayName") or athlete_data.get("shortName")
                            
                            if best_qb:
                                return best_qb
        
        # Fallback: Try scraping ESPN box score page
        from bs4 import BeautifulSoup
        
        # Small delay to avoid rate limiting
        time.sleep(1)
        
        url = f"https://www.espn.com/nfl/boxscore/_/gameId/{game_id}"
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find passing stats table for the team
        # ESPN box score has team sections - need to identify which team is which
        passing_section = soup.find('section', string=re.compile(r'Passing', re.I))
        if not passing_section:
            # Try alternative structure
            passing_section = soup.find('div', class_=re.compile(r'passing', re.I))
        
        if passing_section:
            # Find table with passing stats
            table = passing_section.find_parent('table') or passing_section.find('table')
            if table:
                # Find rows with QB stats, look for highest attempts
                rows = table.find_all('tr')[1:]  # Skip header
                best_qb = None
                best_attempts = -1
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        player_name = cells[0].get_text(strip=True)
                        attempts_text = cells[1].get_text(strip=True)
                        try:
                            attempts = int(attempts_text)
                            if attempts > best_attempts:
                                best_attempts = attempts
                                best_qb = player_name
                        except ValueError:
                            pass
                
                if best_qb:
                    return best_qb
        
        return None
    except Exception as e:
        # Silently fail - return None to allow fallback
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


def save_scores_to_db(scores: List[Dict], week: int, season: int, prompt_for_quarterbacks: bool = False, accept_default_qbs: bool = False):
    """
    Update database with previous week's scores.
    Only updates existing games - does not insert new rows.
    This is because we don't want to add games without known spreads.
    
    Args:
        scores: List of game dictionaries with scores
        week: Week number
        season: Season year
        prompt_for_quarterbacks: If True, prompt for actual quarterbacks
    """
    updated_count = 0
    skipped_count = 0
    
    for game in scores:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        date = game.get("date", "")
        home_score = game.get("home_score")
        visitor_score = game.get("away_score")
        game_id = game.get("game_id")
        
        if not home or not away or not date:
            continue
        
        # Only update if we have scores
        if home_score is not None and visitor_score is not None:
            # Check if game exists - only update existing games, don't insert new ones
            if game_exists(home, away, date):
                # Get actual quarterbacks if prompting
                home_actual_qb = None
                visitor_actual_qb = None
                
                if prompt_for_quarterbacks:
                    # Priority 1: Expected QB from same game
                    home_expected_qb = get_expected_qb_from_game(home, away, date, home)
                    # Priority 2: QB from box score (if available)
                    home_box_qb = fetch_actual_quarterback(game_id, home, True) if game_id else None
                    home_default_qb = home_expected_qb or home_box_qb
                    
                    if accept_default_qbs:
                        # Automatically accept the default
                        home_actual_qb = normalize_qb_name(home_default_qb)
                    else:
                        home_prompt = f"{home} Actual QB"
                        if home_default_qb:
                            home_prompt += f" [{home_default_qb}]"
                        home_prompt += ": "
                        
                        home_input = input(home_prompt).strip()
                        home_actual_qb_raw = home_input if home_input else home_default_qb
                        home_actual_qb = normalize_qb_name(home_actual_qb_raw)
                    
                    # For visitor team
                    visitor_expected_qb = get_expected_qb_from_game(home, away, date, away)
                    visitor_box_qb = fetch_actual_quarterback(game_id, away, False) if game_id else None
                    visitor_default_qb = visitor_expected_qb or visitor_box_qb
                    
                    if accept_default_qbs:
                        # Automatically accept the default
                        visitor_actual_qb = normalize_qb_name(visitor_default_qb)
                    else:
                        visitor_prompt = f"{away} Actual QB"
                        if visitor_default_qb:
                            visitor_prompt += f" [{visitor_default_qb}]"
                        visitor_prompt += ": "
                        
                        visitor_input = input(visitor_prompt).strip()
                        visitor_actual_qb_raw = visitor_input if visitor_input else visitor_default_qb
                        visitor_actual_qb = normalize_qb_name(visitor_actual_qb_raw)
                
                # Update game with scores and actual QBs using upsert_game
                if upsert_game(home, away, date, None, home_score, visitor_score, season, week,
                              None, None, home_actual_qb, visitor_actual_qb, None, None):
                    if updated_count == 0:
                        print("Updating scores:")
                    updated_count += 1
                    print(f"  {home} {home_score}, {away} {visitor_score}")
                    if home_actual_qb or visitor_actual_qb:
                        qb_display = []
                        if home_actual_qb:
                            qb_display.append(f"{home}: {home_actual_qb}")
                        if visitor_actual_qb:
                            qb_display.append(f"{away}: {visitor_actual_qb}")
                        print(f"    Actual QBs: {', '.join(qb_display)}")
            else:
                # Game doesn't exist - skip it (we don't want to add games without spreads)
                skipped_count += 1
    
    print(f"Updated {updated_count} games with scores")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} games that weren't in database (no spread available)")
    
    # Dump database to text file for Git storage
    if updated_count > 0:
        dump_sqlite_to_file(DB_PATH, DB_DUMP_PATH)


def save_schedule_to_db(schedule: List[Dict], week: int, season: int, spreads: Optional[Dict] = None, prompt_for_quarterbacks: bool = False, accept_default_qbs: bool = False):
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
            # For home team: Priority 1) web fetch, Priority 2) last week's actual QB
            home_web_qb = fetch_expected_quarterback(home)
            home_last_actual_qb = get_most_recent_actual_qb(home, season, week) if not home_web_qb else None
            home_default_qb = home_web_qb or home_last_actual_qb
            
            if accept_default_qbs:
                # Automatically accept the default
                home_expected_qb = normalize_qb_name(home_default_qb)
            else:
                home_prompt = f"{home} QB"
                if home_default_qb:
                    home_prompt += f" [{home_default_qb}]"
                home_prompt += ": "
                
                home_input = input(home_prompt).strip()
                home_expected_qb_raw = home_input if home_input else home_default_qb
                home_expected_qb = normalize_qb_name(home_expected_qb_raw)
            
            # For away team: Priority 1) web fetch, Priority 2) last week's actual QB
            away_web_qb = fetch_expected_quarterback(away)
            away_last_actual_qb = get_most_recent_actual_qb(away, season, week) if not away_web_qb else None
            away_default_qb = away_web_qb or away_last_actual_qb
            
            if accept_default_qbs:
                # Automatically accept the default
                visitor_expected_qb = normalize_qb_name(away_default_qb)
            else:
                away_prompt = f"{away} QB"
                if away_default_qb:
                    away_prompt += f" [{away_default_qb}]"
                away_prompt += ": "
                
                away_input = input(away_prompt).strip()
                visitor_expected_qb_raw = away_input if away_input else away_default_qb
                visitor_expected_qb = normalize_qb_name(visitor_expected_qb_raw)
        
        # Check if game exists before upserting
        existed = game_exists(home, away, date)
        
        # Insert/update game without scores (actual_qb columns, log_reg_win_prob, dl_win_prob remain NULL)
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
    parser.add_argument(
        "--accept-default-qbs",
        action="store_true",
        help="Automatically accept default quarterbacks without prompting (only works with --schedule-only)"
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
            save_scores_to_db(scores, previous_week, season, prompt_for_quarterbacks=False, accept_default_qbs=args.accept_default_qbs)
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
            
            save_schedule_to_db(schedule, upcoming_week, season, spreads, prompt_for_quarterbacks=args.schedule_only, accept_default_qbs=args.accept_default_qbs)
        else:
            print("Warning: Could not fetch schedule. Database not updated.")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
