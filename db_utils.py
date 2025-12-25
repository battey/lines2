#!/usr/bin/env -S uv run --
"""
Database utility script for managing the NFL results SQLite database.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone
from tabulate import tabulate

DB_PATH = Path(__file__).parent / "nfl_results.db"


def create_database():
    """Create the SQLite database and result table. Drops existing database if present (after confirmation)."""
    # Check if database already exists
    if DB_PATH.exists():
        print(f"Database already exists at {DB_PATH}")
        print("WARNING: This will delete the existing database and all its data!")

        # Ask for confirmation
        response = input("Are you sure you want to drop and recreate the database? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Database creation cancelled.")
            return

        # Close any existing connections and delete the database file
        try:
            # Try to close any open connections by attempting to connect and close
            temp_conn = sqlite3.connect(DB_PATH)
            temp_conn.close()
        except sqlite3.Error:
            pass

        # Delete the database file
        DB_PATH.unlink()
        print(f"Deleted existing database at {DB_PATH}")

    print("Creating NFL results database...")

    # Connect to database (creates file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create the result table
    # All columns are NOT NULL except home_score, visitor_score, spread, qb columns, and prediction columns
    # Note: created_at and updated_at will be set in application code to ensure Zulu format
    # Note: spread can be NULL when spreads aren't available (e.g., uncertain starting QB)
    # Note: expected_qb columns filled when gathering schedule, actual_qb columns filled when gathering results
    # Note: win_prob_lr and win_prob_dl are predictions (NULL for now)
    cursor.execute(
        """
        CREATE TABLE result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home TEXT NOT NULL,
            visitor TEXT NOT NULL,
            date TEXT NOT NULL,
            spread REAL,
            home_score INTEGER,
            visitor_score INTEGER,
            home_expected_qb TEXT,
            visitor_expected_qb TEXT,
            home_actual_qb TEXT,
            visitor_actual_qb TEXT,
            win_prob_lr REAL,
            win_prob_dl REAL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            UNIQUE(home, visitor, date)
        )
    """
    )

    # Create index for faster lookups
    cursor.execute(
        """
        CREATE INDEX idx_date ON result(date)
    """
    )

    cursor.execute(
        """
        CREATE INDEX idx_season_week ON result(season, week)
    """
    )

    conn.commit()
    conn.close()

    print(f"Database created successfully at {DB_PATH}")
    print("Table 'result' created with columns:")
    print("  - id (primary key)")
    print("  - home (home team name)")
    print("  - visitor (visitor team name)")
    print("  - date (game date/time)")
    print("  - spread (point spread, negative = home team favorite, NULL if unavailable)")
    print("  - home_score (home team score)")
    print("  - visitor_score (visitor team score)")
    print("  - home_expected_qb (home team expected quarterback, filled when gathering schedule)")
    print("  - visitor_expected_qb (visitor team expected quarterback, filled when gathering schedule)")
    print("  - home_actual_qb (home team actual quarterback who played, filled when gathering results)")
    print("  - visitor_actual_qb (visitor team actual quarterback who played, filled when gathering results)")
    print("  - win_prob_lr (win probability from logistic/linear regression, NULL for now)")
    print("  - win_prob_dl (win probability from deep learning, NULL for now)")
    print("  - season (NFL season year)")
    print("  - week (week number)")
    print("  - created_at, updated_at (timestamps)")


def truncate_database():
    """Truncate (delete all data from) the result table."""
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Use --create to create the database first.")
        return

    print(f"Truncating database at {DB_PATH}...")
    print("WARNING: This will delete all data from the result table!")

    # Ask for confirmation
    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("Truncate cancelled.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Delete all rows from the result table
        cursor.execute("DELETE FROM result")
        conn.commit()

        # Reset auto-increment counter (SQLite specific)
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='result'")
        conn.commit()

        print(f"Database truncated successfully. All data removed from result table.")
    except sqlite3.Error as e:
        print(f"Error truncating database: {e}")
        conn.rollback()
    finally:
        conn.close()


def dump_sqlite_to_file(db_path: Path, dump_path: Path) -> None:
    """
    Export the entire SQLite database as SQL text using sqlite3's iterdump().
    This produces a fully reconstructable SQL script.

    Args:
        db_path: Path to input .sqlite / .db file
        dump_path: Path to output .sql file
    """
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Dumping SQLite DB → {dump_path} ...")

    conn = sqlite3.connect(db_path)
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            for line in conn.iterdump():
                f.write(line + "\n")
        print("Dump completed.")
    except Exception as e:
        print(f"Error during dump: {e}")
    finally:
        conn.close()


def rebuild_sqlite_from_dump(dump_path: Path, db_path: Path) -> None:
    """
    Rebuild a SQLite database from a SQL dump file previously generated
    with iterdump() or sqlite3 .dump.

    WARNING: Overwrites the existing DB file.

    Args:
        dump_path: Path to input .sql file
        db_path: Path to output .sqlite DB
    """
    if not dump_path.exists():
        print(f"Error: Dump file not found at {dump_path}")
        return

    if db_path.exists():
        print(f"Deleting existing DB at {db_path} ...")
        try:
            db_path.unlink()
        except Exception as e:
            print(f"Error deleting DB: {e}")
            return

    print(f"Rebuilding SQLite DB from {dump_path} → {db_path} ...")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        with open(dump_path, "r", encoding="utf-8") as f:
            sql_script = f.read()
        cursor.executescript(sql_script)
        conn.commit()
        print("Rebuild completed.")
    except Exception as e:
        print(f"Error rebuilding DB: {e}")
    finally:
        conn.close()


def normalize_to_zulu(timestamp_str: str) -> str:
    """
    Normalize a timestamp string to Zulu (UTC) format with 'Z' suffix.
    Handles various input formats and converts to ISO 8601 with Zulu timezone.
    """
    if not timestamp_str:
        return ""

    # If already ends with Z, assume it's already in Zulu format
    if timestamp_str.endswith("Z"):
        return timestamp_str

    # Try to parse various datetime formats
    formats = [
        "%Y-%m-%d %H:%M:%S",  # SQLite default format
        "%Y-%m-%dT%H:%M:%S",  # ISO format without timezone
        "%Y-%m-%dT%H:%M:%S%z",  # ISO format with timezone
        "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO format with microseconds and timezone
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            # If timezone-aware, convert to UTC; otherwise assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            # Format as ISO 8601 with Zulu suffix
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    # If parsing fails, return as-is (might already be in correct format)
    return timestamp_str


def dump_database(format_type: str):
    """
    Dump all rows from the result table to stdout.

    Args:
        format_type: Either 'tsv' for tab-delimited or 'pretty-print' for aligned columns
    """
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}", file=sys.stderr)
        print("Use --create to create the database first.", file=sys.stderr)
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Get all rows sorted by date, home team, visitor
        cursor.execute(
            """
            SELECT * FROM result 
            ORDER BY date, home, visitor
        """
        )

        rows = cursor.fetchall()

        if not rows:
            # No data to dump
            return

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Prepare data rows with normalized timestamps
        data_rows = []
        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                if value is None:
                    values.append("")
                else:
                    # Normalize timestamp columns to Zulu format
                    if col in ("date", "created_at", "updated_at"):
                        values.append(normalize_to_zulu(str(value)))
                    else:
                        values.append(str(value))
            data_rows.append(values)

        if format_type == "tsv":
            # Tab-delimited format
            print("\t".join(columns))
            for row in data_rows:
                print("\t".join(row))
        elif format_type == "pretty-print":
            # Pretty-printed table using tabulate
            print(tabulate(data_rows, headers=columns, tablefmt="grid"))
        else:
            print(f"Error: Unknown format '{format_type}'", file=sys.stderr)
            sys.exit(1)

    except sqlite3.Error as e:
        print(f"Error dumping database: {e}", file=sys.stderr)
    finally:
        conn.close()


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Database utility script for managing the NFL results SQLite database.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Create the database and result table (if they don't exist)")
    group.add_argument("--truncate", action="store_true", help="Delete all data from the result table")
    group.add_argument("--dump", action="store_true", help="Dump all rows to stdout (sorted by date, home, visitor)")
    group.add_argument("--dump-sql", action="store_true", help="Dump database to db_dump.sql")
    group.add_argument("--restore-sql", action="store_true", help="Restore database from db_dump.sql")

    parser.add_argument(
        "--format",
        choices=["tsv", "pretty-print"],
        default="pretty-print",
        help="Output format for --dump: 'tsv' for tab-delimited, 'pretty-print' for aligned columns (default: pretty-print)",
    )

    args = parser.parse_args()

    if args.create:
        create_database()
    elif args.truncate:
        truncate_database()
    elif args.dump:
        dump_database(args.format)
    elif args.dump_sql:
        dump_sqlite_to_file(DB_PATH, Path("db_dump.sql"))
    elif args.restore_sql:
        rebuild_sqlite_from_dump(Path("db_dump.sql"), DB_PATH)


if __name__ == "__main__":
    main()
