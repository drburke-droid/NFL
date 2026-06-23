"""
NFL Odds Database Schema Setup
Creates SQLite database with tables for games, game odds, player props, and scores.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "nfl_odds.db")


def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Games table - one row per NFL game
    c.execute("""
        CREATE TABLE IF NOT EXISTS games (
            event_id TEXT PRIMARY KEY,
            sport_key TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            season INTEGER NOT NULL,
            week TEXT,
            home_score INTEGER,
            away_score INTEGER,
            completed INTEGER DEFAULT 0
        )
    """)

    # Game odds - moneyline (h2h), spreads, totals
    c.execute("""
        CREATE TABLE IF NOT EXISTS game_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            outcome_name TEXT NOT NULL,
            price REAL NOT NULL,
            point REAL,
            snapshot_time TEXT NOT NULL,
            last_update TEXT,
            FOREIGN KEY (event_id) REFERENCES games(event_id)
        )
    """)

    # Player props odds
    c.execute("""
        CREATE TABLE IF NOT EXISTS player_props (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            player_name TEXT NOT NULL,
            outcome_type TEXT NOT NULL,
            price REAL NOT NULL,
            point REAL,
            snapshot_time TEXT NOT NULL,
            last_update TEXT,
            FOREIGN KEY (event_id) REFERENCES games(event_id)
        )
    """)

    # Fetch log - tracks what we've already fetched for resume capability
    c.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_type TEXT NOT NULL,
            fetch_key TEXT NOT NULL,
            status TEXT NOT NULL,
            credits_used INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL,
            UNIQUE(fetch_type, fetch_key)
        )
    """)

    # Indexes for fast queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_game_odds_event ON game_odds(event_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_game_odds_market ON game_odds(market)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_game_odds_bookmaker ON game_odds(bookmaker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_player_props_event ON player_props(event_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_player_props_market ON player_props(market)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_player_props_player ON player_props(player_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_player_props_bookmaker ON player_props(bookmaker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_games_commence ON games(commence_time)")

    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")


if __name__ == "__main__":
    create_database()
