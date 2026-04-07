"""
NFL Player Stats Fetcher
Pulls individual player game stats from nflverse via nflreadpy
and loads them into the nfl_odds.db SQLite database.

Joins with existing games table using schedule data to map
team abbreviations + week to the odds API event IDs.
"""

import sqlite3
import os
import nflreadpy as nflr
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")
SEASONS = [2023, 2024, 2025]

# Map full team names (odds API) to nflverse abbreviations
TEAM_ABBREV = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

# Columns to store in the database (nflreadpy names -> DB names)
# nflreadpy renamed some columns vs nfl_data_py
COLUMN_MAP = {
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
    "team": "team",  # nflreadpy uses 'team' instead of 'recent_team'
}

# All stat columns we want in the DB
STAT_COLUMNS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "sack_yards", "passing_air_yards", "passing_yards_after_catch",
    "passing_first_downs", "passing_epa", "passing_2pt_conversions", "pacr", "dakota",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share", "wopr",
    "special_teams_tds", "fantasy_points", "fantasy_points_ppr",
]


def create_tables(conn):
    """Create player_stats table."""
    conn.executescript("""
        DROP TABLE IF EXISTS player_stats;

        CREATE TABLE player_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            game_id TEXT,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_display_name TEXT,
            position TEXT,
            team TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            opponent TEXT,

            -- Passing
            completions INTEGER,
            attempts INTEGER,
            passing_yards REAL,
            passing_tds INTEGER,
            interceptions INTEGER,
            sacks INTEGER,
            sack_yards REAL,
            passing_air_yards REAL,
            passing_yards_after_catch REAL,
            passing_first_downs INTEGER,
            passing_epa REAL,
            passing_2pt_conversions INTEGER,
            pacr REAL,
            dakota REAL,

            -- Rushing
            carries INTEGER,
            rushing_yards REAL,
            rushing_tds INTEGER,
            rushing_fumbles INTEGER,
            rushing_fumbles_lost INTEGER,
            rushing_first_downs INTEGER,
            rushing_epa REAL,
            rushing_2pt_conversions INTEGER,

            -- Receiving
            receptions INTEGER,
            targets INTEGER,
            receiving_yards REAL,
            receiving_tds INTEGER,
            receiving_fumbles INTEGER,
            receiving_fumbles_lost INTEGER,
            receiving_air_yards REAL,
            receiving_yards_after_catch REAL,
            receiving_first_downs INTEGER,
            receiving_epa REAL,
            receiving_2pt_conversions INTEGER,
            racr REAL,
            target_share REAL,
            air_yards_share REAL,
            wopr REAL,

            -- Special teams & fantasy
            special_teams_tds INTEGER,
            fantasy_points REAL,
            fantasy_points_ppr REAL,

            FOREIGN KEY (event_id) REFERENCES games(event_id)
        );

        CREATE INDEX idx_player_stats_event ON player_stats(event_id);
        CREATE INDEX idx_player_stats_player ON player_stats(player_name);
        CREATE INDEX idx_player_stats_team_week ON player_stats(team, season, week);
        CREATE INDEX idx_player_stats_game_id ON player_stats(game_id);
        CREATE INDEX idx_player_stats_position ON player_stats(position);
    """)


def build_game_mapping(conn):
    """Build a mapping from (season, week, home_abbrev, away_abbrev) -> event_id
    using the nflverse schedule to bridge dates/weeks with the odds DB games."""

    print("Building game mapping between odds DB and nflverse...")

    # Load schedules (nflreadpy returns polars, convert to pandas)
    schedules = nflr.load_schedules(SEASONS).to_pandas()

    # Load games from DB
    games = pd.read_sql_query(
        "SELECT event_id, home_team, away_team, season, commence_time FROM games", conn
    )

    # Add abbreviations to DB games
    games["home_abbrev"] = games["home_team"].map(TEAM_ABBREV)
    games["away_abbrev"] = games["away_team"].map(TEAM_ABBREV)
    games["game_date"] = pd.to_datetime(games["commence_time"]).dt.date

    # For duplicate event_ids (same game, different odds snapshots), keep the first
    games = games.drop_duplicates(subset=["home_abbrev", "away_abbrev", "game_date"], keep="first")

    # Build mapping using date + teams
    mapping = {}

    for _, sched_row in schedules.iterrows():
        season = sched_row["season"]
        week = sched_row["week"]
        home = sched_row["home_team"]
        away = sched_row["away_team"]
        gameday = pd.to_datetime(sched_row["gameday"]).date()

        # Find matching game in odds DB (match on teams + date within 1 day for timezone issues)
        match = games[
            (games["season"] == season)
            & (games["home_abbrev"] == home)
            & (games["away_abbrev"] == away)
            & (abs((games["game_date"] - gameday).dt.days) <= 1)
        ]

        if len(match) >= 1:
            mapping[(season, week, home, away)] = match.iloc[0]["event_id"]

    print(f"  Mapped {len(mapping)} games out of {len(schedules)} schedule entries")
    return mapping, schedules


def fetch_and_load_stats(conn):
    """Fetch weekly player stats and load into database."""

    # Build game mapping
    mapping, schedules = build_game_mapping(conn)

    # Fetch player stats from nflreadpy (returns polars DataFrame)
    print(f"\nFetching player weekly stats for seasons {SEASONS}...")
    stats_pl = nflr.load_player_stats(seasons=SEASONS)

    # Convert to pandas and rename columns to match our DB schema
    stats = stats_pl.to_pandas()
    stats = stats.rename(columns={
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
    })

    # nflreadpy uses 'team' where nfl_data_py used 'recent_team'
    team_col = "team" if "team" in stats.columns else "recent_team"

    print(f"  Got {len(stats)} player-game rows for {stats['player_id'].nunique()} unique players")

    # Map event_ids onto stats
    def get_event_id(row):
        team = row[team_col]
        opp = row["opponent_team"]
        season = row["season"]
        week = row["week"]
        eid = mapping.get((season, week, team, opp))
        if eid is None:
            eid = mapping.get((season, week, opp, team))
        return eid

    stats["event_id"] = stats.apply(get_event_id, axis=1)

    # Drop rows without a player_id
    before = len(stats)
    stats = stats.dropna(subset=["player_id", "player_name"])
    if before != len(stats):
        print(f"  Dropped {before - len(stats)} rows with null player_id/name")

    matched = stats["event_id"].notna().sum()
    total = len(stats)
    print(f"  Matched {matched}/{total} stat rows to odds DB games ({matched/total*100:.1f}%)")

    # Use game_id from nflreadpy if available, otherwise build it
    if "game_id" not in stats.columns:
        stats["game_id"] = (
            stats["season"].astype(str) + "_"
            + stats["week"].astype(str).str.zfill(2) + "_"
            + stats[team_col] + "_"
            + stats["opponent_team"]
        )

    # Prepare batch insert
    all_db_cols = [
        "event_id", "game_id", "player_id", "player_name", "player_display_name",
        "position", "team", "season", "week", "opponent",
    ] + STAT_COLUMNS

    rows = []
    for _, row in stats.iterrows():
        values = {
            "event_id": row.get("event_id"),
            "game_id": row.get("game_id"),
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "player_display_name": row.get("player_display_name"),
            "position": row.get("position"),
            "team": row[team_col],
            "season": int(row["season"]),
            "week": int(row["week"]),
            "opponent": row.get("opponent_team"),
        }
        for col in STAT_COLUMNS:
            val = row.get(col)
            values[col] = None if pd.isna(val) else val

        rows.append([values[c] for c in all_db_cols])

    placeholders = ", ".join(["?"] * len(all_db_cols))
    col_names = ", ".join(all_db_cols)

    conn.executemany(
        f"INSERT INTO player_stats ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    print(f"\nInserted {len(rows)} player stat rows into database.")

    # Backfill week numbers and scores into games table
    print("\nBackfilling week numbers and scores into games table...")
    updated = 0
    for _, sched_row in schedules.iterrows():
        season = sched_row["season"]
        week = sched_row["week"]
        home = sched_row["home_team"]
        away = sched_row["away_team"]
        home_score = sched_row.get("home_score")
        away_score = sched_row.get("away_score")

        eid = mapping.get((season, week, home, away))
        if eid and pd.notna(home_score) and pd.notna(away_score):
            conn.execute(
                "UPDATE games SET week = ?, home_score = ?, away_score = ?, completed = 1 WHERE event_id = ?",
                (int(week), int(home_score), int(away_score), eid),
            )
            updated += 1

    conn.commit()
    print(f"  Updated {updated} games with week numbers and final scores.")


def print_summary(conn):
    """Print summary stats."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    for season in SEASONS:
        total = conn.execute(
            "SELECT COUNT(DISTINCT player_id) FROM player_stats WHERE season = ?", (season,)
        ).fetchone()[0]
        games_matched = conn.execute(
            "SELECT COUNT(DISTINCT event_id) FROM player_stats WHERE season = ? AND event_id IS NOT NULL",
            (season,),
        ).fetchone()[0]
        total_games = conn.execute(
            "SELECT COUNT(*) FROM games WHERE season = ?", (season,)
        ).fetchone()[0]
        stat_rows = conn.execute(
            "SELECT COUNT(*) FROM player_stats WHERE season = ?", (season,)
        ).fetchone()[0]
        print(f"\n  Season {season}:")
        print(f"    Players: {total}")
        print(f"    Stat rows: {stat_rows}")
        print(f"    Games matched to odds: {games_matched}/{total_games}")

    # Top fantasy scorers
    print("\n  Top 10 fantasy scorers (PPR, single game):")
    rows = conn.execute("""
        SELECT player_display_name, position, team, season, week, fantasy_points_ppr
        FROM player_stats
        ORDER BY fantasy_points_ppr DESC
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:25s} {r[1]:3s} {r[2]:3s} {r[3]} Wk{r[4]:2d}  {r[5]:.1f} pts")

    # Sample prop vs actual
    print("\n  Sample prop vs actual (passing yards, DraftKings):")
    rows = conn.execute("""
        SELECT pp.player_name, pp.point, ps.passing_yards,
               g.home_team || ' vs ' || g.away_team, g.season, g.week
        FROM player_props pp
        JOIN player_stats ps ON ps.event_id = pp.event_id
            AND LOWER(ps.player_display_name) = LOWER(pp.player_name)
        JOIN games g ON g.event_id = pp.event_id
        WHERE pp.market = 'player_pass_yds' AND pp.outcome_type = 'Over'
        AND pp.bookmaker = 'draftkings'
        ORDER BY g.season DESC, g.week DESC
        LIMIT 5
    """).fetchall()
    for r in rows:
        hit = "OVER" if r[2] and r[2] > r[1] else "UNDER"
        print(f"    {r[0]:25s} Line: {r[1]:5.1f} | Actual: {r[2]:6.1f} | {hit:5s} | {r[4]} Wk{r[5]}")


def main():
    print("NFL Player Stats Loader")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    fetch_and_load_stats(conn)
    print_summary(conn)
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
