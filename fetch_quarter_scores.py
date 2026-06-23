"""
NFL Quarter-by-Quarter Scoring
Derives per-quarter scoring for every game from nflverse play-by-play data.
Stores results in nfl_odds.db with links to existing games table.
"""

import sqlite3
import os
import sys
import pandas as pd
import numpy as np
import nflreadpy as nflr

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "nfl_odds.db")
SEASONS = list(range(2012, 2026))

TEAM_ABBREV = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def create_table(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS quarter_scores;

        CREATE TABLE quarter_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            quarter INTEGER NOT NULL,
            home_qtr_pts INTEGER NOT NULL,
            away_qtr_pts INTEGER NOT NULL,
            home_cum_pts INTEGER NOT NULL,
            away_cum_pts INTEGER NOT NULL,
            UNIQUE(game_id, quarter)
        );

        CREATE INDEX idx_qtr_event ON quarter_scores(event_id);
        CREATE INDEX idx_qtr_game ON quarter_scores(game_id);
        CREATE INDEX idx_qtr_season_week ON quarter_scores(season, week);
        CREATE INDEX idx_qtr_quarter ON quarter_scores(quarter);
        CREATE INDEX idx_qtr_teams ON quarter_scores(home_team, away_team);
    """)


def build_event_mapping(conn):
    """Map nflverse game_id -> odds DB event_id using schedule + games table."""
    games = pd.read_sql_query(
        "SELECT event_id, home_team, away_team, season, commence_time FROM games", conn
    )
    games["home_abbrev"] = games["home_team"].map(TEAM_ABBREV)
    games["away_abbrev"] = games["away_team"].map(TEAM_ABBREV)
    games["game_date"] = pd.to_datetime(games["commence_time"]).dt.date
    games = games.drop_duplicates(subset=["home_abbrev", "away_abbrev", "game_date"], keep="first")

    schedules = nflr.load_schedules(SEASONS).to_pandas()

    mapping = {}
    for _, row in schedules.iterrows():
        gid = row["game_id"]
        home = row["home_team"]
        away = row["away_team"]
        season = row["season"]
        gameday = pd.to_datetime(row["gameday"]).date()

        match = games[
            (games["season"] == season)
            & (games["home_abbrev"] == home)
            & (games["away_abbrev"] == away)
            & (abs((games["game_date"] - gameday).dt.days) <= 1)
        ]
        if len(match) >= 1:
            mapping[gid] = match.iloc[0]["event_id"]

    return mapping


def extract_quarter_scores(season):
    """Extract per-quarter scoring from play-by-play for one season."""
    print(f"  Loading PBP for {season}...")
    pbp = nflr.load_pbp(seasons=[season]).to_pandas()

    # Get schedule for week numbers
    sched = nflr.load_schedules(seasons=[season]).to_pandas()
    game_weeks = sched.set_index("game_id")["week"].to_dict()

    # Filter to plays with score data
    scored = pbp.dropna(subset=["total_home_score", "total_away_score", "qtr"]).copy()
    scored["qtr"] = scored["qtr"].astype(int)
    # Keep Q1-Q4 + OT (5)
    scored = scored[scored["qtr"] <= 5]

    # Cumulative score at end of each quarter = max score seen in that quarter
    end_of_qtr = scored.groupby(["game_id", "home_team", "away_team", "qtr"]).agg(
        home_cum=("total_home_score", "max"),
        away_cum=("total_away_score", "max"),
    ).reset_index()

    # Calculate per-quarter points
    rows = []
    for game_id, gdf in end_of_qtr.groupby("game_id"):
        gdf = gdf.sort_values("qtr")
        home_prev, away_prev = 0, 0
        week = game_weeks.get(game_id, 0)

        for _, r in gdf.iterrows():
            home_qtr = int(r["home_cum"] - home_prev)
            away_qtr = int(r["away_cum"] - away_prev)
            rows.append({
                "game_id": game_id,
                "season": season,
                "week": int(week),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "quarter": int(r["qtr"]),
                "home_qtr_pts": home_qtr,
                "away_qtr_pts": away_qtr,
                "home_cum_pts": int(r["home_cum"]),
                "away_cum_pts": int(r["away_cum"]),
            })
            home_prev = r["home_cum"]
            away_prev = r["away_cum"]

    print(f"    {len(rows)} quarter records from {end_of_qtr['game_id'].nunique()} games")
    return pd.DataFrame(rows)


def main():
    print("NFL Quarter-by-Quarter Scoring Loader")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Build event_id mapping
    print("\nBuilding event_id mapping...")
    event_map = build_event_mapping(conn)
    print(f"  Mapped {len(event_map)} games to odds DB")

    # Extract quarter scores for each season
    all_quarters = []
    for season in SEASONS:
        qdf = extract_quarter_scores(season)
        all_quarters.append(qdf)

    combined = pd.concat(all_quarters, ignore_index=True)
    combined["event_id"] = combined["game_id"].map(event_map)

    matched = combined["event_id"].notna().sum()
    print(f"\n  {matched}/{len(combined)} quarter rows linked to odds DB")

    # Insert into DB
    print("\nInserting into database...")
    rows = []
    for _, r in combined.iterrows():
        rows.append((
            r.get("event_id"), r["game_id"], int(r["season"]), int(r["week"]),
            r["home_team"], r["away_team"], int(r["quarter"]),
            int(r["home_qtr_pts"]), int(r["away_qtr_pts"]),
            int(r["home_cum_pts"]), int(r["away_cum_pts"]),
        ))

    conn.executemany("""
        INSERT INTO quarter_scores
        (event_id, game_id, season, week, home_team, away_team, quarter,
         home_qtr_pts, away_qtr_pts, home_cum_pts, away_cum_pts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    # Summary
    print(f"\nInserted {len(rows)} quarter-score rows.")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for season in SEASONS:
        games = conn.execute(
            "SELECT COUNT(DISTINCT game_id) FROM quarter_scores WHERE season = ?", (season,)
        ).fetchone()[0]
        print(f"\n  {season}: {games} games")

        avgs = conn.execute("""
            SELECT quarter,
                   ROUND(AVG(home_qtr_pts + away_qtr_pts), 1) as avg_combined,
                   ROUND(AVG(home_qtr_pts), 1) as avg_home,
                   ROUND(AVG(away_qtr_pts), 1) as avg_away
            FROM quarter_scores
            WHERE season = ? AND quarter <= 4
            GROUP BY quarter ORDER BY quarter
        """, (season,)).fetchall()
        print(f"    {'Qtr':>4s}  {'Combined':>8s}  {'Home':>6s}  {'Away':>6s}")
        for a in avgs:
            print(f"    Q{a[0]}    {a[1]:>7.1f}   {a[2]:>5.1f}   {a[3]:>5.1f}")

        ot = conn.execute(
            "SELECT COUNT(DISTINCT game_id) FROM quarter_scores WHERE season = ? AND quarter = 5",
            (season,)
        ).fetchone()[0]
        if ot:
            print(f"    OT games: {ot}")

    # Fun stat: biggest comeback by quarter
    print("\n  Biggest halftime deficits overcome (win):")
    comebacks = conn.execute("""
        SELECT q2.game_id, q2.home_team, q2.away_team, q2.season, q2.week,
               q2.home_cum_pts as home_half, q2.away_cum_pts as away_half,
               qf.home_cum_pts as home_final, qf.away_cum_pts as away_final,
               CASE
                   WHEN qf.home_cum_pts > qf.away_cum_pts THEN q2.away_cum_pts - q2.home_cum_pts
                   ELSE q2.home_cum_pts - q2.away_cum_pts
               END as deficit_overcome
        FROM quarter_scores q2
        JOIN quarter_scores qf ON qf.game_id = q2.game_id
        WHERE q2.quarter = 2
        AND qf.quarter = (SELECT MAX(quarter) FROM quarter_scores WHERE game_id = q2.game_id)
        AND (
            (qf.home_cum_pts > qf.away_cum_pts AND q2.home_cum_pts < q2.away_cum_pts)
            OR (qf.away_cum_pts > qf.home_cum_pts AND q2.away_cum_pts < q2.home_cum_pts)
        )
        ORDER BY deficit_overcome DESC
        LIMIT 5
    """).fetchall()
    for r in comebacks:
        winner = r[1] if r[7] > r[8] else r[2]
        print(f"    {r[2]:3s} @ {r[1]:3s} ({r[3]} Wk{r[4]:2d}) — {winner} trailed {r[5]}-{r[6]} at half, won {r[7]}-{r[8]} (overcame {r[9]} pt deficit)")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
