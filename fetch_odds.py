"""
NFL Historical Odds Fetcher
Pulls game odds and player props from The Odds API for the 2023-2025 NFL seasons.
Stores everything in SQLite with resume capability.
"""

import requests
import sqlite3
import time
import os
import json
from datetime import datetime, timedelta, timezone

API_KEY = "29d902f2352064232e3d4022f78610b3"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")

# US bookmakers relevant for DFS
REGIONS = "us,us2"
ODDS_FORMAT = "american"

# Game odds markets
GAME_MARKETS = "h2h,spreads,totals"

# Player prop markets (key ones for DFS)
PROP_MARKETS = [
    "player_pass_tds", "player_pass_yds", "player_pass_attempts",
    "player_pass_completions", "player_pass_interceptions",
    "player_rush_yds", "player_rush_tds", "player_rush_attempts",
    "player_receptions", "player_reception_yds", "player_reception_tds",
    "player_anytime_td",
    "player_kicking_points", "player_field_goals",
    "player_pass_rush_yds", "player_rush_reception_yds",
]

# NFL season date ranges (regular season + playoffs)
SEASONS = {
    2023: {
        "start": "2023-09-05T00:00:00Z",
        "end": "2024-02-15T00:00:00Z",
    },
    2024: {
        "start": "2024-09-03T00:00:00Z",
        "end": "2025-02-15T00:00:00Z",
    },
    2025: {
        "start": "2025-09-02T00:00:00Z",
        "end": "2026-02-15T00:00:00Z",
    },
}

credits_used_session = 0


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def is_fetched(conn, fetch_type, fetch_key):
    row = conn.execute(
        "SELECT status FROM fetch_log WHERE fetch_type=? AND fetch_key=?",
        (fetch_type, fetch_key),
    ).fetchone()
    return row is not None and row[0] == "done"


def mark_fetched(conn, fetch_type, fetch_key, credits=0):
    conn.execute(
        """INSERT OR REPLACE INTO fetch_log (fetch_type, fetch_key, status, credits_used, timestamp)
           VALUES (?, ?, 'done', ?, ?)""",
        (fetch_type, fetch_key, credits, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def api_get(endpoint, params, retries=3):
    """Make API request with retry logic."""
    global credits_used_session
    params["apiKey"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 429:
                print("  Rate limited / quota exhausted. Waiting 60s...")
                time.sleep(60)
                continue

            if r.status_code == 422:
                # Usually means no data available for that date
                return None, 0

            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                return None, 0

            cost = int(r.headers.get("x-requests-last", 0))
            remaining = r.headers.get("x-requests-remaining", "?")
            credits_used_session += cost
            return r.json(), cost

        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return None, 0

    return None, 0


def determine_season(commence_time_str):
    """Determine NFL season year from game date. Games Sept-Feb belong to the Sept year's season."""
    dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    if dt.month >= 8:
        return dt.year
    else:
        return dt.year - 1


# ──────────────────────────────────────────────
# Phase 1: Discover all events
# ──────────────────────────────────────────────

def discover_events():
    """Use historical events endpoint to find all NFL games for each season."""
    conn = get_db()
    total_new = 0

    for season, dates in SEASONS.items():
        print(f"\n{'='*60}")
        print(f"Discovering events for {season} season...")
        print(f"{'='*60}")

        fetch_key = f"events_{season}"
        if is_fetched(conn, "discover", fetch_key):
            count = conn.execute(
                "SELECT COUNT(*) FROM games WHERE season=?", (season,)
            ).fetchone()[0]
            print(f"  Already discovered ({count} games). Skipping.")
            continue

        # Sample every 3 days through the season to catch all games
        start = datetime.fromisoformat(dates["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(dates["end"].replace("Z", "+00:00"))
        current = start
        all_events = {}

        while current < end:
            date_str = current.strftime("%Y-%m-%dT12:00:00Z")
            data, cost = api_get(
                f"historical/sports/{SPORT}/events",
                {"date": date_str, "dateFormat": "iso"},
            )

            if data and "data" in data:
                for event in data["data"]:
                    eid = event["id"]
                    if eid not in all_events:
                        all_events[eid] = event

            # Use next_timestamp if available for efficient pagination
            if data and data.get("next_timestamp"):
                next_ts = datetime.fromisoformat(
                    data["next_timestamp"].replace("Z", "+00:00")
                )
                # Jump ahead but at least 2 days to avoid too many calls
                jump = max(next_ts, current + timedelta(days=2))
                current = jump
            else:
                current += timedelta(days=3)

            time.sleep(0.3)

        # Insert events into database
        new_count = 0
        for eid, event in all_events.items():
            s = determine_season(event["commence_time"])
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO games (event_id, sport_key, home_team, away_team, commence_time, season)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (eid, event["sport_key"], event["home_team"],
                     event["away_team"], event["commence_time"], s),
                )
                new_count += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        mark_fetched(conn, "discover", fetch_key)
        total_new += new_count
        print(f"  Found {len(all_events)} events, inserted {new_count} new games")

    total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"\nTotal games in database: {total}")
    conn.close()
    return total


# ──────────────────────────────────────────────
# Phase 2: Fetch game odds (spreads, totals, ML)
# ──────────────────────────────────────────────

def fetch_game_odds():
    """Pull historical odds for all games. Uses bulk snapshot approach grouped by game day."""
    conn = get_db()

    # Get all games grouped by date
    games = conn.execute(
        "SELECT event_id, commence_time, home_team, away_team, season FROM games ORDER BY commence_time"
    ).fetchall()

    # Group games by date (to batch API calls)
    from collections import defaultdict
    daily_games = defaultdict(list)
    for g in games:
        game_date = g[1][:10]  # YYYY-MM-DD
        daily_games[game_date].append(g)

    print(f"\n{'='*60}")
    print(f"Fetching game odds for {len(games)} games across {len(daily_games)} game days...")
    print(f"{'='*60}")

    fetched_count = 0
    skipped_count = 0

    for game_date, day_games in sorted(daily_games.items()):
        fetch_key = f"game_odds_{game_date}"
        if is_fetched(conn, "game_odds", fetch_key):
            skipped_count += len(day_games)
            continue

        # Pick snapshot time: 2 hours before first game of the day
        first_game_time = min(g[1] for g in day_games)
        dt = datetime.fromisoformat(first_game_time.replace("Z", "+00:00"))
        snapshot_dt = dt - timedelta(hours=2)
        snapshot_str = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        season = day_games[0][4]
        teams = f"{day_games[0][3]} @ {day_games[0][2]}"
        extra = f" (+{len(day_games)-1} more)" if len(day_games) > 1 else ""
        print(f"  [{game_date}] {teams}{extra} -- snapshot: {snapshot_str[:16]}")

        data, cost = api_get(
            f"historical/sports/{SPORT}/odds",
            {
                "date": snapshot_str,
                "regions": REGIONS,
                "markets": GAME_MARKETS,
                "oddsFormat": ODDS_FORMAT,
                "dateFormat": "iso",
            },
        )

        if data and "data" in data:
            snapshot_time = data.get("timestamp", snapshot_str)
            event_ids_today = {g[0] for g in day_games}
            rows_inserted = 0

            for event in data["data"]:
                if event["id"] not in event_ids_today:
                    continue
                for bk in event.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        for outcome in mkt.get("outcomes", []):
                            conn.execute(
                                """INSERT INTO game_odds
                                   (event_id, bookmaker, market, outcome_name, price, point, snapshot_time, last_update)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    event["id"],
                                    bk["key"],
                                    mkt["key"],
                                    outcome["name"],
                                    outcome.get("price"),
                                    outcome.get("point"),
                                    snapshot_time,
                                    bk.get("last_update"),
                                ),
                            )
                            rows_inserted += 1

            conn.commit()
            fetched_count += len(day_games)
            if rows_inserted > 0:
                mark_fetched(conn, "game_odds", fetch_key, cost)

        time.sleep(0.3)

    total_rows = conn.execute("SELECT COUNT(*) FROM game_odds").fetchone()[0]
    print(f"\nGame odds: {fetched_count} new game-days fetched, {skipped_count} skipped (already done)")
    print(f"Total game_odds rows: {total_rows}")
    conn.close()


# ──────────────────────────────────────────────
# Phase 3: Fetch player props
# ──────────────────────────────────────────────

def fetch_player_props():
    """Pull historical player props using fresh event IDs from the historical events endpoint.

    The per-event historical endpoint requires event IDs that are valid at the queried timestamp.
    Our stored event IDs may have expired, so we re-discover events via the historical events
    endpoint, match them to our games by team names, then fetch props.
    """
    conn = get_db()
    from collections import defaultdict

    games = conn.execute(
        "SELECT event_id, commence_time, home_team, away_team, season FROM games ORDER BY commence_time"
    ).fetchall()

    # Group games by date
    daily_games = defaultdict(list)
    for g in games:
        game_date = g[1][:10]
        daily_games[game_date].append(g)

    print(f"\n{'='*60}")
    print(f"Fetching player props for {len(games)} games across {len(daily_games)} game days...")
    print(f"{'='*60}")

    # Process props in batches of markets
    prop_batches = []
    batch_size = 5
    for i in range(0, len(PROP_MARKETS), batch_size):
        prop_batches.append(PROP_MARKETS[i:i + batch_size])

    fetched_count = 0
    skipped_count = 0
    total_game_idx = 0

    for game_date, day_games in sorted(daily_games.items()):
        # Check if all games on this day are already fetched
        unfetched = []
        for g in day_games:
            # Use home+away+date as fetch key since event_id changes
            fkey = f"props_{g[2]}_{g[3]}_{game_date}"
            if is_fetched(conn, "player_props", fkey):
                skipped_count += 1
            else:
                unfetched.append(g)

        total_game_idx += len(day_games)
        if not unfetched:
            continue

        # Get fresh event IDs from historical events endpoint
        first_game_time = min(g[1] for g in day_games)
        dt = datetime.fromisoformat(first_game_time.replace("Z", "+00:00"))
        snapshot_dt = dt - timedelta(hours=2)
        snapshot_str = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        events_data, _ = api_get(
            f"historical/sports/{SPORT}/events",
            {"date": snapshot_str, "dateFormat": "iso"},
        )

        if not events_data or "data" not in events_data:
            print(f"  [{game_date}] Could not discover events, skipping...")
            continue

        # Build lookup: (home_team, away_team) -> historical event ID
        hist_events = {}
        for ev in events_data["data"]:
            key = (ev["home_team"], ev["away_team"])
            hist_events[key] = ev["id"]

        time.sleep(0.2)

        # Now fetch props for each unfetched game using the fresh event ID
        for g in unfetched:
            our_event_id, commence_time, home, away, season = g
            fkey = f"props_{home}_{away}_{game_date}"

            hist_eid = hist_events.get((home, away))
            if not hist_eid:
                print(f"  [{game_date}] {away} @ {home} -- no matching historical event")
                mark_fetched(conn, "player_props", fkey, 0)
                fetched_count += 1
                continue

            print(f"  [{game_date}] {away} @ {home} (eid: {hist_eid[:12]}...)")

            total_rows = 0
            total_cost = 0

            for batch in prop_batches:
                markets_str = ",".join(batch)
                data, cost = api_get(
                    f"historical/sports/{SPORT}/events/{hist_eid}/odds",
                    {
                        "date": snapshot_str,
                        "regions": REGIONS,
                        "markets": markets_str,
                        "oddsFormat": ODDS_FORMAT,
                        "dateFormat": "iso",
                    },
                )
                total_cost += cost

                if data and "data" in data:
                    event_data = data["data"]
                    snap_time = data.get("timestamp", snapshot_str)

                    for bk in event_data.get("bookmakers", []):
                        for mkt in bk.get("markets", []):
                            for outcome in mkt.get("outcomes", []):
                                player_name = outcome.get("description", outcome.get("name", ""))
                                outcome_type = outcome.get("name", "")

                                conn.execute(
                                    """INSERT INTO player_props
                                       (event_id, bookmaker, market, player_name, outcome_type, price, point, snapshot_time, last_update)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        our_event_id,  # Store with OUR event_id for FK consistency
                                        bk["key"],
                                        mkt["key"],
                                        player_name,
                                        outcome_type,
                                        outcome.get("price"),
                                        outcome.get("point"),
                                        snap_time,
                                        bk.get("last_update"),
                                    ),
                                )
                                total_rows += 1

                time.sleep(0.25)

            conn.commit()
            mark_fetched(conn, "player_props", fkey, total_cost)
            fetched_count += 1

            if total_rows > 0:
                print(f"    -> {total_rows} prop lines ({total_cost} credits)")
            else:
                print(f"    -> No prop data ({total_cost} credits)")

    total_rows = conn.execute("SELECT COUNT(*) FROM player_props").fetchone()[0]
    print(f"\nPlayer props: {fetched_count} new events fetched, {skipped_count} skipped")
    print(f"Total player_props rows: {total_rows}")
    conn.close()


# ──────────────────────────────────────────────
# Phase 4: Fetch scores
# ──────────────────────────────────────────────

def fetch_scores():
    """Pull final scores for all games using historical scores endpoint."""
    conn = get_db()

    games = conn.execute(
        "SELECT event_id, commence_time, home_team, away_team FROM games WHERE completed=0 ORDER BY commence_time"
    ).fetchall()

    if not games:
        print("\nAll games already have scores.")
        conn.close()
        return

    print(f"\n{'='*60}")
    print(f"Fetching scores for {len(games)} games...")
    print(f"{'='*60}")

    # Group by date and fetch scores in bulk
    from collections import defaultdict
    daily = defaultdict(list)
    for g in games:
        daily[g[1][:10]].append(g)

    updated = 0
    for game_date, day_games in sorted(daily.items()):
        # Use a timestamp 1 day after the games
        dt = datetime.fromisoformat(f"{game_date}T00:00:00+00:00") + timedelta(days=1, hours=12)
        date_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        data, cost = api_get(
            f"historical/sports/{SPORT}/scores",
            {"date": date_str, "dateFormat": "iso"},
        )

        if data and "data" in data:
            event_ids = {g[0] for g in day_games}
            for event in data["data"]:
                if event["id"] in event_ids and event.get("completed"):
                    scores = event.get("scores", [])
                    home_score = away_score = None
                    home_team = event.get("home_team")
                    for s in scores:
                        if s["name"] == home_team:
                            home_score = int(s["score"]) if s["score"] else None
                        else:
                            away_score = int(s["score"]) if s["score"] else None

                    if home_score is not None:
                        conn.execute(
                            "UPDATE games SET home_score=?, away_score=?, completed=1 WHERE event_id=?",
                            (home_score, away_score, event["id"]),
                        )
                        updated += 1

        conn.commit()
        time.sleep(0.3)

    print(f"  Updated scores for {updated} games")
    conn.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def print_summary():
    conn = get_db()
    games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    odds = conn.execute("SELECT COUNT(*) FROM game_odds").fetchone()[0]
    props = conn.execute("SELECT COUNT(*) FROM player_props").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM games WHERE completed=1").fetchone()[0]

    print(f"\n{'='*60}")
    print(f"DATABASE SUMMARY")
    print(f"{'='*60}")
    print(f"  Games:          {games:,}")
    print(f"  Game odds rows: {odds:,}")
    print(f"  Player props:   {props:,}")
    print(f"  Games w/ scores:{scored:,}")
    print(f"  Session credits:{credits_used_session:,}")

    # Per-season breakdown
    for season in [2023, 2024, 2025]:
        sg = conn.execute("SELECT COUNT(*) FROM games WHERE season=?", (season,)).fetchone()[0]
        so = conn.execute(
            "SELECT COUNT(*) FROM game_odds WHERE event_id IN (SELECT event_id FROM games WHERE season=?)",
            (season,),
        ).fetchone()[0]
        sp = conn.execute(
            "SELECT COUNT(*) FROM player_props WHERE event_id IN (SELECT event_id FROM games WHERE season=?)",
            (season,),
        ).fetchone()[0]
        print(f"\n  {season} Season: {sg} games, {so:,} odds rows, {sp:,} prop rows")

    conn.close()


if __name__ == "__main__":
    print("NFL Historical Odds Fetcher")
    print(f"Database: {DB_PATH}")
    print(f"Seasons: 2023, 2024, 2025")

    # Phase 1: Discover events
    discover_events()

    # Phase 2: Game odds
    fetch_game_odds()

    # Phase 3: Player props
    fetch_player_props()

    # Phase 4: Scores
    fetch_scores()

    # Summary
    print_summary()
