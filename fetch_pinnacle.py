"""
Fetch Pinnacle odds (game lines + player props) for all NFL games.
Pinnacle is in the 'eu' region. This adds Pinnacle data to the existing database.
"""

import requests
import sqlite3
import time
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict

API_KEY = "29d902f2352064232e3d4022f78610b3"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")

REGIONS = "eu"
ODDS_FORMAT = "american"
GAME_MARKETS = "h2h,spreads,totals"

PROP_MARKETS = [
    "player_pass_tds", "player_pass_yds", "player_pass_attempts",
    "player_pass_completions", "player_pass_interceptions",
    "player_rush_yds", "player_rush_tds", "player_rush_attempts",
    "player_receptions", "player_reception_yds", "player_reception_tds",
    "player_anytime_td",
    "player_kicking_points", "player_field_goals",
    "player_pass_rush_yds", "player_rush_reception_yds",
]


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
    params["apiKey"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                print("  Rate limited. Waiting 60s...")
                time.sleep(60)
                continue
            if r.status_code in (404, 422):
                return None, 0
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                return None, 0

            cost = int(r.headers.get("x-requests-last", 0))
            return r.json(), cost

        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return None, 0

    return None, 0


def fetch_pinnacle_game_odds():
    """Fetch Pinnacle game odds for all game days."""
    conn = get_db()

    games = conn.execute(
        "SELECT event_id, commence_time, home_team, away_team, season FROM games ORDER BY commence_time"
    ).fetchall()

    daily_games = defaultdict(list)
    for g in games:
        daily_games[g[1][:10]].append(g)

    print(f"\n{'='*60}")
    print(f"Fetching Pinnacle game odds for {len(daily_games)} game days...")
    print(f"{'='*60}")

    fetched = 0
    skipped = 0

    for game_date, day_games in sorted(daily_games.items()):
        fetch_key = f"pinnacle_odds_{game_date}"
        if is_fetched(conn, "pinnacle_odds", fetch_key):
            skipped += 1
            continue

        first_game_time = min(g[1] for g in day_games)
        dt = datetime.fromisoformat(first_game_time.replace("Z", "+00:00"))
        snapshot_dt = dt - timedelta(hours=2)
        snapshot_str = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        teams = f"{day_games[0][3]} @ {day_games[0][2]}"
        extra = f" (+{len(day_games)-1} more)" if len(day_games) > 1 else ""
        print(f"  [{game_date}] {teams}{extra}")

        data, cost = api_get(
            f"historical/sports/{SPORT}/odds",
            {
                "date": snapshot_str,
                "regions": REGIONS,
                "markets": GAME_MARKETS,
                "oddsFormat": ODDS_FORMAT,
                "dateFormat": "iso",
                "bookmakers": "pinnacle",
            },
        )

        if data and "data" in data:
            snapshot_time = data.get("timestamp", snapshot_str)
            event_ids_today = {g[0] for g in day_games}
            rows = 0

            for event in data["data"]:
                if event["id"] not in event_ids_today:
                    continue
                for bk in event.get("bookmakers", []):
                    if bk["key"] != "pinnacle":
                        continue
                    for mkt in bk.get("markets", []):
                        for outcome in mkt.get("outcomes", []):
                            conn.execute(
                                """INSERT INTO game_odds
                                   (event_id, bookmaker, market, outcome_name, price, point, snapshot_time, last_update)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (event["id"], bk["key"], mkt["key"],
                                 outcome["name"], outcome.get("price"),
                                 outcome.get("point"), snapshot_time,
                                 bk.get("last_update")),
                            )
                            rows += 1

            conn.commit()
            mark_fetched(conn, "pinnacle_odds", fetch_key, cost)
            fetched += 1

        time.sleep(0.3)

    pin_total = conn.execute("SELECT COUNT(*) FROM game_odds WHERE bookmaker='pinnacle'").fetchone()[0]
    print(f"\nPinnacle game odds: {fetched} days fetched, {skipped} skipped")
    print(f"Total Pinnacle game odds rows: {pin_total:,}")
    conn.close()


def fetch_pinnacle_props():
    """Fetch Pinnacle player props for all games using historical event IDs."""
    conn = get_db()

    games = conn.execute(
        "SELECT event_id, commence_time, home_team, away_team, season FROM games ORDER BY commence_time"
    ).fetchall()

    daily_games = defaultdict(list)
    for g in games:
        daily_games[g[1][:10]].append(g)

    prop_batches = []
    batch_size = 5
    for i in range(0, len(PROP_MARKETS), batch_size):
        prop_batches.append(PROP_MARKETS[i:i + batch_size])

    print(f"\n{'='*60}")
    print(f"Fetching Pinnacle player props for {len(games)} games...")
    print(f"{'='*60}")

    fetched = 0
    skipped = 0

    for game_date, day_games in sorted(daily_games.items()):
        # Check which games on this day need fetching
        unfetched = []
        for g in day_games:
            fkey = f"pin_props_{g[2]}_{g[3]}_{game_date}"
            if is_fetched(conn, "pinnacle_props", fkey):
                skipped += 1
            else:
                unfetched.append(g)

        if not unfetched:
            continue

        # Get fresh historical event IDs
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

        hist_events = {}
        for ev in events_data["data"]:
            hist_events[(ev["home_team"], ev["away_team"])] = ev["id"]

        time.sleep(0.2)

        for g in unfetched:
            our_event_id, commence_time, home, away, season = g
            fkey = f"pin_props_{home}_{away}_{game_date}"

            hist_eid = hist_events.get((home, away))
            if not hist_eid:
                mark_fetched(conn, "pinnacle_props", fkey, 0)
                fetched += 1
                continue

            print(f"  [{game_date}] {away} @ {home}")

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
                        "bookmakers": "pinnacle",
                    },
                )
                total_cost += cost

                if data and "data" in data:
                    event_data = data["data"]
                    snap_time = data.get("timestamp", snapshot_str)

                    for bk in event_data.get("bookmakers", []):
                        if bk["key"] != "pinnacle":
                            continue
                        for mkt in bk.get("markets", []):
                            for outcome in mkt.get("outcomes", []):
                                player_name = outcome.get("description", outcome.get("name", ""))
                                outcome_type = outcome.get("name", "")

                                conn.execute(
                                    """INSERT INTO player_props
                                       (event_id, bookmaker, market, player_name, outcome_type, price, point, snapshot_time, last_update)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (our_event_id, bk["key"], mkt["key"],
                                     player_name, outcome_type,
                                     outcome.get("price"), outcome.get("point"),
                                     snap_time, bk.get("last_update")),
                                )
                                total_rows += 1

                time.sleep(0.25)

            conn.commit()
            mark_fetched(conn, "pinnacle_props", fkey, total_cost)
            fetched += 1

            if total_rows > 0:
                print(f"    -> {total_rows} Pinnacle prop lines ({total_cost} credits)")
            else:
                print(f"    -> No Pinnacle props ({total_cost} credits)")

    pin_props = conn.execute("SELECT COUNT(*) FROM player_props WHERE bookmaker='pinnacle'").fetchone()[0]
    print(f"\nPinnacle props: {fetched} events fetched, {skipped} skipped")
    print(f"Total Pinnacle player prop rows: {pin_props:,}")
    conn.close()


if __name__ == "__main__":
    print("Fetching Pinnacle odds for all NFL games...")

    # Phase 1: Game odds
    fetch_pinnacle_game_odds()

    # Phase 2: Player props
    fetch_pinnacle_props()

    # Summary
    conn = sqlite3.connect(DB_PATH)
    go = conn.execute("SELECT COUNT(*) FROM game_odds WHERE bookmaker='pinnacle'").fetchone()[0]
    pp = conn.execute("SELECT COUNT(*) FROM player_props WHERE bookmaker='pinnacle'").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"PINNACLE SUMMARY")
    print(f"{'='*60}")
    print(f"  Game odds rows:   {go:,}")
    print(f"  Player prop rows: {pp:,}")

    r = requests.get("https://api.the-odds-api.com/v4/sports", params={"apiKey": API_KEY})
    print(f"  Credits remaining: {r.headers.get('x-requests-remaining')}")
    conn.close()
