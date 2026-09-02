"""
NFL Historical Odds Fetcher — extension run (July 2026, new API key).

Burns the full credit budget in priority order:
  Phase 1: gap-fill props + game odds for real games missed in 2023-2025 (newest first)
  Phase 2: backward extension — 2022, 2021, 2020 game odds (us,us2,eu incl. Pinnacle) + scores
           (The Odds API historical floor: 2020-06-06 for game odds; props only exist from 2023-05)
  Phase 3: extra markets never previously bought (alternates, 1st/last TD, defensive props,
           longest-play props, team totals, alt spreads/totals, Q1/H1 markets) for completed
           games from 2025 backward through 2023, until credits run out
  Phase 4: backstop — opening-line snapshots (T-5 days) for 2023-2025 game days

Same SQLite DB and fetch_log resume scheme as fetch_odds.py.
"""

import requests
import sqlite3
import sys
import time
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

API_KEY = "f203d7862f64ff91f9ceeeeccbeb9a32"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "nfl_odds.db")

ODDS_FORMAT = "american"
REGIONS_STD = "us,us2"
REGIONS_WIDE = "us,us2,eu"  # eu adds Pinnacle

GAME_MARKETS = "h2h,spreads,totals"

# Prop markets already covered for 2023-2025 (used only for gap-filling)
PROP_MARKETS = [
    "player_pass_tds", "player_pass_yds", "player_pass_attempts",
    "player_pass_completions", "player_pass_interceptions",
    "player_rush_yds", "player_rush_tds", "player_rush_attempts",
    "player_receptions", "player_reception_yds", "player_reception_tds",
    "player_anytime_td",
    "player_kicking_points", "player_field_goals",
    "player_pass_rush_yds", "player_rush_reception_yds",
]

# Markets never previously bought. Invalid keys are auto-pruned on 422.
EXTRA_PROP_MARKETS = [
    "player_pass_yds_alternate", "player_pass_tds_alternate",
    "player_pass_completions_alternate", "player_pass_attempts_alternate",
    "player_pass_interceptions_alternate",
    "player_rush_yds_alternate", "player_rush_attempts_alternate",
    "player_receptions_alternate", "player_reception_yds_alternate",
    "player_rush_reception_yds_alternate",
    "player_kicking_points_alternate", "player_field_goals_alternate",
    "player_1st_td", "player_last_td",
    "player_sacks", "player_solo_tackles", "player_tackles_assists",
    "player_defensive_interceptions",
    "player_pass_longest_completion", "player_rush_longest", "player_reception_longest",
    "player_pats",
]
EXTRA_GAME_MARKETS = [
    "alternate_spreads", "alternate_totals",
    "team_totals", "alternate_team_totals",
    "h2h_q1", "spreads_q1", "totals_q1",
    "h2h_h1", "spreads_h1", "totals_h1",
]

NEW_SEASONS = {  # newest first
    2022: {"start": "2022-09-07T00:00:00Z", "end": "2023-02-20T00:00:00Z"},
    2021: {"start": "2021-09-08T00:00:00Z", "end": "2022-02-20T00:00:00Z"},
    2020: {"start": "2020-09-09T00:00:00Z", "end": "2021-02-15T00:00:00Z"},
}

CREDIT_FLOOR = 25  # stop cleanly when the tank is essentially empty

credits_used_session = 0
credits_remaining = None
BAD_MARKETS = set()


class OutOfCredits(Exception):
    pass


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
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
    """API request with retries, credit tracking, and invalid-market pruning.

    Returns (json_or_None, cost). Raises OutOfCredits when the budget is gone.
    """
    global credits_used_session, credits_remaining

    if credits_remaining is not None and credits_remaining <= CREDIT_FLOOR:
        raise OutOfCredits(f"Remaining credits {credits_remaining} <= floor {CREDIT_FLOOR}")

    params = dict(params)
    params["apiKey"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=45)
        except requests.exceptions.RequestException as e:
            print(f"    request error: {e}", flush=True)
            time.sleep(8)
            continue

        if r.status_code == 429:
            print("    429 rate limit — sleeping 65s", flush=True)
            time.sleep(65)
            continue

        if r.status_code == 401:
            txt = r.text[:200]
            if "USAGE" in txt.upper() or "QUOTA" in txt.upper():
                raise OutOfCredits(txt)
            print(f"    401: {txt}", flush=True)
            return None, 0

        if r.status_code == 422:
            # Either an invalid market key, or no snapshot for that date.
            txt = r.text
            if "markets" in params:
                requested = params["markets"].split(",")
                bad = [m for m in requested if m in txt]
                if bad and len(bad) < len(requested):
                    for m in bad:
                        BAD_MARKETS.add(m)
                    print(f"    pruning invalid markets: {bad}", flush=True)
                    params["markets"] = ",".join(m for m in requested if m not in bad)
                    continue
                if bad:
                    for m in bad:
                        BAD_MARKETS.add(m)
            return None, 0

        if r.status_code == 404:
            return None, 0

        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:200]}", flush=True)
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return None, 0

        cost = int(r.headers.get("x-requests-last", 0) or 0)
        rem = r.headers.get("x-requests-remaining")
        if rem is not None:
            credits_remaining = int(float(rem))
        credits_used_session += cost
        return r.json(), cost

    return None, 0


def filter_markets(markets):
    return [m for m in markets if m not in BAD_MARKETS]


def batches(lst, n=5):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def snapshot_for(commence_iso, hours_before=2):
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    return (dt - timedelta(hours=hours_before)).strftime("%Y-%m-%dT%H:%M:%SZ")


def determine_season(commence_time_str):
    dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    return dt.year if dt.month >= 8 else dt.year - 1


def discover_events_at(snapshot_str):
    """Historical events at a snapshot; returns {(home, away): event_id}."""
    data, _ = api_get(
        f"historical/sports/{SPORT}/events",
        {"date": snapshot_str, "dateFormat": "iso"},
    )
    out = {}
    if data and "data" in data:
        for ev in data["data"]:
            out[(ev["home_team"], ev["away_team"])] = ev["id"]
    return out


def insert_prop_rows(conn, our_event_id, event_data, snap_time):
    n = 0
    for bk in event_data.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                player_name = outcome.get("description", outcome.get("name", ""))
                conn.execute(
                    """INSERT INTO player_props
                       (event_id, bookmaker, market, player_name, outcome_type, price, point, snapshot_time, last_update)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (our_event_id, bk["key"], mkt["key"], player_name,
                     outcome.get("name", ""), outcome.get("price"),
                     outcome.get("point"), snap_time, bk.get("last_update")),
                )
                n += 1
    return n


def insert_game_rows(conn, our_event_id, event_data, snap_time):
    n = 0
    for bk in event_data.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                name = outcome.get("name", "")
                desc = outcome.get("description")
                if desc:  # team_totals carry the team in description
                    name = f"{desc} {name}"
                conn.execute(
                    """INSERT INTO game_odds
                       (event_id, bookmaker, market, outcome_name, price, point, snapshot_time, last_update)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (our_event_id, bk["key"], mkt["key"], name,
                     outcome.get("price"), outcome.get("point"),
                     snap_time, bk.get("last_update")),
                )
                n += 1
    return n


def fetch_event_markets(conn, our_event_id, hist_eid, snapshot_str, markets, regions):
    """Fetch a market list for one event, routing rows to player_props / game_odds."""
    total_rows = 0
    total_cost = 0
    for batch in batches(filter_markets(markets)):
        data, cost = api_get(
            f"historical/sports/{SPORT}/events/{hist_eid}/odds",
            {"date": snapshot_str, "regions": regions,
             "markets": ",".join(batch), "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"},
        )
        total_cost += cost
        if data and "data" in data:
            ev = data["data"]
            snap = data.get("timestamp", snapshot_str)
            prop_ev = {"bookmakers": []}
            game_ev = {"bookmakers": []}
            for bk in ev.get("bookmakers", []):
                pm = [m for m in bk.get("markets", []) if m["key"].startswith("player_")]
                gm = [m for m in bk.get("markets", []) if not m["key"].startswith("player_")]
                if pm:
                    prop_ev["bookmakers"].append({**bk, "markets": pm})
                if gm:
                    game_ev["bookmakers"].append({**bk, "markets": gm})
            total_rows += insert_prop_rows(conn, our_event_id, prop_ev, snap)
            total_rows += insert_game_rows(conn, our_event_id, game_ev, snap)
        time.sleep(0.25)
    return total_rows, total_cost


# ──────────────────────────────────────────────
# Phase 1: gap-fill 2023-2025 (real games only)
# ──────────────────────────────────────────────

def gap_fill():
    conn = get_db()

    prop_gaps = conn.execute("""
        SELECT g.event_id, g.commence_time, g.home_team, g.away_team, g.season FROM games g
        WHERE g.completed=1
          AND NOT EXISTS (
            SELECT 1 FROM games g2 JOIN player_props p ON p.event_id=g2.event_id
            WHERE g2.home_team=g.home_team AND g2.away_team=g.away_team
              AND date(g2.commence_time)=date(g.commence_time))
        ORDER BY g.commence_time DESC
    """).fetchall()

    odds_gaps = conn.execute("""
        SELECT g.event_id, g.commence_time, g.home_team, g.away_team, g.season FROM games g
        WHERE g.completed=1
          AND NOT EXISTS (
            SELECT 1 FROM games g2 JOIN game_odds o ON o.event_id=g2.event_id
            WHERE g2.home_team=g.home_team AND g2.away_team=g.away_team
              AND date(g2.commence_time)=date(g.commence_time))
        ORDER BY g.commence_time DESC
    """).fetchall()

    print(f"\n{'='*60}\nPhase 1: gap-fill — {len(prop_gaps)} prop gaps, {len(odds_gaps)} odds gaps\n{'='*60}", flush=True)

    # union of dates so we only discover events once per day
    by_day = defaultdict(lambda: {"props": [], "odds": []})
    for g in prop_gaps:
        by_day[g[1][:10]]["props"].append(g)
    for g in odds_gaps:
        by_day[g[1][:10]]["odds"].append(g)

    for day in sorted(by_day, reverse=True):
        work = by_day[day]
        pending_p = [g for g in work["props"]
                     if not is_fetched(conn, "props_gap", f"{g[2]}_{g[3]}_{day}")]
        pending_o = [g for g in work["odds"]
                     if not is_fetched(conn, "odds_gap", f"{g[2]}_{g[3]}_{day}")]
        if not pending_p and not pending_o:
            continue

        first = min(g[1] for g in pending_p + pending_o)
        snap = snapshot_for(first)
        hist = discover_events_at(snap)
        time.sleep(0.25)

        for g in pending_o:
            eid, ct, home, away, season = g
            key = f"{home}_{away}_{day}"
            hist_eid = hist.get((home, away))
            if not hist_eid:
                print(f"  [{day}] odds-gap {away} @ {home}: no historical event", flush=True)
                mark_fetched(conn, "odds_gap", key, 0)
                continue
            rows, cost = fetch_event_markets(conn, eid, hist_eid, snap,
                                             GAME_MARKETS.split(","), REGIONS_WIDE)
            conn.commit()
            mark_fetched(conn, "odds_gap", key, cost)
            print(f"  [{day}] odds-gap {away} @ {home}: {rows} rows ({cost} cr)", flush=True)

        for g in pending_p:
            eid, ct, home, away, season = g
            key = f"{home}_{away}_{day}"
            hist_eid = hist.get((home, away))
            if not hist_eid:
                print(f"  [{day}] prop-gap {away} @ {home}: no historical event", flush=True)
                mark_fetched(conn, "props_gap", key, 0)
                continue
            rows, cost = fetch_event_markets(conn, eid, hist_eid, snap,
                                             PROP_MARKETS, REGIONS_STD)
            conn.commit()
            mark_fetched(conn, "props_gap", key, cost)
            print(f"  [{day}] prop-gap {away} @ {home}: {rows} rows ({cost} cr)", flush=True)

    conn.close()


# ──────────────────────────────────────────────
# Phase 2: 2020-2022 discovery + game odds + scores
# ──────────────────────────────────────────────

def backfill_old_seasons():
    conn = get_db()
    for season, dates in NEW_SEASONS.items():
        print(f"\n{'='*60}\nPhase 2: {season} season\n{'='*60}", flush=True)

        # discovery
        if not is_fetched(conn, "discover", f"events_{season}"):
            start = datetime.fromisoformat(dates["start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(dates["end"].replace("Z", "+00:00"))
            current = start
            all_events = {}
            while current < end:
                date_str = current.strftime("%Y-%m-%dT12:00:00Z")
                data, _ = api_get(f"historical/sports/{SPORT}/events",
                                  {"date": date_str, "dateFormat": "iso"})
                if data and "data" in data:
                    for ev in data["data"]:
                        all_events.setdefault(ev["id"], ev)
                current += timedelta(days=2)
                time.sleep(0.2)
            for eid, ev in all_events.items():
                conn.execute(
                    """INSERT OR IGNORE INTO games (event_id, sport_key, home_team, away_team, commence_time, season)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (eid, ev["sport_key"], ev["home_team"], ev["away_team"],
                     ev["commence_time"], determine_season(ev["commence_time"])),
                )
            conn.commit()
            mark_fetched(conn, "discover", f"events_{season}")
            print(f"  discovered {len(all_events)} events", flush=True)

        # bulk game odds per game day (wide regions: includes Pinnacle)
        games = conn.execute(
            "SELECT event_id, commence_time, home_team, away_team FROM games WHERE season=? ORDER BY commence_time",
            (season,),
        ).fetchall()
        daily = defaultdict(list)
        for g in games:
            daily[g[1][:10]].append(g)

        for day, day_games in sorted(daily.items(), reverse=True):
            key = f"game_odds_{day}"
            if is_fetched(conn, "game_odds", key):
                continue
            snap = snapshot_for(min(g[1] for g in day_games))
            data, cost = api_get(
                f"historical/sports/{SPORT}/odds",
                {"date": snap, "regions": REGIONS_WIDE, "markets": GAME_MARKETS,
                 "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"},
            )
            rows = 0
            if data and "data" in data:
                snap_time = data.get("timestamp", snap)
                ids_today = {g[0] for g in day_games}
                for ev in data["data"]:
                    if ev["id"] in ids_today:
                        rows += insert_game_rows(conn, ev["id"], ev, snap_time)
                conn.commit()
            mark_fetched(conn, "game_odds", key, cost)
            print(f"  [{day}] {len(day_games)} games -> {rows} odds rows ({cost} cr)", flush=True)
            time.sleep(0.25)

        # scores
        for day, day_games in sorted(daily.items(), reverse=True):
            key = f"scores_{day}"
            if is_fetched(conn, "scores_bf", key):
                continue
            dt = datetime.fromisoformat(f"{day}T00:00:00+00:00") + timedelta(days=1, hours=12)
            data, cost = api_get(
                f"historical/sports/{SPORT}/scores",
                {"date": dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "dateFormat": "iso"},
            )
            updated = 0
            if data and "data" in data:
                ids_today = {g[0] for g in day_games}
                for ev in data["data"]:
                    if ev["id"] in ids_today and ev.get("completed"):
                        hs = as_ = None
                        for s in ev.get("scores") or []:
                            if s["name"] == ev.get("home_team"):
                                hs = int(s["score"]) if s["score"] else None
                            else:
                                as_ = int(s["score"]) if s["score"] else None
                        if hs is not None:
                            conn.execute(
                                "UPDATE games SET home_score=?, away_score=?, completed=1 WHERE event_id=?",
                                (hs, as_, ev["id"]),
                            )
                            updated += 1
                conn.commit()
            mark_fetched(conn, "scores_bf", key, cost)
            time.sleep(0.2)
        print(f"  scores updated for {season}", flush=True)

    conn.close()


# ──────────────────────────────────────────────
# Phase 3: extra markets, 2025 -> 2023
# ──────────────────────────────────────────────

def fetch_extra_markets():
    conn = get_db()
    games = conn.execute("""
        SELECT g.event_id, g.commence_time, g.home_team, g.away_team, g.season FROM games g
        WHERE g.completed=1 AND g.season >= 2023
        ORDER BY g.commence_time DESC
    """).fetchall()

    # de-dup matchup+date (2023 has duplicate event_ids)
    seen = set()
    todo = []
    for g in games:
        k = (g[2], g[3], g[1][:10])
        if k not in seen:
            seen.add(k)
            todo.append(g)

    print(f"\n{'='*60}\nPhase 3: extra markets for {len(todo)} games (newest first)\n{'='*60}", flush=True)

    daily = defaultdict(list)
    for g in todo:
        daily[g[1][:10]].append(g)

    for day in sorted(daily, reverse=True):
        day_games = daily[day]
        pending = [g for g in day_games
                   if not is_fetched(conn, "extra", f"{g[2]}_{g[3]}_{day}")]
        if not pending:
            continue
        snap = snapshot_for(min(g[1] for g in pending))
        hist = discover_events_at(snap)
        time.sleep(0.25)

        for g in pending:
            eid, ct, home, away, season = g
            key = f"{home}_{away}_{day}"
            hist_eid = hist.get((home, away))
            if not hist_eid:
                print(f"  [{day}] {away} @ {home}: no historical event", flush=True)
                mark_fetched(conn, "extra", key, 0)
                continue
            markets = EXTRA_PROP_MARKETS + EXTRA_GAME_MARKETS
            rows, cost = fetch_event_markets(conn, eid, hist_eid, snap, markets, REGIONS_STD)
            conn.commit()
            mark_fetched(conn, "extra", key, cost)
            rem = credits_remaining if credits_remaining is not None else "?"
            print(f"  [{day}] {away} @ {home}: {rows} rows ({cost} cr, {rem} left)", flush=True)

    conn.close()


# ──────────────────────────────────────────────
# Phase 4: opening-line snapshots (T-5 days), 2025 -> 2023
# ──────────────────────────────────────────────

def fetch_opening_lines():
    conn = get_db()
    games = conn.execute("""
        SELECT g.event_id, g.commence_time, g.home_team, g.away_team FROM games g
        WHERE g.completed=1 AND g.season >= 2023
        ORDER BY g.commence_time DESC
    """).fetchall()

    daily = defaultdict(list)
    for g in games:
        daily[g[1][:10]].append(g)

    print(f"\n{'='*60}\nPhase 4: opening lines (T-5d) for {len(daily)} game days\n{'='*60}", flush=True)

    for day in sorted(daily, reverse=True):
        key = f"open_odds_{day}"
        if is_fetched(conn, "open_odds", key):
            continue
        day_games = daily[day]
        snap = snapshot_for(min(g[1] for g in day_games), hours_before=120)
        data, cost = api_get(
            f"historical/sports/{SPORT}/odds",
            {"date": snap, "regions": REGIONS_WIDE, "markets": GAME_MARKETS,
             "oddsFormat": ODDS_FORMAT, "dateFormat": "iso"},
        )
        rows = 0
        if data and "data" in data:
            snap_time = data.get("timestamp", snap)
            # match snapshot events to our stored event ids by teams+date
            lookup = {}
            for g in day_games:
                lookup[(g[2], g[3])] = g[0]
            for ev in data["data"]:
                our_id = lookup.get((ev["home_team"], ev["away_team"]))
                if our_id:
                    rows += insert_game_rows(conn, our_id, ev, snap_time)
            conn.commit()
        mark_fetched(conn, "open_odds", key, cost)
        rem = credits_remaining if credits_remaining is not None else "?"
        print(f"  [{day}] opening: {rows} rows ({cost} cr, {rem} left)", flush=True)
        time.sleep(0.25)

    conn.close()


def print_summary():
    conn = get_db()
    print(f"\n{'='*60}\nRUN SUMMARY\n{'='*60}", flush=True)
    print(f"  Session credits used: {credits_used_session:,}", flush=True)
    print(f"  Credits remaining:    {credits_remaining}", flush=True)
    if BAD_MARKETS:
        print(f"  Invalid markets pruned: {sorted(BAD_MARKETS)}", flush=True)
    for season in [2020, 2021, 2022, 2023, 2024, 2025]:
        sg = conn.execute("SELECT COUNT(*) FROM games WHERE season=?", (season,)).fetchone()[0]
        so = conn.execute(
            "SELECT COUNT(*) FROM game_odds WHERE event_id IN (SELECT event_id FROM games WHERE season=?)",
            (season,)).fetchone()[0]
        sp = conn.execute(
            "SELECT COUNT(*) FROM player_props WHERE event_id IN (SELECT event_id FROM games WHERE season=?)",
            (season,)).fetchone()[0]
        print(f"  {season}: {sg} games | {so:,} game-odds rows | {sp:,} prop rows", flush=True)
    conn.close()


if __name__ == "__main__":
    print(f"NFL odds extension run — {datetime.now(timezone.utc).isoformat()}", flush=True)
    try:
        gap_fill()
        backfill_old_seasons()
        fetch_extra_markets()
        fetch_opening_lines()
    except OutOfCredits as e:
        print(f"\n*** Credit budget exhausted: {e} ***", flush=True)
    except KeyboardInterrupt:
        print("\n*** Interrupted ***", flush=True)
    finally:
        print_summary()
