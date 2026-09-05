"""
Second-pass gap fill for flexed games.

Some stored games carry placeholder kickoff times (e.g. Sunday 18:00Z) but were
flexed to Saturday. The T-2h snapshot then lands after the real kickoff and the
event is gone from the historical events list. This pass retries with snapshots
26h and 50h before the stored time, matches by team pair in either orientation
within +/-2 days, corrects games.commence_time/home/away, and fetches the
missing game odds + props.
"""

import sqlite3
import time
import os
from datetime import datetime, timedelta, timezone

from fetch_odds_extended import (
    api_get, OutOfCredits, SPORT, ODDS_FORMAT,
    REGIONS_STD, REGIONS_WIDE, GAME_MARKETS, PROP_MARKETS,
    fetch_event_markets, mark_fetched, is_fetched, DB_PATH,
)

import fetch_odds_extended as fx


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def find_real_event(stored_commence, home, away):
    """Search earlier snapshots for the real event; match team pair, either orientation."""
    base = datetime.fromisoformat(stored_commence.replace("Z", "+00:00"))
    pair = frozenset((home, away))
    for hours_back in (26, 50, 74):
        snap = iso(base - timedelta(hours=hours_back))
        data, _ = api_get(f"historical/sports/{SPORT}/events",
                          {"date": snap, "dateFormat": "iso"})
        time.sleep(0.2)
        if not data or "data" not in data:
            continue
        for ev in data["data"]:
            if frozenset((ev["home_team"], ev["away_team"])) == pair:
                ct = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                if abs((ct - base).total_seconds()) <= 2 * 86400:
                    return ev
    return None


def main():
    conn = get_db()

    gaps = conn.execute("""
        SELECT g.event_id, g.commence_time, g.home_team, g.away_team, g.season,
          NOT EXISTS (SELECT 1 FROM games g2 JOIN game_odds o ON o.event_id=g2.event_id
            WHERE g2.home_team IN (g.home_team,g.away_team) AND g2.away_team IN (g.home_team,g.away_team)
              AND abs(julianday(g2.commence_time)-julianday(g.commence_time)) <= 2) AS need_odds,
          NOT EXISTS (SELECT 1 FROM games g2 JOIN player_props p ON p.event_id=g2.event_id
            WHERE g2.home_team IN (g.home_team,g.away_team) AND g2.away_team IN (g.home_team,g.away_team)
              AND abs(julianday(g2.commence_time)-julianday(g.commence_time)) <= 2) AS need_props
        FROM games g
        WHERE g.completed=1 AND g.season >= 2023
        ORDER BY g.commence_time DESC
    """).fetchall()

    todo = [g for g in gaps if g[5] or g[6]]
    print(f"Flex second pass: {len(todo)} games still missing data", flush=True)

    for eid, ct, home, away, season, need_odds, need_props in todo:
        day = ct[:10]
        key = f"flex_{home}_{away}_{day}"
        if is_fetched(conn, "flex_gap", key):
            continue

        ev = find_real_event(ct, home, away)
        if not ev:
            print(f"  [{day}] {away} @ {home}: real event not found, giving up", flush=True)
            mark_fetched(conn, "flex_gap", key, 0)
            continue

        real_ct = ev["commence_time"]
        print(f"  [{day}] {away} @ {home}: real kickoff {real_ct} "
              f"({ev['away_team']} @ {ev['home_team']})", flush=True)

        conn.execute(
            "UPDATE games SET commence_time=?, home_team=?, away_team=? WHERE event_id=?",
            (real_ct, ev["home_team"], ev["away_team"], eid),
        )
        conn.commit()

        snap = iso(datetime.fromisoformat(real_ct.replace("Z", "+00:00")) - timedelta(hours=2))
        total_cost = 0
        if need_odds:
            rows, cost = fetch_event_markets(conn, eid, ev["id"], snap,
                                             GAME_MARKETS.split(","), REGIONS_WIDE)
            total_cost += cost
            print(f"    odds: {rows} rows ({cost} cr)", flush=True)
        if need_props:
            rows, cost = fetch_event_markets(conn, eid, ev["id"], snap,
                                             PROP_MARKETS, REGIONS_STD)
            total_cost += cost
            print(f"    props: {rows} rows ({cost} cr)", flush=True)
        conn.commit()
        mark_fetched(conn, "flex_gap", key, total_cost)

    print(f"\nDone. Session credits used: {fx.credits_used_session:,}; remaining: {fx.credits_remaining}", flush=True)
    conn.close()


if __name__ == "__main__":
    try:
        main()
    except OutOfCredits as e:
        print(f"*** Out of credits: {e} ***", flush=True)
