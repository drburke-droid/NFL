"""Append-only store of every Odds API line we pull, so models can be re-analysed later against the
lines that were actually up at the time (the DB's 2023-25 prop history is closing-only; this is the
early-line history it never had).

  data/props_frames/snapshots/props_<season>.csv   one row per (snapshot, event, market, player, side)

Columns: snapshot_utc, source, season, week, event, commence, bookmaker, market, player, side, point, price.
`source` = which pull wrote it (props_watch = the Wed/Thu receiving-yards pull; sabersim_send = the T-90
slate pull that feeds the projection blend; backfill = a cache file imported after the fact).

    from odds_snapshots import record
    record(rows, source="props_watch", season=2026, week=2)

`rows` are dicts; keys are taken loosely (player / player_name, event / event_id, commence / kick), so both
pull scripts can pass what they already build.  Returns the number of rows written.
"""
import csv, os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "props_frames", "snapshots")
COLS = ["snapshot_utc", "source", "season", "week", "event", "commence", "bookmaker", "market", "player", "side", "point", "price"]


def record(rows, source, season, week, bookmaker="draftkings", snapshot_utc=None):
    if not rows: return 0
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, f"props_{int(season)}.csv")
    ts = snapshot_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    n = 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new: w.writeheader()
        for r in rows:
            g = lambda *ks: next((r[k] for k in ks if k in r and r[k] is not None), "")
            w.writerow({"snapshot_utc": ts, "source": source, "season": int(season), "week": int(week) if week is not None else "",
                        "event": g("event", "event_id", "eid"), "commence": g("commence", "kick", "commence_time"),
                        "bookmaker": g("bookmaker") or bookmaker, "market": g("market"), "player": g("player", "player_name", "description"),
                        "side": g("side", "name"), "point": g("point"), "price": g("price")})
            n += 1
    return n
