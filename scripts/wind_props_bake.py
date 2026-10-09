"""
Publish this week's latest DraftKings lines for the Wind Props markets to docs/wind_props_lines.json, the
file the 🌬️ tab reads when you press Check forecasts.

Source: data/props_frames/snapshots/props_<season>.csv (every line the automated pulls record -- the
SaberSim send's slate pull ~80 min before each kickoff, and props_watch). For each (game, market, player) the
newest two-sided DK line is kept. Snapshot rows identify a game only by Odds API event id, so each event's two
teams are read off its players (the week's FFA stat file gives every player's team) and matched to the
nflverse schedule for home/away -- the same team codes the tab's forecast scan uses.

Markets (outputs/reports/props_pattern_mine.md, UNDERS by forecast wind, 2023-25 DK closing lines):
pass yds / completions / attempts, receptions, reception yds. Interceptions are excluded on purpose: wind
pushes them UP.

    python scripts/wind_props_bake.py                    # games kicking off from 6 h ago to 8 days out
    python scripts/wind_props_bake.py --now 2026-10-04T12:00:00Z     # as of another moment (testing)

Games are matched by team pair + kickoff date, not the snapshot's week label: the Wednesday props_watch pull
tags next week's games with the current week number.
"""
import argparse, glob, json, os, re
from datetime import datetime, timedelta, timezone
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "data", "props_frames", "snapshots")
FFA = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
OUT = os.path.join(ROOT, "docs", "wind_props_lines.json")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
MARKETS = {"player_pass_yds": "pass yds", "player_pass_completions": "completions",
           "player_pass_attempts": "pass att", "player_receptions": "receptions",
           "player_reception_yds": "rec yds"}
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "WSH": "WAS"}
NAME2ABBR = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
             "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
             "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
             "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
             "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
             "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
             "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
             "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


def norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("'", ""))
    return re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)).strip()


def player_teams(season, week):
    """norm(name) -> team from the newest FFA stat file at or before the week (any week if none)."""
    files = []
    for f in glob.glob(os.path.join(FFA, f"raw_stats_{season}_wk*.csv")):
        m = re.search(r"_wk(\d+)\.csv$", f)
        if m: files.append((int(m.group(1)), f))
    files = [x for x in files if x[0] <= week] or files
    if not files: return {}
    p = pd.read_csv(sorted(files)[-1][1], usecols=["player", "team"])
    p["team"] = p.team.map(lambda t: FIX.get(t, t))
    return dict(zip(p.player.map(norm), p.team))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--now", default=None, help="ISO UTC time to bake as of (default: now)")
    ap.add_argument("--games", default=None, help="path to nflverse games.csv (default: download)")
    A = ap.parse_args()
    files = sorted(glob.glob(os.path.join(SNAP, "props_*.csv")))
    if not files: raise SystemExit("no snapshot files")
    path = os.path.join(SNAP, f"props_{A.season}.csv") if A.season else files[-1]
    d = pd.read_csv(path)
    season = int(d.season.max())
    now = pd.Timestamp(A.now) if A.now else pd.Timestamp.now(tz="UTC")
    d["ko"] = pd.to_datetime(d.commence, utc=True, errors="coerce")
    d = d[(d.season == season) & d.market.isin(MARKETS) & (d.bookmaker == "draftkings")
          & (d.ko >= now - pd.Timedelta(hours=6)) & (d.ko <= now + pd.Timedelta(days=8))]
    d = d[d.price.abs() >= 100]
    week = int(d.week.max()) if len(d) else 0
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "season": season,
           "as_of_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": MARKETS, "games": []}
    if len(d):
        # newest line per (event, market, player, side), then pair Over/Under on the same point
        d = d.sort_values("snapshot_utc").groupby(["event", "market", "player", "side"]).tail(1)
        o = d[d.side == "Over"]; u = d[d.side == "Under"]
        k = ["event", "market", "player"]
        pr = u.merge(o[k + ["point", "price"]], on=k, suffixes=("", "_o"))
        pr = pr[pr.point == pr.point_o]
        team = player_teams(season, week)
        pr["team"] = pr.player.map(norm).map(team)
        g = pd.read_csv(A.games or GAMES_URL)
        g = g[g.season == season]
        pairs = {(frozenset((r.home_team, r.away_team)), r.gameday): r for r in g.itertuples()}
        # snapshot kickoffs are UTC; nflverse gameday is the Eastern date (a 8:20 pm ET game is the next UTC day)
        et_day = lambda t: (t - pd.Timedelta(hours=5)).strftime("%Y-%m-%d")
        for ev, x in pr.groupby("event"):
            if " @ " in str(ev):                                   # backfill rows carry "Away @ Home" names
                a, h = str(ev).split(" @ ", 1); ts = {NAME2ABBR.get(a), NAME2ABBR.get(h)}
            else:
                ts = set(x.team.dropna().value_counts().index[:2])
            day = et_day(x.ko.iloc[0])
            gm = pairs.get((frozenset(ts), day))
            if gm is None and len(ts) == 1:                          # only one side's players priced so far
                t = next(iter(ts))
                gm = next((r for (k, dd), r in pairs.items() if t in k and dd == day), None)
            if gm is None:
                print(f"  {ev}: teams {sorted(t for t in ts if t)} on {day} not matched to a scheduled game, skipped"); continue
            if any(G["gid"] == gm.game_id for G in out["games"]):     # same game under two event ids (backfill + live)
                G = next(G for G in out["games"] if G["gid"] == gm.game_id)
                have = {(p["market"], p["player"]) for p in G["props"]}
                x = x[[(m, p) not in have for m, p in zip(x.market, x.player)]]
            props = []
            for r in x.sort_values(["market", "point"], ascending=[True, False]).itertuples():
                props.append({"market": r.market, "label": MARKETS[r.market], "player": r.player,
                              "team": r.team if isinstance(r.team, str) else "", "line": float(r.point),
                              "under": int(r.price), "over": int(r.price_o), "pulled": r.snapshot_utc})
            if any(G["gid"] == gm.game_id for G in out["games"]):
                next(G for G in out["games"] if G["gid"] == gm.game_id)["props"] += props
            else:
                out["games"].append({"gid": gm.game_id, "week": int(gm.week), "away": gm.away_team, "home": gm.home_team,
                                     "commence": str(x.commence.iloc[0]), "props": props})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    n = sum(len(g["props"]) for g in out["games"])
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(out['games'])} games, {n} props (as of {out['as_of_utc']})")


if __name__ == "__main__":
    main()
