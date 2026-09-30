"""Bake the public trade analyzer's player file: docs/trade_2026.js (global TRADE_DATA).

The page (docs/trade.html) is for anyone's league, so it cannot carry one league's scoring. What it
needs instead is a per-game STAT LINE for every player that the browser can score under whatever
rules the visitor enters. The level of that line comes from the same fitted rest-of-season ridge the
gm trade calculator uses (gm/players.py, walk-forward MAE 3.11 vs 3.31 for the naive blend); the MIX
of stats comes from FFA's next-week projection and this season's box scores. So:

    mix    = weighted mean of the per-game stat lines available: FFA next week 0.5, this season
             to date 0.25, last season 0.25 (renormalised over whichever exist)
    scaled = mix x (ridge ROS ppg / mix scored in the fitting league's rules)

Scored under the fitting league's rules the scaled line reproduces the ridge exactly; under 4-pt
passing TDs or half PPR it moves the way the player's stat mix says it should. The approximation is
the uniform scale: a player whose ridge level is above his mix is assumed to be above it in every stat.

Also carried: ids for matching a visitor's league (ESPN, Sleeper, Yahoo, gsis), bye week, injury
status (FFA's weekly tag, nflverse's RES for injured reserve) and the hot-start "luck" figure from
outputs/reports/hot_start_board_<season>.csv when that file exists (display only; it is not blended
into the level -- it was measured against a manager's naive read, not against the ridge).

Usage: python scripts/trade_tab_bake.py [--season 2026]
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm import config, players as gp  # noqa: E402

OUT = os.path.join(ROOT, "docs", "trade_{season}.js")
ROSTER_URL = ("https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/"
              "roster_weekly_{season}.parquet")
# nflverse box-score column -> the stat names the page scores (FFA's raw-stat file uses the right side)
NFLV = {"passing_yards": "pass_yds", "passing_tds": "pass_tds", "passing_interceptions": "pass_int",
        "rushing_yards": "rush_yds", "rushing_tds": "rush_tds", "receptions": "rec",
        "receiving_yards": "rec_yds", "receiving_tds": "rec_tds", "fumbles_lost": "fumbles_lost"}
STATS = list(NFLV.values())
BONUS = ["pass_300_yds", "rush_100_yds", "rec_100_yds"]   # FFA's per-game probabilities, for bonus rules
FFA_INJ = {"Q": "QUESTIONABLE", "D": "DOUBTFUL", "O": "OUT"}
# Weights for the stat MIX only (the level is the ridge's). Three weeks of box scores are mostly
# touchdown noise -- Josh Allen ran for 1.1 TDs a game on this season alone -- so the projection and
# last season's full line carry more of the shape.
MIX_W = {"ffa": 0.5, "ytd": 0.25, "p1": 0.25}
MIN_PPG = 2.0          # below this in the fitting league's scoring nobody trades for him


def rosters(season):
    """Latest weekly row per player: ids, team, status. Fresh download, cache as the fallback."""
    try:
        r = pd.read_parquet(ROSTER_URL.format(season=season))
    except Exception:
        p = os.path.join(gp.CACHE, f"roster_{season}.csv")
        r = pd.read_csv(p, dtype=str) if os.path.exists(p) else pd.DataFrame()
    if r.empty:
        return r
    r["week"] = pd.to_numeric(r["week"], errors="coerce")
    r = r.sort_values("week").drop_duplicates("gsis_id", keep="last")
    return r.set_index("gsis_id")


def clean_id(x):
    s = str(x).split(".")[0] if pd.notna(x) else ""
    return "" if s in ("", "nan", "None") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--config", default=os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json"))
    a = ap.parse_args()
    S = a.season
    cfg = config.load(a.config)
    per_stat = cfg.raw["scoring"]["per_stat"]
    score = lambda line: sum(float(line.get(k, 0)) * float(per_stat.get(k, 0)) for k in STATS)

    weekly = gp._weekly_all(cfg, S)
    cur = weekly[weekly.season == S]
    week = int(cur.week.max()) if len(cur) else 0
    ffa, stale = gp.ffa_next_week(cfg, S, week)
    feats = gp.history_features(weekly, S, week, ffa)
    feats["ridge"] = gp.ridge_predict(gp.load_model(), feats)
    print(f"{S} through week {week}; ridge fitted for {feats.ridge.notna().sum()} players"
          f"{' (FFA file is last week: stale)' if stale else ''}")

    # per-game stat lines: this season, last season, FFA next week
    def per_game(frame):
        g = frame.groupby("player_id")
        return g[list(NFLV)].mean().rename(columns=NFLV)
    ytd = per_game(cur)
    p1 = per_game(weekly[weekly.season == S - 1])
    ffa_week = week if stale else week + 1
    fp = os.path.join(gp.FFA_DIR, f"raw_stats_{S}_wk{ffa_week}.csv")
    F = pd.read_csv(fp, low_memory=False) if os.path.exists(fp) else pd.DataFrame(columns=["player", "position"])
    F = F[F.position.isin(gp.SKILL)].copy()
    F["key"] = F.player.map(gp.norm)
    for c in STATS + BONUS:
        F[c] = pd.to_numeric(F[c], errors="coerce").fillna(0.0) if c in F.columns else 0.0
    F = F.drop_duplicates(["key", "position"]).set_index(["key", "position"])

    R = rosters(S)
    hot = {}
    hp = os.path.join(ROOT, "outputs", "reports", f"hot_start_board_{S}.csv")
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        hot = {r.player_id: (float(r.luck), float(r.pred_gap)) for r in h.itertuples()
               if np.isfinite(r.luck) and np.isfinite(r.pred_gap)}

    out = []
    for pid, f in feats.iterrows():
        key, pos = f["key"], f["position"]
        fk = (key, pos) if (key, pos) in F.index else None
        rr = R.loc[pid] if pid in R.index else None
        if rr is not None and str(rr.get("status")) in ("RET", "CUT") and fk is None:
            continue
        if rr is None and fk is None and pid not in ytd.index:
            continue                                   # not on a 2026 roster, not projected, not playing
        ridge = f["ridge"]
        if not np.isfinite(ridge):
            continue
        lines = []                                     # (weight, per-game line)
        if fk is not None:
            lines.append((MIX_W["ffa"], {s: float(F.loc[fk, s]) for s in STATS}))
        if pid in ytd.index:
            lines.append((MIX_W["ytd"], {s: float(ytd.loc[pid, s]) for s in STATS}))
        if pid in p1.index:
            lines.append((MIX_W["p1"], {s: float(p1.loc[pid, s]) for s in STATS}))
        if not lines:
            continue
        wsum = sum(w for w, _ in lines)
        mix = {s: sum(w * l[s] for w, l in lines) / wsum for s in STATS}
        base = score(mix)
        if ridge < MIN_PPG or base <= 0.5:
            continue
        k = float(np.clip(ridge / base, 0.3, 3.0))
        line = {s: round(mix[s] * k, 3) for s in STATS}
        for b in BONUS:
            v = float(F.loc[fk].get(b, 0) or 0) if fk is not None else 0.0
            if b in F.columns and fk is not None and np.isfinite(v):
                line[b] = round(v, 3)
        team = (str(rr["team"]) if rr is not None and pd.notna(rr.get("team")) else
                str(F.loc[fk, "team"]) if fk is not None else "")
        team = gp.TEAM_FIX.get(team, team)
        inj = ""
        if fk is not None and pd.notna(F.loc[fk].get("injury_status")):
            inj = FFA_INJ.get(str(F.loc[fk, "injury_status"]), "")
        if rr is not None and str(rr.get("status")) == "RES":
            inj = "INJURY_RESERVE"
        rec = {"n": f["player_display_name"], "p": pos, "t": team, "g": pid,
               "e": clean_id(rr.get("espn_id")) if rr is not None else "",
               "s": clean_id(rr.get("sleeper_id")) if rr is not None else "",
               "y": clean_id(rr.get("yahoo_id")) if rr is not None else "",
               "ppg": round(float(ridge), 2), "gp": int(f["ytd_games"]) if pd.notna(f["ytd_games"]) else 0,
               "ytd": round(float(f["ytd_mean"]), 2) if pd.notna(f["ytd_mean"]) else None,
               "l": line}
        if inj:
            rec["i"] = inj
        if pid in hot:
            rec["luck"], rec["gap"] = round(hot[pid][0], 2), round(hot[pid][1], 2)
        out.append(rec)
    out.sort(key=lambda r: -r["ppg"])

    byes = gp.bye_weeks(S)
    sched = pd.read_csv(gp.SCHEDULE.format(season=S), usecols=["game_type", "week"])
    last_reg = int(sched[sched.game_type == "REG"].week.max())
    doc = {"season": S, "week": week, "next_week": week + 1, "last_week": last_reg,
           "ffa_week": ffa_week, "ffa_stale": bool(stale),
           "hot_start": bool(hot), "byes": {k: int(v) for k, v in byes.items()},
           "fit_scoring": {k: per_stat.get(k, 0) for k in STATS},
           "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "players": out}
    path = OUT.format(season=S)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("// baked by scripts/trade_tab_bake.py -- do not edit\nwindow.TRADE_DATA=")
        json.dump(doc, fh, separators=(",", ":"))
        fh.write(";\n")
    by = pd.Series([r["p"] for r in out]).value_counts().to_dict()
    ids = sum(bool(r["e"]) for r in out), sum(bool(r["s"]) for r in out)
    print(f"wrote {os.path.relpath(path, ROOT)}: {len(out)} players {by}; espn ids {ids[0]}, sleeper ids {ids[1]}; "
          f"{os.path.getsize(path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
