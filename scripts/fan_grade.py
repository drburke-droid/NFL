"""Grade Fan Picks against the box scores — shadow mode: this reads the recorded arrows and the
actuals and reports whether the fans were right; nothing here feeds the model.

Every recorded row (data/fan_adjustments/fan_adjustments_long.csv, from fan_adjust_record.py) is one
fan's arrow on one player-stat: the baseline the fan saw, the adjusted value the page showed
(baseline x (1 + 0.1 x arrows)), and the arrows themselves. Once nflverse has the game's box score:

  direction hit   the actual moved from the baseline in the arrow's direction (actual == baseline is
                  neutral and excluded from the rate)
  error removed   |actual - baseline| - |actual - adjusted|, in the stat's own units and in DK
                  fantasy points (yards x0.1 / x0.04 passing, TD x6 / x4 passing, rec x1, INT x1);
                  positive = the fan's number was closer than ours
  pending         the game has not been ingested by nflverse yet (both teams must have rows)

A player with no box-score row after his game is ingested scored 0 in every stat (inactive).

Outputs docs/fan/grade.json (what docs/fan.html renders: overall, leaderboard by fan, by stat, by
arrow size, the graded rows) and prints a summary. Refreshed by the daily grade in the send
workflow and by the grade workflow; run by hand any time.

Usage: python scripts/fan_grade.py [--season 2026] [--selftest]
"""
import os, re, sys, csv, json, argparse
from datetime import datetime, timezone
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--selftest", action="store_true", help="grade a synthetic submission against synthetic actuals and print the checks")
A = ap.parse_args()
LONG = os.path.join(ROOT, "data", "fan_adjustments", "fan_adjustments_long.csv")
OUT = os.path.join(ROOT, "docs", "fan", "grade.json")
COL = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "pass_int": "passing_interceptions", "rush_yds": "rushing_yards",
       "rush_tds": "rushing_tds", "rec": "receptions", "rec_yds": "receiving_yards", "rec_tds": "receiving_tds"}
PTS = {"pass_yds": 0.04, "pass_tds": 4.0, "pass_int": 1.0, "rush_yds": 0.1, "rush_tds": 6.0, "rec": 1.0, "rec_yds": 0.1, "rec_tds": 6.0}
def norm(s): return re.sub(r"[^a-z]", "", str(s).lower().replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", ""))

def grade(rows, act):
    """rows: DataFrame of recorded arrows; act: nflverse weekly frame (player_id, player_display_name, position, team, week + stat cols).
    Returns the graded frame (one row per arrow) with actual / pending / err columns."""
    act = act.copy(); act["nname"] = act.player_display_name.map(norm)
    have = act.groupby("week").team.apply(set).to_dict()
    by_id = {(r.player_id, int(r.week)): r for r in act.itertuples()}
    by_nm = {(r.nname, r.position, int(r.week)): r for r in act.itertuples()}
    out = []
    for r in rows.itertuples():
        wk = int(r.week); stat = r.stat
        ingested = r.team in have.get(wk, set()) and r.opp in have.get(wk, set())
        a = by_id.get((r.player_id, wk)) if isinstance(r.player_id, str) and r.player_id else None
        if a is None: a = by_nm.get((norm(r.player), r.pos, wk))
        actual = float(getattr(a, COL[stat]) or 0.0) if a is not None else (0.0 if ingested else np.nan)
        base, adj, n = float(r.baseline), float(r.adjusted), int(r.arrows)
        d = dict(r._asdict()); d.pop("Index", None)
        # every column exists on every row, graded or not: with all rows pending (a submission
        # in before its games kick off) these were absent entirely and summarise() died on
        # gd.direction, so a fan who submitted early broke the whole grade until his games ran
        d.update({"pending": not ingested, "actual": None if not ingested else round(actual, 2),
                  "direction": None, "err_base": np.nan, "err_adj": np.nan,
                  "removed": np.nan, "removed_pts": np.nan})
        if ingested:
            move = actual - base
            d["direction"] = "neutral" if abs(move) < 1e-9 else ("hit" if np.sign(move) == np.sign(n) else "miss")
            d["err_base"] = round(abs(actual - base), 3); d["err_adj"] = round(abs(actual - adj), 3)
            d["removed"] = round(d["err_base"] - d["err_adj"], 3)
            d["removed_pts"] = round(d["removed"] * PTS[stat], 3)
        out.append(d)
    return pd.DataFrame(out)

def summarise(g):
    def block(d):
        gd = d[~d.pending]
        dec = gd[gd.direction != "neutral"]
        return {"n": int(len(d)), "graded": int(len(gd)), "pending": int(d.pending.sum()),
                "hit_rate": round(float((dec.direction == "hit").mean()), 3) if len(dec) else None, "hits": int((dec.direction == "hit").sum()), "misses": int((dec.direction == "miss").sum()),
                "closer": int((gd.removed > 0).sum()) if len(gd) else 0, "farther": int((gd.removed < 0).sum()) if len(gd) else 0,
                "removed_pts": round(float(gd.removed_pts.sum()), 2) if len(gd) else None,
                "mae_base_pts": round(float((gd.err_base * gd.stat.map(PTS)).mean()), 3) if len(gd) else None,
                "mae_adj_pts": round(float((gd.err_adj * gd.stat.map(PTS)).mean()), 3) if len(gd) else None}
    s = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "overall": block(g)}
    s["by_fan"] = sorted([dict(fan=f, weeks=sorted(int(w) for w in d.week.unique()), **block(d)) for f, d in g.groupby("fan")],
                         key=lambda x: (-(x["removed_pts"] or 0), -x["graded"], x["fan"]))
    s["by_stat"] = {st: block(d) for st, d in g.groupby("stat")}
    g["size"] = g.arrows.abs().clip(upper=4).map(lambda k: {1: "1", 2: "2", 3: "3", 4: "4+"}[int(k)])
    s["by_size"] = {k: block(d) for k, d in g.groupby("size")}
    s["by_direction"] = {k: block(d) for k, d in g.groupby(g.arrows.map(lambda n: "boost" if n > 0 else "fade"))}
    s["by_week"] = {int(w): block(d) for w, d in g.groupby("week")}
    gd = g[~g.pending].sort_values(["week", "fan", "removed_pts"], ascending=[False, True, False])
    s["rows"] = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items() if k in ("week", "fan", "player", "team", "pos", "stat", "arrows", "baseline", "adjusted", "actual", "direction", "removed_pts")}
                 for r in gd.head(400).to_dict("records")]
    pend = g[g.pending]
    s["pending_games"] = sorted({f"wk{int(r.week)} {r.team} v {r.opp}" for r in pend.itertuples()})
    return s

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return x

if A.selftest:
    rows = pd.DataFrame([
        dict(season=2026, week=1, fan="A", submitted_at="t", bake_id="b", comment="", player_index=0, player_id="00-1", player="X One", team="BUF", pos="WR", opp="DET", stat="rec_yds", arrows=2, baseline=50.0, adjusted=60.0, proj_pts=10),
        dict(season=2026, week=1, fan="A", submitted_at="t", bake_id="b", comment="", player_index=1, player_id="00-2", player="Y Two", team="BUF", pos="RB", opp="DET", stat="rush_tds", arrows=-1, baseline=0.8, adjusted=0.72, proj_pts=10),
        dict(season=2026, week=1, fan="B", submitted_at="t", bake_id="b", comment="", player_index=2, player_id="", player="Z Three", team="DET", pos="TE", opp="BUF", stat="rec", arrows=3, baseline=4.0, adjusted=5.2, proj_pts=8),
        dict(season=2026, week=2, fan="B", submitted_at="t", bake_id="b", comment="", player_index=3, player_id="00-4", player="W Four", team="KC", pos="QB", opp="LAC", stat="pass_yds", arrows=1, baseline=250.0, adjusted=275.0, proj_pts=18)])
    act = pd.DataFrame([dict(player_id="00-1", player_display_name="X One", position="WR", team="BUF", week=1, receiving_yards=80.0, rushing_tds=0, receptions=0, passing_yards=0),
                        dict(player_id="00-2", player_display_name="Y Two", position="RB", team="BUF", week=1, receiving_yards=0, rushing_tds=1.0, receptions=0, passing_yards=0),
                        dict(player_id="00-9", player_display_name="Someone", position="WR", team="DET", week=1, receiving_yards=0, rushing_tds=0, receptions=0, passing_yards=0)])
    for c in COL.values():
        if c not in act.columns: act[c] = 0.0
    g = grade(rows, act); s = summarise(g)
    assert list(g.direction.fillna("")) == ["hit", "miss", "miss", ""], list(g.direction)
    assert abs(g.removed_pts.iloc[0] - 1.0) < 1e-6, g.removed_pts.iloc[0]        # 30 -> 20 yds error = 1.0 pt closer
    assert abs(g.removed_pts.iloc[1] - (-0.48)) < 1e-6, g.removed_pts.iloc[1]     # faded a TD that happened: 0.08 TD farther = -0.48 pt
    assert g.pending.tolist() == [False, False, False, True] and g.actual.iloc[2] == 0.0   # Z Three: game ingested, no row -> 0
    assert s["by_fan"][0]["fan"] == "A" and s["overall"]["pending"] == 1
    print("selftest ok:", json.dumps(clean(s["overall"]))); sys.exit(0)

if not os.path.exists(LONG):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "overall": {"n": 0, "graded": 0, "pending": 0}, "by_fan": [], "rows": [], "note": "no submissions recorded yet"}, open(OUT, "w"), indent=1)
    print("no submissions recorded yet -> wrote an empty docs/fan/grade.json"); sys.exit(0)
rows = pd.read_csv(LONG, dtype={"player_id": str}); rows = rows[rows.season == A.season].copy()
url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{A.season}.parquet"
act = pd.read_parquet(url); act = act[act.season_type == "REG"]
for c in COL.values():
    if c not in act.columns: act[c] = 0.0
g = grade(rows, act); s = summarise(g); s["season"] = A.season
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(clean(s), open(OUT, "w"), indent=1)
o = s["overall"]
print(f"fan picks: {o['n']} arrows from {len(s['by_fan'])} fan(s); {o['graded']} graded, {o['pending']} pending"
      + (f"; direction {o['hit_rate']:.0%} ({o['hits']}-{o['misses']}), closer {o['closer']} / farther {o['farther']}, net {o['removed_pts']:+.2f} pts of error removed" if o["graded"] else ""))
for f in s["by_fan"][:10]:
    print(f"  {f['fan']:<24} graded {f['graded']:>3}  pending {f['pending']:>3}  hit {f['hit_rate'] if f['hit_rate'] is not None else '-'}  removed {f['removed_pts'] if f['removed_pts'] is not None else '-'} pts")
print(f"wrote docs/fan/grade.json")
