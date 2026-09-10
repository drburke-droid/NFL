"""Grade the SaberSim sends against actual box scores.

Eligibility: a send counts only if it was generated at least 75 minutes before that game's kickoff
(SaberSim's deadline). For each (game, player) the LATEST eligible send is the one graded, so a
Wednesday-night send that was superseded by a Sunday-morning send is not double counted.

Inputs
  --sends DIR [DIR ...]   folders holding Burke_Model_Burke_*.csv (+ the run .parquet the wrapper
                          publishes next to each CSV: FFA baseline / DK market for the benchmarks)
  --season 2026
Actuals: nflverse stats_player_week_{season}.parquet. QB/RB/WR/TE = fantasy_points_ppr (the target
the model was trained on: 4-pt pass TD, -2 INT, PPR, -2 fumble lost). K = DK kicker scoring from
fg_made_* and pat_made. DST is not graded (no team-defense rows in that file).

Outputs
  docs/sabersim_accuracy.json          what the SaberSim page renders
  outputs/reports/sabersim_accuracy.md the same, readable
Usage: python scripts/sabersim_grade.py --sends outputs/sabersim pkg/sends
"""
import os, re, sys, glob, json, argparse
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET = ZoneInfo("America/New_York")
ap = argparse.ArgumentParser()
ap.add_argument("--sends", nargs="+", default=[os.path.join(ROOT, "outputs", "sabersim")])
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--week1-tuesday", default="2026-09-08", help="Tuesday that starts week 1 (weeks roll on Tuesdays)")
ap.add_argument("--min-lead", type=float, default=75.0, help="minutes before kickoff a send must be generated to count")
A = ap.parse_args()
W1 = date.fromisoformat(A.week1_tuesday)
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)

# ---------- 1. sends ----------
frames = []
for d in A.sends:
    for f in sorted(glob.glob(os.path.join(d, "Burke_Model_Burke_*.csv"))):
        try: x = pd.read_csv(f)
        except Exception: continue
        if "Generated" not in x.columns or "Kickoff" not in x.columns: continue
        x["send_file"] = os.path.basename(f); x["send_dir"] = d; frames.append(x)
if not frames: raise SystemExit("no sends found")
s = pd.concat(frames, ignore_index=True)
s["gen"] = pd.to_datetime(s.Generated.str.replace(" ET", "", regex=False), format="%Y-%m-%d %H:%M", errors="coerce").dt.tz_localize(ET)
def kick(row):
    m = re.match(r"^\w{3} (\d{2})/(\d{2}) (\d{2}):(\d{2}) (AM|PM) ET$", str(row.Kickoff))
    if not m or pd.isna(row.gen): return pd.NaT
    mo, dd, hh, mi, ap_ = int(m[1]), int(m[2]), int(m[3]) % 12, int(m[4]), m[5]
    yr = row.gen.year + (1 if (mo < row.gen.month - 6) else 0)
    return pd.Timestamp(yr, mo, dd, hh + (12 if ap_ == "PM" else 0), mi, tz=ET)
s["kick"] = s.apply(kick, axis=1)
s = s.dropna(subset=["gen", "kick"])
s["lead_min"] = (s.kick - s.gen).dt.total_seconds() / 60
s["week"] = ((s.kick.dt.tz_convert(ET).dt.date - W1).map(lambda t: t.days) // 7 + 1).astype(int)
s["eligible"] = s.lead_min >= A.min_lead
elig = s[s.eligible].sort_values("gen").drop_duplicates(["Game", "ID", "Player", "Pos"], keep="last").copy()
late = s[~s.eligible].groupby("week").send_file.nunique().to_dict()
elig = elig[elig.kick < pd.Timestamp.now(tz=ET)]                     # games that have kicked off
if elig.empty: raise SystemExit("no eligible sends for games that have kicked off")

# ---------- 2. actuals ----------
url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{A.season}.parquet"
a = pd.read_parquet(url)
a = a[a.season_type == "REG"].copy()
a["nname"] = a.player_display_name.map(norm)
z = lambda c: a[c].fillna(0) if c in a.columns else 0
a["k_pts"] = 3 * (z("fg_made_0_19") + z("fg_made_20_29") + z("fg_made_30_39")) + 4 * z("fg_made_40_49") + 5 * (z("fg_made_50_59") + z("fg_made_60_")) + z("pat_made")
a["actual"] = np.where(a.position == "K", a.k_pts, a.fantasy_points_ppr)
act = a[["player_id", "nname", "position", "week", "actual"]]
elig["nname"] = elig.Player.map(norm)
act = act.assign(team=a.team.values, last=a.player_display_name.map(lambda n: norm(n).split()[-1]))
by_id = {(r.player_id, int(r.week)): r.actual for r in act.itertuples() if isinstance(r.player_id, str)}
by_name = act.groupby(["nname", "position", "week"]).actual.sum().to_dict()
lt = act.groupby(["last", "team", "position", "week"]).actual.agg(["sum", "size"])
by_last = {k: v for k, v in lt["sum"].items() if lt.loc[k, "size"] == 1}      # only when unambiguous
def lookup(r):
    w = int(r.week)
    if isinstance(r.ID, str) and (r.ID, w) in by_id: return by_id[(r.ID, w)]
    if (r.nname, r.Pos, w) in by_name: return by_name[(r.nname, r.Pos, w)]
    return by_last.get((r.nname.split()[-1], r.Team, r.Pos, w), np.nan)
g = elig.copy(); g["actual"] = [lookup(r) for r in g.itertuples()]
g = g[g.Pos.isin(["QB", "RB", "WR", "TE", "K"])].copy()
# a player with no box-score row after the game = 0 points (inactive / no touches) — but only for
# games whose week has any actuals at all (nflverse may not have published yet)
have = set(act.week.unique())
g = g[g.week.isin(have)].copy(); g["actual"] = g.actual.fillna(0.0)
g["err"] = g.actual - g.Proj

# ---------- 3. benchmarks from the run parquets (FFA baseline, DK market) ----------
bench = []
for d in A.sends:
    for f in glob.glob(os.path.join(d, "Burke_Model_Burke_*.parquet")):
        try: r = pd.read_parquet(f, columns=["player_id", "season", "week", "baseline_proj", "market_proj", "Model_Burke"])
        except Exception: continue
        r = r[(r.season == A.season)]; r["send_file"] = os.path.basename(f).replace(".parquet", ".csv"); bench.append(r)
if bench:
    b = pd.concat(bench).drop_duplicates(["send_file", "player_id"]).rename(columns={"player_id": "ID", "baseline_proj": "ffa", "market_proj": "dk"})
    g = g.merge(b[["send_file", "ID", "ffa", "dk"]], on=["send_file", "ID"], how="left")
else:
    g["ffa"] = np.nan; g["dk"] = np.nan

# ---------- 4. metrics ----------
from scipy.stats import spearmanr
def block(d):
    d = d.dropna(subset=["actual"])
    if d.empty: return None
    o = {"n": int(len(d)), "mae": round(float(d.err.abs().mean()), 3), "rmse": round(float(np.sqrt((d.err ** 2).mean())), 3),
         "bias": round(float(d.err.mean()), 3),
         "spearman": round(float(spearmanr(d.Proj, d.actual)[0]), 3) if len(d) >= 8 else None,
         "cov80": round(float(((d.actual >= d.Floor_p10) & (d.actual <= d.Ceiling_p90)).mean()), 3) if "Floor_p10" in d else None,
         "median_mae": round(float((d.actual - d.Median).abs().mean()), 3) if "Median" in d else None}
    for lab, col in (("ffa", "ffa"), ("dk", "dk")):
        dd = d.dropna(subset=[col])
        if len(dd) >= 8:
            o[f"{lab}_n"] = int(len(dd)); o[f"{lab}_mae"] = round(float((dd.actual - dd[col]).abs().mean()), 3)
            o[f"model_mae_on_{lab}_rows"] = round(float(dd.err.abs().mean()), 3)
    return o
weeks = []
for wk, d in g.groupby("week"):
    o = block(d); o.update({"week": int(wk), "sends": int(d.send_file.nunique()), "games": int(d.Game.nunique()),
                            "late_sends_ignored": int(late.get(wk, 0)), "by_pos": {p: block(x) for p, x in d.groupby("Pos")}})
    weeks.append(o)
overall = block(g); overall["by_pos"] = {p: block(x) for p, x in g.groupby("Pos")}
misses = g.reindex(g.err.abs().sort_values(ascending=False).index).head(12)
# ---------- 5. scale: what the numbers mean (measured on the 2025 season, ~300 FFA-projected QB/RB/WR/TE per week) ----------
SCALE = {
  "note": ("Bands are for the full slate pool (every projected skill player, ~10 per team). MAE is dominated by outcome "
           "noise: a hindsight oracle that knows each player's true season average still scores MAE 4.04 on this pool, "
           "so ~4.0 is the floor and 3.0 is not attainable. Starters-only pools run 1.5-2 points higher (FFA 5.95 on "
           "players projected 8+). A single 30-player slate has an MAE standard deviation of ~0.8, so judge on 5+ weeks "
           "(~1,500 player-games) and on the same-rows comparison with FFA and DraftKings."),
  "reference_2025": {"previous game points": {"mae": 5.73, "rmse": 8.25, "spearman": 0.532}, "trailing 4-game mean": {"mae": 4.78, "rmse": 6.65, "spearman": 0.625},
                     "season-to-date mean": {"mae": 4.67, "rmse": 6.58, "spearman": 0.641}, "DK market-implied (rows with a line)": {"mae": 4.74, "rmse": 6.40, "spearman": 0.608},
                     "FFA consensus": {"mae": 4.17, "rmse": 5.85, "spearman": 0.719}, "Model_Burke (walk-forward)": {"mae": 4.12, "rmse": 5.90, "spearman": 0.714},
                     "oracle: true season mean, hindsight": {"mae": 4.04, "rmse": 5.65, "spearman": 0.737}},
  "bands": {   # lower is better unless noted; thresholds = upper edge of each band
    "mae":      {"elite": 4.05, "top": 4.15, "consensus": 4.30, "fair": 4.80, "poor": 99},
    "rmse":     {"elite": 5.65, "top": 5.85, "consensus": 6.05, "fair": 6.70, "poor": 99},
    "spearman": {"higher": True, "elite": 0.74, "top": 0.72, "consensus": 0.69, "fair": 0.60, "poor": -1},
    "abs_bias": {"elite": 0.15, "top": 0.30, "consensus": 0.50, "fair": 0.80, "poor": 99},
    "cov80":    {"target": 0.80, "elite": 0.02, "top": 0.04, "consensus": 0.06, "fair": 0.10, "poor": 1},   # |coverage - 0.80|
    "vs_ffa_ratio": {"elite": 0.97, "top": 0.99, "consensus": 1.01, "fair": 1.04, "poor": 99}                # model MAE / FFA MAE on the same rows
  },
  "labels": {"elite": "elite (at the noise floor)", "top": "top tier (beats the consensus)", "consensus": "consensus-grade (FFA / market blend)",
             "fair": "fair (trailing averages)", "poor": "poor"}
}
out = {"generated_at": datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%MZ"), "season": A.season, "min_lead_min": A.min_lead, "scale": SCALE,
       "rule": "latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded",
       "weeks": weeks, "overall": overall,
       "misses": [{"week": int(r.week), "player": r.Player, "pos": r.Pos, "team": r.Team, "proj": round(float(r.Proj), 1), "actual": round(float(r.actual), 1)} for r in misses.itertuples()]}
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True); os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "docs", "sabersim_accuracy.json"), "w"), indent=1)
L = [f"# SaberSim send accuracy — {A.season} (graded {out['generated_at']})", "", out["rule"], "",
     "| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for w in weeks:
    L.append(f"| {w['week']} | {w['sends']} | {w['games']} | {w['n']} | {w['mae']} | {w['rmse']} | {w['bias']:+} | {w['spearman']} | {w['cov80']} | "
             f"{w.get('ffa_mae', '—')}{' (model ' + str(w['model_mae_on_ffa_rows']) + ')' if 'ffa_mae' in w else ''} | {w.get('dk_mae', '—')}{' (model ' + str(w['model_mae_on_dk_rows']) + ')' if 'dk_mae' in w else ''} |")
L += ["", f"**Overall:** n {overall['n']} · MAE {overall['mae']} · RMSE {overall['rmse']} · bias {overall['bias']:+} · Spearman {overall['spearman']} · 80% coverage {overall['cov80']}", "",
      "| pos | n | MAE | RMSE | bias | 80% cov |", "|---|---|---|---|---|---|"]
for p, o in overall["by_pos"].items(): L.append(f"| {p} | {o['n']} | {o['mae']} | {o['rmse']} | {o['bias']:+} | {o['cov80']} |")
L += ["", "## Scale (2025 reference, full slate pool)", "", SCALE["note"], "", "| projection | MAE | RMSE | Spearman |", "|---|---|---|---|"]
L += [f"| {k} | {v['mae']} | {v['rmse']} | {v['spearman']} |" for k, v in SCALE["reference_2025"].items()]
L += ["", "Bands (upper edge): MAE elite ≤4.05 · top ≤4.15 · consensus ≤4.30 · fair ≤4.80 · poor above. RMSE 5.65/5.85/6.05/6.70. "
      "Spearman ≥0.74/0.72/0.69/0.60. |bias| ≤0.15/0.30/0.50/0.80. 80% coverage within ±0.02/0.04/0.06/0.10 of 0.80. "
      "Model÷FFA MAE on the same rows ≤0.97/0.99/1.01/1.04."]
L += ["", "Largest misses:", ""] + [f"- wk{m['week']} {m['player']} ({m['pos']} {m['team']}): proj {m['proj']}, actual {m['actual']}" for m in out["misses"]]
open(os.path.join(ROOT, "outputs", "reports", "sabersim_accuracy.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:12])); print(f"\nwrote docs/sabersim_accuracy.json ({overall['n']} graded rows, {len(weeks)} week(s))")
