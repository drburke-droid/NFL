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
g = elig.merge(act[["player_id", "week", "actual"]].rename(columns={"player_id": "ID"}), on=["ID", "week"], how="left")
miss = g.actual.isna() & g.Pos.isin(["QB", "RB", "WR", "TE", "K"])
if miss.any():   # name fallback (no gsis id on the K rows, or an id mismatch)
    fb = g[miss].drop(columns=["actual"]).merge(act[["nname", "position", "week", "actual"]].rename(columns={"position": "Pos"}), on=["nname", "Pos", "week"], how="left")
    g.loc[miss, "actual"] = fb.actual.values
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
out = {"generated_at": datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%MZ"), "season": A.season, "min_lead_min": A.min_lead,
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
L += ["", "Largest misses:", ""] + [f"- wk{m['week']} {m['player']} ({m['pos']} {m['team']}): proj {m['proj']}, actual {m['actual']}" for m in out["misses"]]
open(os.path.join(ROOT, "outputs", "reports", "sabersim_accuracy.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:12])); print(f"\nwrote docs/sabersim_accuracy.json ({overall['n']} graded rows, {len(weeks)} week(s))")
