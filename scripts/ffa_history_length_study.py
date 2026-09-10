"""Does the 2016-25 FFA weekly history (Downloads/ffa_weekly_raw_2015_2025.csv, split into
data/ffanalytics/FFAn_weekly/raw_stats_S_wkW.csv on 2026-09-10) improve the SaberSim model?

  1. FFA weekly baseline scorecard by season 2016-25 (skill positions, PPR): MAE, RMSE,
     weekly Spearman, bias; week-1 / week-2 bias by position and season (the 2024-vs-2025
     sign flip question, now with ten seasons).
  2. Model_Burke walk-forward (the shipped package) trained from 2016 vs from 2023, scored on
     2024 and 2025 by season: point MAE/RMSE vs baseline and control_k0, 80% coverage.
  3. Direct residual GBM (feature_candidates_study harness) with training start 2016 vs 2023.
Usage: python scripts/ffa_history_length_study.py <model_burke pkg dir>
"""
import os, re, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_BURKE_PKG", "")
sys.path.insert(0, PKG)
from model_burke import pipeline
from model_burke.features import build_lagged_features
from model_burke.residual import auto_features
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t, flush=True); L.append(t)

wk = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "weekly_skill_2015_2025.parquet"))
gl = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "game_lines_2015_2026.parquet"))
for c in ("team", "opponent_team"): wk[c] = wk[c].map(lambda t: FIX.get(t, t))
for c in ("team", "opp"): gl[c] = gl[c].map(lambda t: FIX.get(t, t))
wk = wk.drop_duplicates(["player_id", "season", "week"]); wk["nname"] = wk.player_display_name.map(norm)
lag = build_lagged_features(wk)
# name -> id, resolved per season so name reuse across eras doesn't cross-wire (keep latest id within season)
ids = wk.sort_values(["season", "week"]).drop_duplicates(["nname", "position", "season"], keep="last")[["nname", "position", "season", "player_id"]]
def score_ppr(d):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    return (z("pass_yds") * .04 + z("pass_tds") * 4 - z("pass_int") * 2 + z("rush_yds") * .1 + z("rush_tds") * 6
            + z("rec") + z("rec_yds") * .1 + z("rec_tds") * 6 - z("fumbles_lost") * 2 + z("two_pts") * 2 + z("return_tds") * 6)
rows = []
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    s_, w_ = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
    if s_ >= 2026 or s_ < 2016: continue
    d = pd.read_csv(f, na_values=["NA"], low_memory=False); d = d[d.position.isin(["QB", "RB", "WR", "TE"])].copy()
    d["team"] = d.team.map(lambda t: FIX.get(t, t)); d["nname"] = d.player.map(norm); d["season"], d["week"] = s_, w_
    d["baseline_proj"] = score_ppr(d)
    rows.append(d[["nname", "position", "team", "season", "week", "baseline_proj"]].drop_duplicates(["nname", "position"]))
h = pd.concat(rows).merge(ids, on=["nname", "position", "season"], how="inner")
g_ = gl.copy(); g_["home"] = np.where(g_.is_home == 1, g_.team, g_.opp)
h = h.merge(g_[["season", "week", "team", "opp", "team_spread", "game_total", "implied_team_total", "is_home", "home"]],
            on=["season", "week", "team"], how="left").rename(columns={"team_spread": "spread", "opp": "opponent_team"})
h = h.merge(lag, on=["player_id", "season", "week"], how="left")
h = h.merge(wk[["player_id", "season", "week", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual_ppr"}),
            on=["player_id", "season", "week"], how="inner")
h["wind_kn"] = 0.0; h["market_proj"] = np.nan; h["player"] = h.nname
h = h.drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
h["res"] = h.actual_ppr - h.baseline_proj
say(f"# FFA weekly history 2016-25: baseline scorecard + does longer training help Model_Burke? ({pd.Timestamp.now():%Y-%m-%d})\n")
say(f"History frame: {len(h):,} player-weeks, {h.season.min()}-{h.season.max()} (FFA weekly baseline x nflverse actuals).\n")

# ---------- 1. baseline scorecard ----------
say("## 1. FFA weekly baseline by season (QB/RB/WR/TE, PPR, all projected players with a box score)\n")
say("| season | n | MAE | RMSE | weekly Spearman | bias (actual − FFA) | n/week |"); say("|---|---|---|---|---|---|---|")
for s, d in h.groupby("season"):
    sp = d.groupby("week").apply(lambda x: spearmanr(x.baseline_proj, x.actual_ppr)[0]).mean()
    say(f"| {s} | {len(d):,} | {np.abs(d.res).mean():.3f} | {np.sqrt((d.res**2).mean()):.3f} | {sp:.3f} | {d.res.mean():+.2f} | {len(d)/d.week.nunique():.0f} |")
say("\n### Week-1 bias (actual − FFA) by position and season, players projected ≥ 5\n")
say("| season | QB | RB | WR | TE | all |"); say("|---|---|---|---|---|---|")
w1 = h[(h.week == 1) & (h.baseline_proj >= 5)]
for s, d in w1.groupby("season"):
    cells = [f"{d[d.position==p].res.mean():+.2f} (n={len(d[d.position==p])})" for p in ("QB", "RB", "WR", "TE")]
    say(f"| {s} | " + " | ".join(cells) + f" | {d.res.mean():+.2f} |")
say(f"| **pooled** | " + " | ".join(f"{w1[w1.position==p].res.mean():+.2f}" for p in ("QB", "RB", "WR", "TE")) + f" | {w1.res.mean():+.2f} |")
say("\n### Bias by week band (players projected ≥ 5), pooled 2016-25\n")
say("| weeks | QB | RB | WR | TE | all | n |"); say("|---|---|---|---|---|---|---|")
for lab, lo, hi in (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-9", 4, 9), ("10-18", 10, 18)):
    d = h[(h.week >= lo) & (h.week <= hi) & (h.baseline_proj >= 5)]
    say(f"| {lab} | " + " | ".join(f"{d[d.position==p].res.mean():+.2f}" for p in ("QB", "RB", "WR", "TE")) + f" | {d.res.mean():+.2f} | {len(d):,} |")

# ---------- 2. Model_Burke with long vs short history ----------
say("\n## 2. Model_Burke walk-forward: training history from 2016 vs from 2023\n")
say("Same package, same features; only the input frame differs. Scored by season on the rows each run reaches.\n")
res = {}
for lab, start in (("from 2016", 2016), ("from 2023", 2023)):
    df = h[h.season >= start].copy()
    ev, rep = pipeline.run(df, verbose=False)
    bs = rep["by_season"]; res[lab] = (ev, rep, bs)
    say(f"### {lab}: {rep['n_scored']:,} scored rows; shrinkage k by position = "
        + ", ".join(f"{k} {v:.2f}" for k, v in rep["shrinkage"].items()) if isinstance(rep["shrinkage"], dict) else f"### {lab}: {rep['n_scored']:,} scored rows")
    say(""); say(bs.round(4).to_string(index=False)); say("")
say("### Head-to-head on 2024 and 2025 (identical rows)\n")
say("| season | history | MAE Model_Burke | MAE control_k0 | MAE baseline | RMSE Model_Burke | 80% coverage | pinball |"); say("|---|---|---|---|---|---|---|---|")
for s in (2024, 2025):
    ee = {}
    for lab in res:
        ev = res[lab][0]; e = ev[ev.season == s].copy(); e["key"] = e.player_id.astype(str) + "_" + e.week.astype(str); ee[lab] = e
    ks = set(ee["from 2016"].key) & set(ee["from 2023"].key)
    for lab, e in ee.items():
        e = e[e.key.isin(ks)]
        mae = lambda c: np.abs(e.actual_ppr - e[c]).mean()
        eq = e.dropna(subset=["mb_p10", "mb_p90"])
        cov = ((eq.actual_ppr >= eq.mb_p10) & (eq.actual_ppr <= eq.mb_p90)).mean() if len(eq) else np.nan
        pin = np.mean([np.mean(np.maximum(q * (eq.actual_ppr - eq[f"mb_p{int(q*100)}"]), (q - 1) * (eq.actual_ppr - eq[f"mb_p{int(q*100)}"])))
                       for q in (0.1, 0.25, 0.5, 0.75, 0.9)]) if len(eq) else np.nan
        say(f"| {s} | {lab} | {mae('Model_Burke'):.3f} | {mae('control_k0'):.3f} | {mae('baseline_proj'):.3f} | "
            f"{np.sqrt(((e.actual_ppr - e.Model_Burke)**2).mean()):.3f} | {cov:.3f} | {pin:.3f} |")
    say(f"| | | (n = {len(ks):,} common rows) | | | | | |")

# ---------- 3. direct residual GBM, long vs short training ----------
say("\n## 3. Direct residual GBM (harness), training start 2016 vs 2023, scored 2025\n")
from sklearn.ensemble import HistGradientBoostingRegressor
for pos_ in ("QB", "RB", "WR", "TE"): h[f"pos_{pos_}"] = (h.position == pos_).astype(float)
feats = [c for c in auto_features(h) if c != "week" and pd.api.types.is_numeric_dtype(h[c])] + [f"pos_{p_}" for p_ in ("QB", "RB", "WR", "TE")]
h["tk"] = h.season * 100 + h.week; weeks = sorted(h.tk.unique())
def wf(start, cols):
    pred = pd.Series(np.nan, index=h.index)
    for t in weeks:
        if t < 202501: continue
        tr = h[(h.tk < t) & (h.season >= start)]; te = h[h.tk == t]
        if len(tr) < 2000 or not len(te): continue
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=40, random_state=0)
        m.fit(tr[cols].values, tr.res.values); pred.loc[te.index] = m.predict(te[cols].values)
    return pred
e25 = h[h.season == 2025]
say(f"FFA baseline 2025: MAE {np.abs(e25.res).mean():.3f} · RMSE {np.sqrt((e25.res**2).mean()):.3f}\n")
say("| training from | k | MAE | RMSE | weekly Spearman |"); say("|---|---|---|---|---|")
for start in (2016, 2020, 2023):
    pr = wf(start, feats); e = h[pr.notna() & (h.season == 2025)]
    for k in (0.5, 1.0):
        f = e.baseline_proj + k * pr.loc[e.index]
        sp = e.assign(f=f).groupby("week").apply(lambda d: spearmanr(d.f, d.actual_ppr)[0]).mean()
        say(f"| {start} | {k} | {np.abs(e.actual_ppr - f).mean():.3f} | {np.sqrt(((e.actual_ppr - f)**2).mean()):.3f} | {sp:.3f} |")
os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "reports", "ffa_history_length_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/ffa_history_length_study.md")
