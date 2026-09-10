"""Spec (Sept-2026 DFS projection spec) Tier-A opportunity features, tested on the SaberSim
walk-forward harness — built from the FREE nflverse participation feed, no subscription.

Same history frame as feature_candidates_study.py (FFA weekly baseline, 2023-25, Model_Burke
lags). Candidates, all from prior games only (knowable pre-kick):
  route_share_l1 / _r3 / _r6   routes / team dropbacks (A5)
  tprr_r6, yprr_r6             targets / yards per route run, prior 6, shrunk to position mean (A5/A26)
  route_surprise               route_share_l1 - prior r6; role-change signal (A10)
  route_up2                    two consecutive weeks of route share >5pts above the prior r6 (A10)
  rz_route_share_r6            red-zone targets per red-zone route, prior 6 (A13)
  opp_man_r6 / opp_press_r6    opponent defence man-coverage / pressure rate, prior 6 (A11/A12)
  man_edge                     (player TPRR vs man - vs zone, prior 6) x opp_man_r6 (A11)
  team_db_r6                   team dropbacks per game, prior 6 (A1)
  vacated_rs                   summed r3 route share of same-team players who ran routes last game
                               but are absent from THIS game (A8; lineups are known pre-kick and the
                               live generator already uses them)
Usage: python scripts/spec_features_study.py <model_burke pkg dir>
"""

import os, re, sys, glob, sqlite3, warnings
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
NAME2ABBR = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t, flush=True); L.append(t)

# ---------- base frame (mirrors sabersim_weekly.py) ----------
wk = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "weekly_skill_2022_2025.parquet"))
gl = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "game_lines_2023_2026.parquet"))
for c in ("team", "opponent_team"): wk[c] = wk[c].map(lambda t: FIX.get(t, t))
for c in ("team", "opp"): gl[c] = gl[c].map(lambda t: FIX.get(t, t))
wk = wk.drop_duplicates(["player_id", "season", "week"]); wk["nname"] = wk.player_display_name.map(norm)
lag = build_lagged_features(wk)
ids = wk.sort_values(["season", "week"]).drop_duplicates(["nname", "position"], keep="last")[["nname", "position", "player_id"]]
def score_ppr(d):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    return (z("pass_yds") * .04 + z("pass_tds") * 4 - z("pass_int") * 2 + z("rush_yds") * .1 + z("rush_tds") * 6
            + z("rec") + z("rec_yds") * .1 + z("rec_tds") * 6 - z("fumbles_lost") * 2 + z("two_pts") * 2 + z("return_tds") * 6)
rows = []
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    s_, w_ = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
    if s_ >= 2026: continue
    d = pd.read_csv(f, na_values=["NA"]); d = d[d.position.isin(["QB", "RB", "WR", "TE"])].copy()
    d["team"] = d.team.map(lambda t: FIX.get(t, t)); d["nname"] = d.player.map(norm); d["season"], d["week"] = s_, w_
    d["baseline_proj"] = score_ppr(d)
    rows.append(d[["nname", "position", "team", "season", "week", "baseline_proj"]].drop_duplicates(["nname", "position"]))
h = pd.concat(rows).merge(ids, on=["nname", "position"], how="inner")
g_ = gl.copy(); g_["home"] = np.where(g_.is_home == 1, g_.team, g_.opp)
h = h.merge(g_[["season", "week", "team", "opp", "team_spread", "game_total", "implied_team_total", "is_home", "home"]],
            on=["season", "week", "team"], how="left").rename(columns={"team_spread": "spread", "opp": "opponent_team"})
h = h.merge(lag, on=["player_id", "season", "week"], how="left")
h = h.merge(wk[["player_id", "season", "week", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual_ppr"}),
            on=["player_id", "season", "week"], how="inner")
h["wind_kn"] = 0.0; h["market_proj"] = np.nan; h["player"] = h.nname
h = h.drop_duplicates(["player_id", "season", "week"])
say(f"# Feature candidates for the SaberSim model ({pd.Timestamp.now():%Y-%m-%d})\n")
say(f"History frame: {len(h):,} player-weeks 2023-25 (FFA weekly baseline, Model_Burke lags).\n")


# ---------- spec candidates from data/participation ----------
PD = os.path.join(ROOT, "data", "participation")
rt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "routes_*.parquet")))])
rt = rt[rt.week <= 18].rename(columns={"gsis_id": "player_id", "posteam": "team", "opponent_team": "opponent_team"})
rt = rt.rename(columns={"defteam": "opponent_team"})
for c in ("team", "opponent_team"): rt[c] = rt[c].map(lambda t: FIX.get(t, t))
rt = rt.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
g = rt.groupby("player_id", sort=False)
def roll(col, n, mp=1): return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=mp).mean())
rt["route_share_l1"] = g.route_share.shift(1)
rt["route_share_r3"] = roll("route_share", 3); rt["route_share_r6"] = roll("route_share", 6, 2)
rt["rs_r6_prev"] = g.route_share.transform(lambda s: s.shift(2).rolling(6, min_periods=2).mean())
rt["route_surprise"] = rt.route_share_l1 - rt.rs_r6_prev
rt["rs_l2"] = g.route_share.shift(2); rt["rs_r6_prev2"] = g.route_share.transform(lambda s: s.shift(3).rolling(6, min_periods=2).mean())
rt["route_up2"] = ((rt.route_share_l1 > rt.rs_r6_prev + 0.05) & (rt.rs_l2 > rt.rs_r6_prev2 + 0.05)).astype(float)
for c in ("targets_on_routes", "routes", "yds_on_tgt", "rz_targets", "rz_routes", "tgt_man", "man_routes"):
    rt[c + "_s6"] = g[c].transform(lambda s: s.shift(1).rolling(6, min_periods=2).sum())
posm = rt.groupby("pos").apply(lambda d: d.targets_on_routes.sum() / d.routes.sum()).rename("tprr_pos")
rt = rt.merge(posm, left_on="pos", right_index=True, how="left")
K = 60.0   # routes of shrinkage toward the position mean
rt["tprr_r6"] = (rt.targets_on_routes_s6 + K * rt.tprr_pos) / (rt.routes_s6 + K)
rt["yprr_r6"] = rt.yds_on_tgt_s6 / (rt.routes_s6 + K)
rt["rz_route_share_r6"] = rt.rz_targets_s6 / (rt.rz_routes_s6 + 10)
tp_m = rt.tgt_man_s6 / (rt.man_routes_s6 + K)
tp_z = (rt.targets_on_routes_s6 - rt.tgt_man_s6) / (rt.routes_s6 - rt.man_routes_s6 + K)
rt["man_gap"] = tp_m - tp_z
# team / opponent context, prior 6
td = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "team_def_*.parquet")))])
td = td[td.week <= 18].rename(columns={"posteam": "team", "defteam": "opponent_team"})
for c in ("team", "opponent_team"): td[c] = td[c].map(lambda t: FIX.get(t, t))
td = td.sort_values(["season", "week"]).drop_duplicates(["season", "week", "team"])
td["team_db_r6"] = td.groupby(["team", "season"]).team_dropbacks.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
td["opp_man_r6"] = td.groupby(["opponent_team", "season"]).def_man_rate.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
td["opp_press_r6"] = td.groupby(["opponent_team", "season"]).def_pressure_rate.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
h = h.merge(td[["season", "week", "team", "team_db_r6"]], on=["season", "week", "team"], how="left")
h = h.merge(td[["season", "week", "opponent_team", "opp_man_r6", "opp_press_r6"]].drop_duplicates(["season", "week", "opponent_team"]),
            on=["season", "week", "opponent_team"], how="left")
# vacated routes: teammates who ran routes last team-game but are absent this game
tw = rt[["season", "week", "team"]].drop_duplicates().sort_values(["season", "week"])
tw["prev_week"] = tw.groupby(["season", "team"]).week.shift(1)
last = rt[["season", "week", "team", "player_id", "route_share_r3", "route_share"]].rename(columns={"week": "prev_week"})
cur = rt[["season", "week", "team", "player_id"]].assign(here=1)
v = tw.merge(last, on=["season", "prev_week", "team"]).merge(cur, on=["season", "week", "team", "player_id"], how="left")
v = v[v.here.isna() & (v.route_share >= 0.3)]
vac = v.groupby(["season", "week", "team"]).route_share_r3.sum().rename("vacated_rs").reset_index()
h = h.merge(vac, on=["season", "week", "team"], how="left"); h["vacated_rs"] = h.vacated_rs.fillna(0)
h = h.merge(rt[["player_id", "season", "week", "route_share_l1", "route_share_r3", "route_share_r6", "route_surprise", "route_up2",
                "tprr_r6", "yprr_r6", "rz_route_share_r6", "man_gap"]], on=["player_id", "season", "week"], how="left")
h["man_edge"] = h.man_gap * h.opp_man_r6
for c in ("route_surprise", "route_up2", "man_edge"): h[c] = h[c].where(h.position != "QB")

CANDS = ["route_share_l1", "route_share_r3", "route_share_r6", "tprr_r6", "yprr_r6", "route_surprise", "route_up2",
         "rz_route_share_r6", "opp_man_r6", "opp_press_r6", "man_edge", "team_db_r6", "vacated_rs"]
h["res"] = h.actual_ppr - h.baseline_proj
say("## 1. Univariate: correlation with the FFA residual (2023-25), by position\n")
say("| feature | coverage | QB | RB | WR | TE | all |"); say("|---|---|---|---|---|---|---|")
for c in CANDS:
    cov = h[c].notna().mean()
    cs = []
    for pos in ("QB", "RB", "WR", "TE", None):
        d = h if pos is None else h[h.position == pos]
        d = d.dropna(subset=[c, "res"])
        cs.append(f"{np.corrcoef(d[c], d.res)[0,1]:+.3f}" if len(d) > 100 and d[c].std() > 0 else "—")
    say(f"| {c} | {cov:.0%} | " + " | ".join(cs) + " |")
say("\n### Mean FFA residual by quintile (RB/WR/TE, 2023-25)\n")
say("| feature | Q1 | Q2 | Q3 | Q4 | Q5 | n |"); say("|---|---|---|---|---|---|---|")
sk = h[h.position.isin(["RB", "WR", "TE"])]
for c in ["route_share_l1", "tprr_r6", "yprr_r6", "route_surprise", "rz_route_share_r6", "man_edge", "opp_press_r6", "vacated_rs"]:
    d = sk.dropna(subset=[c])
    if c == "vacated_rs": d = d[d[c] > 0]
    try: q = pd.qcut(d[c].rank(method="first"), 5, labels=False)
    except Exception: continue
    m = d.groupby(q).res.mean()
    say(f"| {c} | " + " | ".join(f"{m.get(i, np.nan):+.2f}" for i in range(5)) + f" | {len(d):,} |")
say(f"\nroute_up2 flag rate {sk.route_up2.mean():.1%}; residual when flagged {sk[sk.route_up2==1].res.mean():+.2f} vs {sk[sk.route_up2==0].res.mean():+.2f} (n={int(sk.route_up2.sum())})")
# ---------- 2. direct walk-forward residual model ----------
# Model_Burke shrinks its learned correction to ~0 on the FFA baseline (k_pos ~0.05-0.3),
# so an ablation through the package cannot see any feature. Test the signal directly:
# HistGradientBoosting on (actual - baseline), refit each week on all earlier weeks,
# scored on 2024-25 as baseline + k * pred_resid for k in (0.5, 1).
from sklearn.ensemble import HistGradientBoostingRegressor
for pos_ in ("QB", "RB", "WR", "TE"): h[f"pos_{pos_}"] = (h.position == pos_).astype(float)
base_feats = [c for c in auto_features(h) if c != "week" and pd.api.types.is_numeric_dtype(h[c])] + [f"pos_{p_}" for p_ in ("QB", "RB", "WR", "TE")]
h["tk"] = h.season * 100 + h.week; h["res"] = h.actual_ppr - h.baseline_proj
weeks = sorted(h.tk.unique())
def wf_resid(cols):
    pred = pd.Series(np.nan, index=h.index)
    for t in weeks:
        if t < 202401: continue
        tr, te = h[h.tk < t], h[h.tk == t]
        if len(tr) < 2000 or not len(te): continue
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=40, random_state=0)
        m.fit(tr[cols].values, tr.res.values); pred.loc[te.index] = m.predict(te[cols].values)
    return pred
def score_direct(pred):
    e = h[pred.notna() & (h.season == 2025)]; pr = pred.loc[e.index]
    out = {}
    for k in (0.5, 1.0):
        f = e.baseline_proj + k * pr
        out[k] = (np.abs(e.actual_ppr - f).mean(), np.sqrt(((e.actual_ppr - f) ** 2).mean()),
                  e.assign(f=f).groupby("week").apply(lambda d: spearmanr(d.f, d.actual_ppr)[0]).mean())
    return out, len(e)
say("\n## 2. Direct walk-forward residual GBM (2025; baseline + k*pred_resid)\n")
say("The shipped package shrinks its correction to ~0 on this baseline, so this is the honest sensitivity test.\n")
e25 = h[h.season == 2025]
say(f"FFA baseline alone: MAE {np.abs(e25.res).mean():.3f} · RMSE {np.sqrt((e25.res**2).mean()):.3f} · "
    f"Spearman {e25.groupby('week').apply(lambda d: spearmanr(d.baseline_proj, d.actual_ppr)[0]).mean():.3f}\n")
say("| features | k | MAE | RMSE | weekly Spearman |"); say("|---|---|---|---|---|")
p0 = wf_resid(base_feats); s0, n = score_direct(p0)
for k, (mae, rmse, sp) in s0.items(): say(f"| base lags+context | {k} | {mae:.3f} | {rmse:.3f} | {sp:.3f} |")
groups = {"route share": ["route_share_l1", "route_share_r3", "route_share_r6"], "tprr/yprr": ["tprr_r6", "yprr_r6"],
          "usage surprise": ["route_surprise", "route_up2"], "rz routes": ["rz_route_share_r6"],
          "opp man/pressure": ["opp_man_r6", "opp_press_r6"], "man edge": ["man_edge"], "team dropbacks": ["team_db_r6"],
          "vacated routes": ["vacated_rs"], "ALL spec": CANDS}
res = {}
for lab, cols in groups.items():
    pr = wf_resid(base_feats + cols); sc, _ = score_direct(pr); res[lab] = sc
    for k, (mae, rmse, sp) in sc.items():
        b = s0[k]
        say(f"| + {lab} | {k} | {mae:.3f} ({(mae-b[0])/b[0]:+.2%}) | {rmse:.3f} ({(rmse-b[1])/b[1]:+.2%}) | {sp:.3f} ({sp-b[2]:+.3f}) |")
keep = [lab for lab, sc in res.items() if sc[1.0][1] < s0[1.0][1] and sc[1.0][2] >= s0[1.0][2]]
if keep:
    cols = sum((groups[k] for k in keep), [])
    pr = wf_resid(base_feats + cols); sc, _ = score_direct(pr)
    for k, (mae, rmse, sp) in sc.items():
        b = s0[k]
        say(f"| + all that helped ({', '.join(keep)}) | {k} | {mae:.3f} ({(mae-b[0])/b[0]:+.2%}) | {rmse:.3f} ({(rmse-b[1])/b[1]:+.2%}) | {sp:.3f} ({sp-b[2]:+.3f}) |")
say(f"\n(n = {n:,} 2025 player-weeks)")
os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "reports", "spec_features_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/spec_features_study.md")
