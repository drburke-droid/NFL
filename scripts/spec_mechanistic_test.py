"""Two follow-ups to spec_features_study.py (Sept-2026 DFS spec, free nflverse participation data):

  A. Low-capacity calibrator (spec sec. 30): walk-forward ridge on the FFA residual using the
     few route features that showed a mean gradient (route_surprise, yprr_r6, tprr_r6, vacated_rs),
     interacted with position. Does a ridge monetise what the GBM could not?
  B. Mechanistic component (spec sec. 18/25): receiving yards = team dropbacks r6 x route share r3
     x yards per route r6 (shrunk), targets = dropbacks x route share x TPRR. Compared with FFA's
     own rec_yds / rec lines and the closing DK line on 2025 player-weeks, MAE + Spearman.
Usage: python scripts/spec_mechanistic_test.py
"""
import os, re, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t, flush=True); L.append(t)

wk = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "weekly_skill_2022_2025.parquet"))
for c in ("team", "opponent_team"): wk[c] = wk[c].map(lambda t: FIX.get(t, t))
wk = wk.drop_duplicates(["player_id", "season", "week"]); wk["nname"] = wk.player_display_name.map(norm)
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
    for c in ("rec", "rec_yds", "rush_yds"):
        if c not in d.columns: d[c] = np.nan
    rows.append(d[["nname", "position", "team", "season", "week", "baseline_proj", "rec", "rec_yds"]]
                .rename(columns={"rec": "ffa_rec", "rec_yds": "ffa_rec_yds"}).drop_duplicates(["nname", "position"]))
h = pd.concat(rows).merge(ids, on=["nname", "position"], how="inner")
h = h.merge(wk[["player_id", "season", "week", "opponent_team", "fantasy_points_ppr", "receptions", "receiving_yards", "targets"]]
            .rename(columns={"fantasy_points_ppr": "actual_ppr"}), on=["player_id", "season", "week"], how="inner")
h = h.drop_duplicates(["player_id", "season", "week"])

# ---- route features (same construction as spec_features_study) ----
PD = os.path.join(ROOT, "data", "participation")
rt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "routes_*.parquet")))])
rt = rt[rt.week <= 18].rename(columns={"gsis_id": "player_id", "posteam": "team", "defteam": "opponent_team"})
for c in ("team", "opponent_team"): rt[c] = rt[c].map(lambda t: FIX.get(t, t))
rt = rt.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
g = rt.groupby("player_id", sort=False)
rt["route_share_l1"] = g.route_share.shift(1)
rt["route_share_r3"] = g.route_share.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
rt["rs_r6_prev"] = g.route_share.transform(lambda s: s.shift(2).rolling(6, min_periods=2).mean())
rt["route_surprise"] = rt.route_share_l1 - rt.rs_r6_prev
for c in ("targets_on_routes", "routes", "yds_on_tgt", "rec_on_tgt"):
    rt[c + "_s6"] = g[c].transform(lambda s: s.shift(1).rolling(6, min_periods=2).sum())
    rt[c + "_s16"] = g[c].transform(lambda s: s.shift(1).rolling(16, min_periods=4).sum())
posm = rt.groupby("pos").apply(lambda d: pd.Series({"tprr_pos": d.targets_on_routes.sum() / d.routes.sum(),
                                                     "yprr_pos": d.yds_on_tgt.sum() / d.routes.sum(),
                                                     "cprr_pos": d.rec_on_tgt.sum() / d.routes.sum()}))
rt = rt.merge(posm, left_on="pos", right_index=True, how="left")
K = 60.0
rt["tprr_r6"] = (rt.targets_on_routes_s6 + K * rt.tprr_pos) / (rt.routes_s6 + K)
rt["yprr_r6"] = (rt.yds_on_tgt_s6 + K * rt.yprr_pos) / (rt.routes_s6 + K)
rt["yprr_r16"] = (rt.yds_on_tgt_s16 + K * rt.yprr_pos) / (rt.routes_s16 + K)
rt["cprr_r16"] = (rt.rec_on_tgt_s16 + K * rt.cprr_pos) / (rt.routes_s16 + K)
rt["tprr_r16"] = (rt.targets_on_routes_s16 + K * rt.tprr_pos) / (rt.routes_s16 + K)
td = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "team_def_*.parquet")))])
td = td[td.week <= 18].rename(columns={"posteam": "team"}); td["team"] = td.team.map(lambda t: FIX.get(t, t))
td = td.sort_values(["season", "week"]).drop_duplicates(["season", "week", "team"])
td["team_db_r6"] = td.groupby(["team", "season"]).team_dropbacks.transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
td["team_db_r16"] = td.groupby("team").team_dropbacks.transform(lambda s: s.shift(1).rolling(16, min_periods=4).mean())
rt = rt.merge(td[["season", "week", "team", "team_db_r6", "team_db_r16"]], on=["season", "week", "team"], how="left")
# vacated
tw = rt[["season", "week", "team"]].drop_duplicates().sort_values(["season", "week"])
tw["prev_week"] = tw.groupby(["season", "team"]).week.shift(1)
last = rt[["season", "week", "team", "player_id", "route_share_r3", "route_share"]].rename(columns={"week": "prev_week"})
cur = rt[["season", "week", "team", "player_id"]].assign(here=1)
v = tw.merge(last, on=["season", "prev_week", "team"]).merge(cur, on=["season", "week", "team", "player_id"], how="left")
v = v[v.here.isna() & (v.route_share >= 0.3)]
vac = v.groupby(["season", "week", "team"]).route_share_r3.sum().rename("vacated_rs").reset_index()
h = h.merge(vac, on=["season", "week", "team"], how="left"); h["vacated_rs"] = h.vacated_rs.fillna(0)
h = h.merge(rt[["player_id", "season", "week", "route_share_l1", "route_share_r3", "route_surprise", "tprr_r6", "yprr_r6",
                "yprr_r16", "cprr_r16", "tprr_r16", "team_db_r6", "team_db_r16", "yprr_pos", "tprr_pos"]],
            on=["player_id", "season", "week"], how="left")
h["res"] = h.actual_ppr - h.baseline_proj; h["tk"] = h.season * 100 + h.week
say(f"# Spec follow-ups: ridge calibrator + mechanistic component test ({pd.Timestamp.now():%Y-%m-%d})\n")

# ---- A. walk-forward ridge on the residual ----
say("## A. Low-capacity ridge calibrator on the FFA residual (walk-forward, 2025)\n")
sk = h[h.position.isin(["RB", "WR", "TE"])].copy()
for c in ("route_surprise", "yprr_r6", "tprr_r6", "vacated_rs", "route_share_l1"):
    sk[c] = sk[c].fillna(sk.groupby("position")[c].transform("median")).fillna(0)
    sk[c + "_c"] = sk[c] - sk.groupby(["season", "position"])[c].transform("median")   # centred within season-position
feats = ["route_surprise_c", "yprr_r6_c", "tprr_r6_c", "vacated_rs", "route_share_l1_c"]
X = []
for p in ("RB", "WR", "TE"):
    for f in feats: sk[f + "_" + p] = sk[f] * (sk.position == p); X.append(f + "_" + p)
weeks = sorted(sk.tk.unique())
def wf_ridge(cols, alpha):
    pred = pd.Series(np.nan, index=sk.index)
    for t in weeks:
        if t < 202401: continue
        tr, te = sk[sk.tk < t], sk[sk.tk == t]
        if len(tr) < 1500 or not len(te): continue
        m = Ridge(alpha=alpha).fit(tr[cols].values, tr.res.values); pred.loc[te.index] = m.predict(te[cols].values)
    return pred
e = sk[sk.season == 2025]
b_mae, b_rmse = np.abs(e.res).mean(), np.sqrt((e.res ** 2).mean())
b_sp = e.groupby("week").apply(lambda d: spearmanr(d.baseline_proj, d.actual_ppr)[0]).mean()
say(f"FFA baseline (RB/WR/TE 2025, n={len(e):,}): MAE {b_mae:.3f} · RMSE {b_rmse:.3f} · Spearman {b_sp:.3f}\n")
say("| features | alpha | MAE | RMSE | Spearman | mean |corr| |"); say("|---|---|---|---|---|---|")
sets = {"intercept only (bias)": [], "route_surprise": [f for f in X if f.startswith("route_surprise")],
        "yprr_r6": [f for f in X if f.startswith("yprr")], "surprise + yprr": [f for f in X if f.startswith(("route_surprise", "yprr"))],
        "all five": X}
for lab, cols in sets.items():
    for alpha in (1.0, 30.0):
        if not cols and alpha > 1: continue
        cc = cols + ["one"]; sk["one"] = 1.0
        pr = wf_ridge(cc, alpha); ee = sk[pr.notna() & (sk.season == 2025)]; f = ee.baseline_proj + pr.loc[ee.index]
        mae = np.abs(ee.actual_ppr - f).mean(); rmse = np.sqrt(((ee.actual_ppr - f) ** 2).mean())
        sp = ee.assign(f=f).groupby("week").apply(lambda d: spearmanr(d.f, d.actual_ppr)[0]).mean()
        say(f"| {lab} | {alpha:g} | {mae:.3f} ({(mae-b_mae)/b_mae:+.2%}) | {rmse:.3f} ({(rmse-b_rmse)/b_rmse:+.2%}) | {sp:.3f} ({sp-b_sp:+.3f}) | {np.abs(pr.loc[ee.index]).mean():.2f} |")

# ---- B. mechanistic receiving component vs FFA vs DK line ----
say("\n## B. Mechanistic receiving component vs FFA line vs DK closing line (2025 RB/WR/TE)\n")
m = h[h.position.isin(["RB", "WR", "TE"]) & (h.season == 2025)].dropna(subset=["route_share_r3", "team_db_r6", "ffa_rec_yds"]).copy()
m = m[m.receiving_yards.notna() & (m.ffa_rec_yds > 0)]
m["mech_routes"] = m.team_db_r6 * m.route_share_r3
m["mech_rec_yds_6"] = m.mech_routes * m.yprr_r6
m["mech_rec_yds_16"] = m.mech_routes * m.yprr_r16
m["mech_targets"] = m.mech_routes * m.tprr_r16
m["mech_rec"] = m.mech_routes * m.cprr_r16
# DK closing lines from the props frames
pf = pd.read_parquet(os.path.join(ROOT, "data", "props_frames", "props_player_reception_yds.parquet"))
pf = pf[pf.season == 2025] if "season" in pf.columns else pf
say(f"props frame columns: {list(pf.columns)[:14]}")
say("\n| projection | MAE rec_yds | RMSE | Spearman (within week) | bias | n |"); say("|---|---|---|---|---|---|")
def rep(lab, col, d):
    d = d.dropna(subset=[col]); err = d[col] - d.receiving_yards
    sp = d.groupby("week").apply(lambda x: spearmanr(x[col], x.receiving_yards)[0]).mean()
    say(f"| {lab} | {np.abs(err).mean():.2f} | {np.sqrt((err**2).mean()):.2f} | {sp:.3f} | {err.mean():+.2f} | {len(d):,} |")
for lab, col in [("FFA rec_yds", "ffa_rec_yds"), ("mechanistic r6 (db×rs×yprr6)", "mech_rec_yds_6"),
                 ("mechanistic r16 (db×rs×yprr16)", "mech_rec_yds_16")]:
    rep(lab, col, m)
m["blend"] = 0.5 * m.ffa_rec_yds + 0.5 * m.mech_rec_yds_16
rep("0.5 FFA + 0.5 mechanistic r16", "blend", m)
# receptions
say("\n| projection | MAE rec | RMSE | Spearman | bias | n |"); say("|---|---|---|---|---|---|")
mr = m.dropna(subset=["ffa_rec", "receptions"]); mr = mr[mr.ffa_rec > 0]
for lab, col in [("FFA rec", "ffa_rec"), ("mechanistic r16 (db×rs×cprr16)", "mech_rec")]:
    d = mr.dropna(subset=[col]); err = d[col] - d.receptions
    sp = d.groupby("week").apply(lambda x: spearmanr(x[col], x.receptions)[0]).mean()
    say(f"| {lab} | {np.abs(err).mean():.3f} | {np.sqrt((err**2).mean()):.3f} | {sp:.3f} | {err.mean():+.3f} | {len(d):,} |")
# residual of FFA explained by mechanistic?
m["ffa_err"] = m.ffa_rec_yds - m.receiving_yards; m["gap"] = m.mech_rec_yds_16 - m.ffa_rec_yds; mm = m.dropna(subset=["gap", "ffa_err"])
say(f"\ncorr(mech − FFA, actual − FFA) = {np.corrcoef(mm.gap, -mm.ffa_err)[0,1]:+.3f}  (positive = mechanism knows something FFA missed)")
q = pd.qcut(mm.gap.rank(method="first"), 5, labels=False)
say("mean (actual − FFA rec_yds) by quintile of (mech − FFA): " + ", ".join(f"Q{i+1} {(-mm.ffa_err)[q==i].mean():+.1f}" for i in range(5)))
os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "reports", "spec_mechanistic_test.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/spec_mechanistic_test.md")
