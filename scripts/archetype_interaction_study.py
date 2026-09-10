"""QB / WR archetype x matchup interactions, tested on the SaberSim walk-forward harness
(FFA weekly baseline, 2023-25, Model_Burke lags). All inputs are prior games only.

QB archetype (prior 6, nflv_pbp_skill_wk): aDOT, sack rate, scramble rate; team time to throw
(participation). WR/TE archetype: deep-target share (air yards >= 15, prior 6), NGS prior-season
separation / cushion, TPRR under pressure vs clean. RB: TPRR. Opponent (prior 6, z within week):
pressure rate, man rate, two-high rate, box count. Interactions are centred-archetype x opponent-z.
Usage: python scripts/archetype_interaction_study.py <model_burke pkg dir>
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



# ---------- archetype x matchup interactions ----------
PD = os.path.join(ROOT, "data", "participation")
rt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "routes_*.parquet")))])
rt = rt[rt.week <= 18].rename(columns={"gsis_id": "player_id", "posteam": "team", "defteam": "opponent_team"})
for c in ("team", "opponent_team"): rt[c] = rt[c].map(lambda t: FIX.get(t, t))
rt = rt.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
g = rt.groupby("player_id", sort=False)
for c in ("targets_on_routes", "routes", "deep_targets", "tgt_press", "press_routes", "tgt_man", "man_routes"):
    rt[c + "_s6"] = g[c].transform(lambda s: s.shift(1).rolling(6, min_periods=2).sum())
rt["deep_share_r6"] = rt.deep_targets_s6 / (rt.targets_on_routes_s6 + 5)          # deep-target archetype (air yards >= 15)
rt["tprr_r6"] = (rt.targets_on_routes_s6 + 8) / (rt.routes_s6 + 60)
rt["tprr_press_gap"] = rt.tgt_press_s6 / (rt.press_routes_s6 + 30) - (rt.targets_on_routes_s6 - rt.tgt_press_s6) / (rt.routes_s6 - rt.press_routes_s6 + 30)
# team-game context (prior 6): offence time-to-throw; defence pressure / man / two-high / box
td = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "team_def_*.parquet")))])
td = td[td.week <= 18].rename(columns={"posteam": "team", "defteam": "opponent_team"})
for c in ("team", "opponent_team"): td[c] = td[c].map(lambda t: FIX.get(t, t))
td = td.sort_values(["season", "week"]).drop_duplicates(["season", "week", "team"])
td["off_ttt_r6"] = td.groupby(["team", "season"]).ttt_mean.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
for c, src in (("opp_press_r6", "def_pressure_rate"), ("opp_man_r6", "def_man_rate"), ("opp_two_high_r6", "def_two_high_rate"), ("opp_box_r6", "box_mean")):
    td[c] = td.groupby(["opponent_team", "season"])[src].transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    lg = td.groupby(["season", "week"])[c].agg(["mean", "std"]).reset_index()
    td = td.merge(lg, on=["season", "week"]); td[c + "_z"] = (td[c] - td["mean"]) / td["std"]; td = td.drop(columns=["mean", "std"])
h = h.merge(td[["season", "week", "team", "off_ttt_r6"]], on=["season", "week", "team"], how="left")
h = h.merge(td[["season", "week", "opponent_team", "opp_press_r6_z", "opp_man_r6_z", "opp_two_high_r6_z", "opp_box_r6_z"]]
            .drop_duplicates(["season", "week", "opponent_team"]), on=["season", "week", "opponent_team"], how="left")
# QB archetype: prior-6 aDOT, sack rate, scramble rate (nflv_pbp_skill_wk), attached to the team via last game's QB1
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
qb = pd.read_sql("SELECT gsis_id player_id, season, week, adot, sack_rate, scramble_rate, cpoe FROM nflv_pbp_skill_wk WHERE role='pass' AND season BETWEEN 2022 AND 2025", con)
con.close()
qb = qb.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"])
gq = qb.groupby("player_id", sort=False)
for c in ("adot", "sack_rate", "scramble_rate", "cpoe"):
    qb[f"qb_{c}_r6"] = gq[c].transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
tgames = wk[["season", "week", "team"]].drop_duplicates().sort_values(["season", "week"])
tgames["prev_week"] = tgames.groupby(["season", "team"]).week.shift(1)
lead = wk[wk.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates(["season", "week", "team"])
lead = lead[lead.attempts >= 10][["season", "week", "team", "player_id"]].rename(columns={"week": "prev_week", "player_id": "qb1"})
tq = tgames.merge(lead, on=["season", "prev_week", "team"], how="inner")
tq = tq.merge(qb[["player_id", "season", "week", "qb_adot_r6", "qb_sack_rate_r6", "qb_scramble_rate_r6", "qb_cpoe_r6"]]
              .rename(columns={"player_id": "qb1"}), on=["qb1", "season", "week"], how="left")
h = h.merge(tq[["season", "week", "team", "qb_adot_r6", "qb_sack_rate_r6", "qb_scramble_rate_r6", "qb_cpoe_r6"]], on=["season", "week", "team"], how="left")
# WR separation archetype: NGS prior-season avg separation / cushion
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
ngs = pd.read_sql("SELECT player_gsis_id player_id, season, week, avg_separation, avg_cushion FROM nflv_ngs_receiving WHERE season BETWEEN 2021 AND 2025 AND week > 0", con)
con.close()
sep = ngs.groupby(["player_id", "season"])[["avg_separation", "avg_cushion"]].mean().reset_index(); sep["season"] += 1
h = h.merge(sep.rename(columns={"avg_separation": "sep_prev", "avg_cushion": "cushion_prev"}), on=["player_id", "season"], how="left")
h = h.merge(rt[["player_id", "season", "week", "deep_share_r6", "tprr_r6", "tprr_press_gap"]], on=["player_id", "season", "week"], how="left")
# centre archetype variables within position so interactions are signed
for c in ("off_ttt_r6", "qb_adot_r6", "qb_sack_rate_r6", "qb_scramble_rate_r6", "deep_share_r6", "sep_prev", "cushion_prev", "tprr_r6"):
    h[c + "_c"] = h[c] - h.groupby(["season", "position"])[c].transform("median")
isqb = h.position == "QB"; isrec = h.position.isin(["WR", "TE"]); isrb = h.position == "RB"
h["qb_ttt_x_press"] = (h.off_ttt_r6_c * h.opp_press_r6_z).where(isqb)
h["qb_sack_x_press"] = (h.qb_sack_rate_r6_c * h.opp_press_r6_z).where(isqb)
h["qb_scr_x_man"] = (h.qb_scramble_rate_r6_c * h.opp_man_r6_z).where(isqb)
h["qb_adot_x_twohigh"] = (h.qb_adot_r6_c * h.opp_two_high_r6_z).where(isqb)
h["qb_adot_x_press"] = (h.qb_adot_r6_c * h.opp_press_r6_z).where(isqb)
h["wr_deep_x_twohigh"] = (h.deep_share_r6_c * h.opp_two_high_r6_z).where(isrec)
h["wr_deep_x_press"] = (h.deep_share_r6_c * h.opp_press_r6_z).where(isrec)
h["wr_deep_x_qbadot"] = (h.deep_share_r6_c * h.qb_adot_r6_c).where(isrec)
h["wr_sep_x_man"] = (h.sep_prev_c * h.opp_man_r6_z).where(isrec)
h["wr_cushion_x_man"] = (h.cushion_prev_c * h.opp_man_r6_z).where(isrec)
h["wr_pressgap_x_press"] = (h.tprr_press_gap * h.opp_press_r6_z).where(isrec)
h["rb_tprr_x_press"] = (h.tprr_r6_c * h.opp_press_r6_z).where(isrb)
h["rb_x_box"] = h.opp_box_r6_z.where(isrb)
h["qb_ttt_x_twohigh"] = (h.off_ttt_r6_c * h.opp_two_high_r6_z).where(isqb)

CANDS = ["qb_ttt_x_press", "qb_sack_x_press", "qb_scr_x_man", "qb_adot_x_twohigh", "qb_adot_x_press", "qb_ttt_x_twohigh",
         "wr_deep_x_twohigh", "wr_deep_x_press", "wr_deep_x_qbadot", "wr_sep_x_man", "wr_cushion_x_man", "wr_pressgap_x_press",
         "rb_tprr_x_press", "rb_x_box"]
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
say("\n### Mean FFA residual by quintile of the interaction (own position only)\n")
say("| feature | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 | n |"); say("|---|---|---|---|---|---|---|---|")
for c in CANDS:
    d = h.dropna(subset=[c])
    try: q = pd.qcut(d[c].rank(method="first"), 5, labels=False)
    except Exception: continue
    m = d.groupby(q).res.mean()
    say(f"| {c} | " + " | ".join(f"{m.get(i, np.nan):+.2f}" for i in range(5)) + f" | {m.get(4)-m.get(0):+.2f} | {len(d):,} |")
say("\n### Main effects for reference (own position, quintile Q5−Q1 of the FFA residual)\n")
for c, pos in (("off_ttt_r6", "QB"), ("qb_adot_r6", "QB"), ("qb_scramble_rate_r6", "QB"), ("opp_press_r6_z", "QB"), ("opp_two_high_r6_z", "QB"),
               ("deep_share_r6", "WR"), ("sep_prev", "WR"), ("opp_man_r6_z", "WR"), ("opp_two_high_r6_z", "WR"), ("opp_press_r6_z", "RB"), ("opp_box_r6_z", "RB")):
    d = h[(h.position == pos)].dropna(subset=[c])
    if len(d) < 200: say(f"- {pos} {c}: n={len(d)} (not available)"); continue
    q = pd.qcut(d[c].rank(method="first"), 5, labels=False); m = d.groupby(q).res.mean()
    say(f"- {pos} {c}: Q1 {m[0]:+.2f} … Q5 {m[4]:+.2f} (Δ {m[4]-m[0]:+.2f}, r {np.corrcoef(d[c], d.res)[0,1]:+.3f}, n={len(d):,})")
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
groups = {"QB x pressure": ["qb_ttt_x_press", "qb_sack_x_press", "qb_adot_x_press"], "QB x man/two-high": ["qb_scr_x_man", "qb_adot_x_twohigh", "qb_ttt_x_twohigh"],
          "WR deep x D": ["wr_deep_x_twohigh", "wr_deep_x_press"], "WR deep x QB aDOT": ["wr_deep_x_qbadot"],
          "WR separation x man": ["wr_sep_x_man", "wr_cushion_x_man"], "WR pressure gap": ["wr_pressgap_x_press"],
          "RB x pressure/box": ["rb_tprr_x_press", "rb_x_box"], "ALL interactions": CANDS}
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
open(os.path.join(ROOT, "outputs", "reports", "archetype_interaction_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/archetype_interaction_study.md")

# ---------- 3. do archetypes predict the SPREAD (|residual|) rather than the mean? ----------
say("\n## 3. Archetype vs absolute residual (spread), own position, 2023-25\n")
say("Model_Burke's quantile width is conditioned on the projection only; a variable that predicts |residual| beyond the projection would widen/narrow the right players.\n")
say("| position | variable | corr with \|res\| | partial corr (after proj) | Q1 mean \|res\| | Q5 mean \|res\| | n |"); say("|---|---|---|---|---|---|---|")
h["ares"] = h.res.abs()
from sklearn.linear_model import LinearRegression
for pos, c in (("QB", "qb_adot_r6"), ("QB", "off_ttt_r6"), ("QB", "qb_scramble_rate_r6"), ("QB", "opp_press_r6_z"),
               ("WR", "deep_share_r6"), ("WR", "sep_prev"), ("WR", "opp_two_high_r6_z"), ("WR", "opp_man_r6_z"), ("WR", "tprr_r6"),
               ("TE", "deep_share_r6"), ("RB", "tprr_r6"), ("RB", "opp_box_r6_z"), ("RB", "opp_press_r6_z")):
    d = h[(h.position == pos)].dropna(subset=[c, "ares", "baseline_proj"])
    if len(d) < 300: continue
    r = np.corrcoef(d[c], d.ares)[0, 1]
    # partial: residualise both on baseline_proj (+ proj^2)
    X = np.c_[d.baseline_proj, d.baseline_proj ** 2]
    ra = d.ares - LinearRegression().fit(X, d.ares).predict(X); rc = d[c] - LinearRegression().fit(X, d[c]).predict(X)
    pr = np.corrcoef(rc, ra)[0, 1]
    q = pd.qcut(d[c].rank(method="first"), 5, labels=False); m = d.groupby(q).ares.mean()
    say(f"| {pos} | {c} | {r:+.3f} | {pr:+.3f} | {m[0]:.2f} | {m[4]:.2f} | {len(d):,} |")
open(os.path.join(ROOT, "outputs", "reports", "archetype_interaction_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
