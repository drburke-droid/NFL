"""Candidate features for the SaberSim pipeline — measured, not guessed.

Builds the same 2023-25 history frame the generator trains on (FFA weekly baseline +
Model_Burke lags), adds each candidate, and re-runs Model_Burke walk-forward with and
without it. Scored on 2025: MAE, RMSE, weekly Spearman, 80% coverage.

Candidates (all knowable before kickoff):
  snap_l1 / snap_r3     prior-game(s) offensive snap share (nflv_snaps)
  qb1_absent            the team's usage-leading QB from last game is not playing
  rest_days / short_wk  days since the team's last game (Thu game, post-bye)
  is_home
  temp_c / precip_mm / wind_kn   measured game weather (forecast proxy)
  pass_funnel_z         opponent's prior-6 pass-yds share of yards allowed, z within week
  td_luck_r6            player's TD-per-touch over prior 6 vs position mean (regression)
  plays_r6              team offensive plays per game, prior 6
  dvp_z                 opponent prior-6 fantasy points allowed to the position
  wiki_spike            Wikipedia views 3 days pre-game / prior-30-day median (news)
Usage: python scripts/feature_candidates_study.py <model_burke pkg dir>
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

# ---------- candidates ----------
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
# snaps via pfr id
ros = pd.read_sql("SELECT DISTINCT season, gsis_id player_id, pfr_id FROM nflv_rosters WHERE season BETWEEN 2022 AND 2025", con).dropna()
sn = pd.read_sql("SELECT season, week, pfr_player_id pfr_id, offense_pct FROM nflv_snaps WHERE game_type='REG' AND season BETWEEN 2022 AND 2025", con)
sn = sn.merge(ros, on=["season", "pfr_id"]).drop_duplicates(["player_id", "season", "week"]).sort_values(["season", "week"])
sn["snap_l1"] = sn.groupby(["player_id", "season"]).offense_pct.shift(1)
sn["snap_r3"] = sn.groupby(["player_id", "season"]).offense_pct.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
h = h.merge(sn[["player_id", "season", "week", "snap_l1", "snap_r3"]], on=["player_id", "season", "week"], how="left")
# QB1 absent
qb = wk[wk.position == "QB"].sort_values(["season", "week"])
tgames = wk[["season", "week", "team"]].drop_duplicates().sort_values(["season", "week"])
tgames["prev_week"] = tgames.groupby(["season", "team"]).week.shift(1)
lead_prev = qb.sort_values("attempts", ascending=False).drop_duplicates(["season", "week", "team"])[["season", "week", "team", "player_id", "attempts"]]
lead_prev = lead_prev[lead_prev.attempts >= 15].rename(columns={"week": "prev_week", "player_id": "qb1"})
lead = tgames.merge(lead_prev, on=["season", "prev_week", "team"], how="inner")
played = wk[["season", "week", "player_id"]].assign(pl=1).rename(columns={"player_id": "qb1"})
lead = lead.merge(played, on=["season", "week", "qb1"], how="left")
lead["qb1_absent"] = lead.pl.isna().astype(int)
h = h.merge(lead[["season", "week", "team", "qb1_absent"]], on=["season", "week", "team"], how="left")
h["qb1_absent"] = h.qb1_absent.fillna(0)
# rest days + home + weather from games table (2020-25)
gm = pd.read_sql("SELECT season, week, home_team, away_team, commence_time FROM games WHERE season BETWEEN 2022 AND 2025", con)
gm["season"] = pd.to_numeric(gm.season, errors="coerce"); gm["week"] = pd.to_numeric(gm.week, errors="coerce")
gm = gm.dropna(subset=["season", "week"]).astype({"season": int, "week": int})
gm["home"] = gm.home_team.map(NAME2ABBR); gm["away"] = gm.away_team.map(NAME2ABBR)
gm["kick"] = pd.to_datetime(gm.commence_time)
tg = pd.concat([gm.assign(team=gm.home), gm.assign(team=gm.away)])[["season", "week", "team", "kick", "home"]].drop_duplicates(["season", "week", "team"])
tg = tg.sort_values("kick"); tg["rest_days"] = tg.groupby(["team", "season"]).kick.diff().dt.days
tg["short_wk"] = (tg.rest_days <= 5).astype(float); tg["post_bye"] = (tg.rest_days >= 12).astype(float)
h = h.merge(tg[["season", "week", "team", "rest_days", "short_wk", "post_bye"]], on=["season", "week", "team"], how="left")
wx = pd.read_sql("SELECT w.game_id, w.temp_c, w.precip_mm, d.wind_kn FROM nflv_game_wx_hist w LEFT JOIN nflv_game_wind_hist d ON d.game_id=w.game_id", con)
gid = wx.game_id.str.split("_", expand=True); wx["season"] = gid[0].astype(int); wx["week"] = gid[1].astype(int); wx["home"] = gid[3].map(lambda t: FIX.get(t, t))
h = h.merge(wx[["season", "week", "home", "temp_c", "precip_mm", "wind_kn"]].rename(columns={"wind_kn": "wind_meas"}), on=["season", "week", "home"], how="left")
# pass funnel + DvP + plays
tm = wk.groupby(["season", "week", "team", "opponent_team"]).agg(pass_y=("passing_yards", "sum"), rush_y=("rushing_yards", "sum"),
                                                              plays=("attempts", "sum"), car=("carries", "sum")).reset_index()
tm["plays"] = tm.plays + tm.car
tm = tm.sort_values(["season", "week"])
tm["plays_r6"] = tm.groupby(["team", "season"]).plays.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
dfn = tm.rename(columns={"team": "off", "opponent_team": "opponent_team_"}).copy()
dfn = dfn.rename(columns={"opponent_team_": "team"})   # rows keyed by the DEFENSE
dfn["pshare"] = dfn.pass_y / (dfn.pass_y + dfn.rush_y)
dfn["pf_r6"] = dfn.groupby(["team", "season"]).pshare.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
lgm = dfn.groupby(["season", "week"]).pf_r6.agg(["mean", "std"]).reset_index()
dfn = dfn.merge(lgm, on=["season", "week"]); dfn["pass_funnel_z"] = (dfn.pf_r6 - dfn["mean"]) / dfn["std"]
h = h.merge(dfn[["season", "week", "team", "pass_funnel_z"]].rename(columns={"team": "opponent_team"}), on=["season", "week", "opponent_team"], how="left")
h = h.merge(tm[["season", "week", "team", "plays_r6"]], on=["season", "week", "team"], how="left")
al = wk.groupby(["season", "week", "opponent_team", "position"]).fantasy_points_ppr.sum().reset_index().sort_values(["season", "week"])
al["dvp_r6"] = al.groupby(["opponent_team", "position", "season"]).fantasy_points_ppr.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
lg = al.groupby(["season", "week", "position"]).dvp_r6.agg(["mean", "std"]).reset_index()
al = al.merge(lg, on=["season", "week", "position"]); al["dvp_z"] = (al.dvp_r6 - al["mean"]) / al["std"]
h = h.merge(al[["season", "week", "opponent_team", "position", "dvp_z"]], on=["season", "week", "opponent_team", "position"], how="left")
# TD luck
w2 = wk.sort_values(["season", "week"]).copy()
w2["td"] = w2.get("rushing_tds", 0) if "rushing_tds" in w2 else 0
# nflv_weekly parquet lacks TD columns -> derive TDs from ppr identity is unreliable; use fantasy points per touch instead
w2["touch"] = w2.carries.fillna(0) + w2.receptions.fillna(0) + w2.attempts.fillna(0)
w2["fpt_r6"] = w2.groupby(["player_id", "season"]).apply(lambda g: (g.fantasy_points_ppr.shift(1).rolling(6, min_periods=3).sum() / g.touch.shift(1).rolling(6, min_periods=3).sum())).reset_index(level=[0, 1], drop=True)
pm = w2.groupby(["season", "position"]).apply(lambda g: g.fantasy_points_ppr.sum() / g.touch.sum()).rename("fpt_pos").reset_index()
w2 = w2.merge(pm, on=["season", "position"]); w2["eff_luck"] = w2.fpt_r6 - w2.fpt_pos
h = h.merge(w2[["player_id", "season", "week", "eff_luck"]], on=["player_id", "season", "week"], how="left")
h["wiki_spike"] = np.nan   # nflv_wiki_buzz_daily only covers Jun-Aug 2026 — no history to test
con.close()

CANDS = ["snap_l1", "snap_r3", "qb1_absent", "rest_days", "short_wk", "post_bye", "is_home", "temp_c", "precip_mm",
         "wind_meas", "pass_funnel_z", "plays_r6", "dvp_z", "eff_luck", "wiki_spike"]
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
groups = {"snaps": ["snap_l1", "snap_r3"], "qb1_absent": ["qb1_absent"], "rest/home": ["rest_days", "short_wk", "post_bye", "is_home"],
          "weather": ["temp_c", "precip_mm", "wind_meas"], "pass_funnel": ["pass_funnel_z"], "plays_r6": ["plays_r6"],
          "dvp": ["dvp_z"], "eff_luck": ["eff_luck"]}
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
open(os.path.join(ROOT, "outputs", "reports", "feature_candidates_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/feature_candidates_study.md")
