"""Build the league's frozen data frame.

One row per player-week (QB/RB/WR/TE with an FFA weekly stat line), 2016-2025, with every
feature a candidate may use and, for the VISIBLE seasons only, the actual outcomes.

  league/data/visible_frame.parquet     2016-2024, features + actuals (targets)
  league/data/holdout_features.parquet  2025, features only (no actual_* columns)
  <private dir>/holdout_labels.parquet  2025, player_id/season/week + actual_* — written OUTSIDE
                                        the repo (--labels-out), the private judge reads it there

Point-in-time: every feature is built from prior games only (lags/rolling windows shift by 1),
prior-season labels, or a pre-kick source (FFA weekly line, DK closing line, game line).
Known compromise, documented: DK "closing" lines are ~2h pre-kick, not Wednesday lines.

Feature families (candidates may use any subset):
  ffa_*      FFA weekly consensus stat line and its PPR score (baseline_proj), injury_status flag
  mkt_*      DK-implied PPR (market_proj; 2023+ weeks 5-22 only, NaN elsewhere) + rec-yds line
  ctx_*      spread, total, implied team total, home, outdoor
  lag_*      Model_Burke rolling box-score lags (l1/r3/r6/std3) + games_played/fp_trend
  rt_*       participation-derived routes: route share (l1/r3/r6), tprr/yprr r6 (shrunk), usage
             surprise, red-zone target rate, man/pressure target splits; team dropbacks r6
  opp_*      opponent defence prior-6: pressure rate, man rate, two-high rate, box (z within week)
  qb_*       team QB1 prior-6 aDOT, sack rate, scramble rate, CPOE (from last game's QB1)
  ngs_*      prior-season Next Gen avg separation / cushion
  arch_*     prior-season archetype labels (from the package's add_prior_season_labels)
Targets (visible frame only): actual_ppr, actual_targets, actual_carries, actual_receptions,
  actual_rec_yds, actual_rush_yds, actual_pass_yds, actual_active (played: any touch/attempt)
Usage: python league/data/build_frame.py <model_burke pkg dir> --labels-out <private dir>
"""
import os, re, sys, glob, argparse, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("pkg"); ap.add_argument("--labels-out", required=True)
ap.add_argument("--holdout-season", type=int, default=2025)
A = ap.parse_args()
sys.path.insert(0, A.pkg)
from model_burke.features import build_lagged_features, add_prior_season_labels
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
OUT = os.path.join(ROOT, "league", "data"); os.makedirs(OUT, exist_ok=True)

# ---------- actuals + lags ----------
wk = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "weekly_skill_2015_2025.parquet"))
wk = wk[wk.week <= 18].drop_duplicates(["player_id", "season", "week"])
for c in ("team", "opponent_team"): wk[c] = wk[c].map(lambda t: FIX.get(t, t))
wk["nname"] = wk.player_display_name.map(norm)
lag = build_lagged_features(wk)
lag = lag.rename(columns={c: ("lag_" + c) for c in lag.columns if c not in ("player_id", "season", "week")})
ids = wk.sort_values(["season", "week"]).drop_duplicates(["nname", "position", "season"], keep="last")[["nname", "position", "season", "player_id"]]

# ---------- FFA weekly lines ----------
def score_ppr(d):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    return (z("pass_yds") * .04 + z("pass_tds") * 4 - z("pass_int") * 2 + z("rush_yds") * .1 + z("rush_tds") * 6
            + z("rec") + z("rec_yds") * .1 + z("rec_tds") * 6 - z("fumbles_lost") * 2 + z("two_pts") * 2 + z("return_tds") * 6)
rows = []
FSTATS = ["pass_yds", "pass_tds", "pass_int", "rush_yds", "rush_tds", "rec", "rec_yds", "rec_tds", "fumbles_lost",
          "pass_yds_sd", "rush_yds_sd", "rec_yds_sd", "rec_sd"]
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    s_, w_ = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
    if s_ < 2016 or s_ > A.holdout_season or w_ > 18: continue
    d = pd.read_csv(f, na_values=["NA"], low_memory=False); d = d[d.position.isin(["QB", "RB", "WR", "TE"])].copy()
    d["team"] = d.team.map(lambda t: FIX.get(t, t)); d["nname"] = d.player.map(norm); d["season"], d["week"] = s_, w_
    d["baseline_proj"] = score_ppr(d)
    for c in FSTATS:
        if c not in d.columns: d[c] = np.nan
    d["ffa_injury_q"] = d.get("injury_status", pd.Series(index=d.index, dtype=object)).isin(["Q", "Questionable"]).astype(float)
    d["ffa_injury_o"] = d.get("injury_status", pd.Series(index=d.index, dtype=object)).isin(["O", "Out", "D", "Doubtful"]).astype(float)
    rows.append(d[["nname", "position", "team", "season", "week", "baseline_proj", "ffa_injury_q", "ffa_injury_o"] + FSTATS].drop_duplicates(["nname", "position"]))
h = pd.concat(rows).merge(ids, on=["nname", "position", "season"], how="inner")
h = h.rename(columns={c: "ffa_" + c for c in FSTATS})

# ---------- game context ----------
gl = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "game_lines_2015_2026.parquet"))
for c in ("team", "opp"): gl[c] = gl[c].map(lambda t: FIX.get(t, t))
gl = gl.drop_duplicates(["season", "week", "team"])
DOME = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LAC", "LA", "MIN", "NO"}
gl["home"] = np.where(gl.is_home == 1, gl.team, gl.opp)
h = h.merge(gl[["season", "week", "team", "opp", "team_spread", "game_total", "implied_team_total", "is_home", "home"]]
            .rename(columns={"opp": "opponent_team", "team_spread": "ctx_spread", "game_total": "ctx_total",
                             "implied_team_total": "ctx_implied", "is_home": "ctx_home"}), on=["season", "week", "team"], how="left")
h["ctx_outdoor"] = (~h.home.isin(DOME)).astype(float)

# ---------- actuals ----------
act = wk[["player_id", "season", "week", "fantasy_points_ppr", "targets", "carries", "receptions", "receiving_yards", "rushing_yards", "passing_yards", "attempts"]]
act = act.rename(columns={"fantasy_points_ppr": "actual_ppr", "targets": "actual_targets", "carries": "actual_carries", "receptions": "actual_receptions",
                          "receiving_yards": "actual_rec_yds", "rushing_yards": "actual_rush_yds", "passing_yards": "actual_pass_yds", "attempts": "actual_attempts"})
h = h.merge(act, on=["player_id", "season", "week"], how="left")
h["actual_active"] = ((h.actual_targets.fillna(0) + h.actual_carries.fillna(0) + h.actual_attempts.fillna(0)) > 0).astype(float)
h["actual_ppr"] = h.actual_ppr.fillna(0.0)      # projected but no box score = did not play = 0
for c in ("actual_targets", "actual_carries", "actual_receptions", "actual_rec_yds", "actual_rush_yds", "actual_pass_yds", "actual_attempts"): h[c] = h[c].fillna(0.0)
h = h.merge(lag, on=["player_id", "season", "week"], how="left")

# ---------- market (DK-implied PPR from the package sample; rec-yds closing line from the props frame) ----------
mb = pd.read_csv(os.path.join(A.pkg, "data", "sample_input.csv.gz"), low_memory=False)
mk = mb[["player_id", "season", "week", "market_proj"]].dropna().drop_duplicates(["player_id", "season", "week"]).rename(columns={"market_proj": "mkt_ppr"})
h = h.merge(mk, on=["player_id", "season", "week"], how="left")
pf = os.path.join(ROOT, "data", "props_frames", "props_player_reception_yds.parquet")
if os.path.exists(pf):
    p = pd.read_parquet(pf)
    if "player_id" in p.columns:
        p = p.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season", "week"])[["player_id", "season", "week", "baseline_proj"]].rename(columns={"baseline_proj": "mkt_rec_yds_line"})
        h = h.merge(p, on=["player_id", "season", "week"], how="left")
h["mkt_has_line"] = h.mkt_ppr.notna().astype(float)

# ---------- routes / participation ----------
PD = os.path.join(ROOT, "data", "participation")
rt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "routes_*.parquet")))])
rt = rt[rt.week <= 18].rename(columns={"gsis_id": "player_id", "posteam": "team", "defteam": "opponent_team"})
for c in ("team", "opponent_team"): rt[c] = rt[c].map(lambda t: FIX.get(t, t))
rt = rt.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
g = rt.groupby("player_id", sort=False)
rt["rt_route_share_l1"] = g.route_share.shift(1)
rt["rt_route_share_r3"] = g.route_share.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
rt["rt_route_share_r6"] = g.route_share.transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
rt["rt_rs_r6_prev"] = g.route_share.transform(lambda s: s.shift(2).rolling(6, min_periods=2).mean())
rt["rt_route_surprise"] = rt.rt_route_share_l1 - rt.rt_rs_r6_prev
for c in ("targets_on_routes", "routes", "yds_on_tgt", "rz_targets", "rz_routes", "tgt_man", "man_routes", "tgt_press", "press_routes", "deep_targets"):
    rt[c + "_s6"] = g[c].transform(lambda s: s.shift(1).rolling(6, min_periods=2).sum())
posm = rt.groupby("pos").apply(lambda d: d.targets_on_routes.sum() / d.routes.sum()).rename("tprr_pos")
rt = rt.merge(posm, left_on="pos", right_index=True, how="left"); K = 60.0
rt["rt_tprr_r6"] = (rt.targets_on_routes_s6 + K * rt.tprr_pos) / (rt.routes_s6 + K)
rt["rt_yprr_r6"] = rt.yds_on_tgt_s6 / (rt.routes_s6 + K)
rt["rt_rz_tgt_rate_r6"] = rt.rz_targets_s6 / (rt.rz_routes_s6 + 10)
rt["rt_deep_share_r6"] = rt.deep_targets_s6 / (rt.targets_on_routes_s6 + 5)
rt["rt_man_gap_r6"] = rt.tgt_man_s6 / (rt.man_routes_s6 + K) - (rt.targets_on_routes_s6 - rt.tgt_man_s6) / (rt.routes_s6 - rt.man_routes_s6 + K)
rt["rt_press_gap_r6"] = rt.tgt_press_s6 / (rt.press_routes_s6 + 30) - (rt.targets_on_routes_s6 - rt.tgt_press_s6) / (rt.routes_s6 - rt.press_routes_s6 + 30)
h = h.merge(rt[["player_id", "season", "week"] + [c for c in rt.columns if c.startswith("rt_")]], on=["player_id", "season", "week"], how="left")
td = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PD, "team_def_*.parquet")))])
td = td[td.week <= 18].rename(columns={"posteam": "team", "defteam": "opponent_team"})
for c in ("team", "opponent_team"): td[c] = td[c].map(lambda t: FIX.get(t, t))
td = td.sort_values(["season", "week"]).drop_duplicates(["season", "week", "team"])
td["rt_team_db_r6"] = td.groupby(["team", "season"]).team_dropbacks.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
td["rt_team_ttt_r6"] = td.groupby(["team", "season"]).ttt_mean.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
for c, src in (("opp_press_r6", "def_pressure_rate"), ("opp_man_r6", "def_man_rate"), ("opp_two_high_r6", "def_two_high_rate"), ("opp_box_r6", "box_mean")):
    td[c] = td.groupby(["opponent_team", "season"])[src].transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    lg = td.groupby(["season", "week"])[c].agg(["mean", "std"]).reset_index()
    td = td.merge(lg, on=["season", "week"]); td[c + "_z"] = (td[c] - td["mean"]) / td["std"]; td = td.drop(columns=["mean", "std"])
h = h.merge(td[["season", "week", "team", "rt_team_db_r6", "rt_team_ttt_r6"]], on=["season", "week", "team"], how="left")
h = h.merge(td[["season", "week", "opponent_team"] + [c for c in td.columns if c.startswith("opp_")]].drop_duplicates(["season", "week", "opponent_team"]),
            on=["season", "week", "opponent_team"], how="left")

# ---------- QB1 archetype (prior 6), attached via last game's QB1 ----------
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
qb = pd.read_sql("SELECT gsis_id player_id, season, week, adot, sack_rate, scramble_rate, cpoe FROM nflv_pbp_skill_wk WHERE role='pass' AND season BETWEEN 2015 AND 2025", con)
ngs = pd.read_sql("SELECT player_gsis_id player_id, season, avg_separation, avg_cushion FROM nflv_ngs_receiving WHERE week > 0", con)
con.close()
qb = qb.sort_values(["season", "week"]).drop_duplicates(["player_id", "season", "week"]); gq = qb.groupby("player_id", sort=False)
for c in ("adot", "sack_rate", "scramble_rate", "cpoe"): qb[f"qb_{c}_r6"] = gq[c].transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
tg = wk[["season", "week", "team"]].drop_duplicates().sort_values(["season", "week"]); tg["prev_week"] = tg.groupby(["season", "team"]).week.shift(1)
lead = wk[wk.position == "QB"].sort_values("attempts", ascending=False).drop_duplicates(["season", "week", "team"])
lead = lead[lead.attempts >= 10][["season", "week", "team", "player_id"]].rename(columns={"week": "prev_week", "player_id": "qb1"})
tq = tg.merge(lead, on=["season", "prev_week", "team"], how="inner").merge(qb[["player_id", "season", "week"] + [c for c in qb.columns if c.startswith("qb_")]].rename(columns={"player_id": "qb1"}), on=["qb1", "season", "week"], how="left")
h = h.merge(tq[["season", "week", "team"] + [c for c in tq.columns if c.startswith("qb_")]], on=["season", "week", "team"], how="left")
sep = ngs.groupby(["player_id", "season"])[["avg_separation", "avg_cushion"]].mean().reset_index(); sep["season"] += 1
h = h.merge(sep.rename(columns={"avg_separation": "ngs_sep_prev", "avg_cushion": "ngs_cushion_prev"}), on=["player_id", "season"], how="left")
# archetype labels (prior season) via the package helper
try:
    h = add_prior_season_labels(h)
    for c in [c for c in h.columns if c.endswith("_prev") and not c.startswith(("ngs_", "rt_", "lag_"))]: h = h.rename(columns={c: "arch_" + c})
except Exception as e: print("prior-season labels skipped:", str(e)[:80])

# ---------- finalise ----------
h = h.drop_duplicates(["player_id", "season", "week"]).sort_values(["season", "week", "player_id"]).reset_index(drop=True)
h["t"] = h.season * 100 + h.week
keep_meta = ["player_id", "nname", "position", "team", "opponent_team", "season", "week", "t"]
feat = [c for c in h.columns if c.startswith(("ffa_", "mkt_", "ctx_", "lag_", "rt_", "opp_", "qb_", "ngs_", "arch_"))] + ["baseline_proj"]
tgt = [c for c in h.columns if c.startswith("actual_")]
vis = h[h.season < A.holdout_season][keep_meta + feat + tgt]
hold = h[h.season == A.holdout_season]
vis.to_parquet(os.path.join(OUT, "visible_frame.parquet"), index=False)
hold[keep_meta + feat].to_parquet(os.path.join(OUT, "holdout_features.parquet"), index=False)
os.makedirs(A.labels_out, exist_ok=True)
hold[["player_id", "season", "week"] + tgt].to_parquet(os.path.join(A.labels_out, "holdout_labels.parquet"), index=False)
open(os.path.join(OUT, "FRAME_MANIFEST.txt"), "w").write(
    f"built {pd.Timestamp.now():%Y-%m-%d %H:%M}\nvisible rows {len(vis):,} seasons {vis.season.min()}-{vis.season.max()}\nholdout rows {len(hold):,} season {A.holdout_season}\n"
    f"features ({len(feat)}): {', '.join(feat)}\ntargets: {', '.join(tgt)}\n")
print(f"visible {len(vis):,} rows 2016-{A.holdout_season-1}, {len(feat)} features; holdout {len(hold):,} rows; labels -> {A.labels_out}")
print("coverage:", {c: round(float(vis[c].notna().mean()), 2) for c in ["mkt_ppr", "rt_route_share_r3", "opp_press_r6_z", "qb_adot_r6", "ngs_sep_prev", "lag_targets_r3"] if c in vis})
