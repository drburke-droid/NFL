"""Player-skill Elo v2: adds a PASSER term to the ridge APM for receivers.

Motivation: skill_elo.py adjusts WR/TE efficiency for opponent (split WR1 vs
WR2/3) but not for who threw the ball. A WR who posts league-average EPA/target
on 185 targets from backup QBs (Ja'Marr Chase, 2025: Browning/Flacco for 10 of
16 games) is graded as if his QB were average — so elite receivers on broken
offenses get buried. Fix:

    epa_play(WR/TE-week) = intercept + player + defense@rank + passer + e

Passer = the team-week's primary QB (most pass plays). QB effects are
identified from every receiver they throw to plus mid-season QB changes;
ridge shrinkage handles the within-team collinearity, and the whole thing is
judged out-of-time exactly like v1: corr(effect_t, next season's raw skill z).

QB/RB-rush fits are unchanged (no passer term). Writes player_skill_elo_qb
and prints head-to-head vs v1 (player_skill_elo) and the unadjusted Kalman.
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("sk", os.path.join(HERE, "skill_kalman.py"))
SK = importlib.util.module_from_spec(_s)
_s.loader.exec_module(SK)

DB = os.path.join(os.path.dirname(HERE), "db", "nfl_odds.db")
POS_ROLE = {"QB": "pass", "RB": "rush", "WR": "rec", "TE": "rec"}
MIN_WK_PLAYS = {"QB": 10, "RB": 5, "WR": 3, "TE": 3}
SEASONS = range(2011, 2026)

con = sqlite3.connect(DB)
wk = pd.read_sql("""SELECT gsis_id AS player_id, season, week, role, plays, epa_play
                    FROM nflv_pbp_skill_wk WHERE epa_play IS NOT NULL""", con)
pos = pd.read_sql("SELECT DISTINCT player_id, position FROM player_skill_seasons", con)
opp = pd.read_sql("""SELECT player_id, season, week, team, opponent_team
                     FROM nflv_weekly""", con).drop_duplicates(["player_id", "season", "week"])
raw = pd.read_sql("""SELECT s.player_id, s.position, s.season, s.plays, s.skill_z,
                            t.skill_true_causal
                     FROM player_skill_seasons s
                     JOIN player_skill_true t USING(player_id, season)""", con)
names = pd.read_sql("SELECT DISTINCT player_id, player_display_name AS name FROM nflv_traj", con)

wk = wk.merge(pos, on="player_id").merge(opp, on=["player_id", "season", "week"], how="inner")
wk = wk[wk.apply(lambda r: POS_ROLE.get(r["position"]) == r["role"], axis=1)]
wk = wk[wk["plays"] >= wk["position"].map(MIN_WK_PLAYS)]

# WR depth rank within team-season (by total targets) for rank-split defense effects
tt = wk[wk["position"] == "WR"].groupby(["team", "season", "player_id"])["plays"].sum().reset_index()
tt["wrrank"] = tt.groupby(["team", "season"])["plays"].rank(ascending=False, method="first")
wk = wk.merge(tt[["team", "season", "player_id", "wrrank"]], on=["team", "season", "player_id"], how="left")

# primary passer per team-week (most pass plays that week)
qbwk = wk[wk["role"] == "pass"].sort_values("plays", ascending=False) \
         .drop_duplicates(["team", "season", "week"])[["team", "season", "week", "player_id"]] \
         .rename(columns={"player_id": "qb_id"})
wk = wk.merge(qbwk, on=["team", "season", "week"], how="left")
n_rec = len(wk[wk["role"] == "rec"])
n_miss = wk[(wk["role"] == "rec") & wk["qb_id"].isna()].shape[0]
print(f"receiver-weeks: {n_rec:,}; missing primary QB: {n_miss:,} "
      f"({100*n_miss/max(n_rec,1):.1f}%) -> mapped to 'UNK'")
wk["qb_id"] = wk["qb_id"].fillna("UNK")


def fit_season(d, position, alpha):
    """Ridge APM for one position-season. Adds passer dummies for rec roles."""
    d = d.copy()
    if position == "WR":
        d["defkey"] = d["opponent_team"] + np.where(d["wrrank"] == 1, "@1", "@23")
    else:
        d["defkey"] = d["opponent_team"]
    blocks = [pd.get_dummies(d["player_id"], sparse=False)]
    npl = blocks[0].shape[1]
    blocks.append(pd.get_dummies(d["defkey"], sparse=False))
    if POS_ROLE[position] == "rec":
        blocks.append(pd.get_dummies(d["qb_id"], prefix="qb", sparse=False))
    X = np.hstack([b.to_numpy(float) for b in blocks])
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, d["epa_play"], sample_weight=d["plays"])
    pe = pd.Series(m.coef_[:npl], index=blocks[0].columns, name="eff")
    nde = blocks[1].shape[1]
    de = pd.Series(m.coef_[npl:npl + nde], index=blocks[1].columns, name="def_eff")
    qe = None
    if POS_ROLE[position] == "rec":
        qe = pd.Series(m.coef_[npl + nde:], index=blocks[2].columns, name="qb_eff")
    return pe, de, qe


def run_all(alpha_by_pos):
    effs, qbs = [], []
    for position in POS_ROLE:
        for season in SEASONS:
            d = wk[(wk["position"] == position) & (wk["season"] == season)]
            if len(d) < 100: continue
            pe, de, qe = fit_season(d, position, alpha_by_pos[position])
            pl = d.groupby("player_id")["plays"].sum()
            e = pe.reset_index().rename(columns={"index": "player_id"})
            e["season"] = season; e["position"] = position
            e["plays"] = e["player_id"].map(pl)
            effs.append(e)
            if qe is not None:
                q = qe.reset_index().rename(columns={"index": "qb_id"})
                q["season"] = season; q["position"] = position
                qbs.append(q)
    return pd.concat(effs, ignore_index=True), pd.concat(qbs, ignore_index=True)


# ---- choose alpha per position by out-of-time signal ----
print("\n=== Ridge alpha selection: corr(adjusted effect_t, raw skill z_{t+1}) ===")
nxt_raw = raw[["player_id", "season", "skill_z"]].copy()
nxt_raw["season"] -= 1
best_alpha = {}
for position in POS_ROLE:
    scores = {}
    for alpha in (4.0, 12.0, 40.0, 120.0):
        rows = []
        for season in SEASONS:
            d = wk[(wk["position"] == position) & (wk["season"] == season)]
            if len(d) < 100: continue
            pe, _, _ = fit_season(d, position, alpha)
            e = pe.reset_index().rename(columns={"index": "player_id"})
            e["season"] = season
            rows.append(e)
        e = pd.concat(rows).merge(nxt_raw, on=["player_id", "season"]).dropna()
        scores[alpha] = pearsonr(e["eff"], e["skill_z"])[0] if len(e) > 50 else np.nan
    best_alpha[position] = max(scores, key=lambda a: scores[a])
    print(f"  {position}: " + "  ".join(f"a={a}: r={scores[a]:.3f}" for a in scores)
          + f"  -> alpha {best_alpha[position]}")

eff, qeff = run_all(best_alpha)

# era-normalize adjusted effects (plays-weighted z within position-season)
def wz(g):
    w = g["plays"]; m = np.average(g["eff"], weights=w)
    sd = np.sqrt(np.average((g["eff"] - m) ** 2, weights=w)) or 1.0
    return (g["eff"] - m) / sd
eff["elo_z"] = np.nan
for _, g in eff.groupby(["position", "season"]):
    eff.loc[g.index, "elo_z"] = wz(g)

# ---- Kalman over adjusted series ----
print("\n=== Kalman over QB-adjusted skill (one-step MAE vs repeat-last) ===")
MINP = {"QB": 150, "RB": 80, "WR": 40, "TE": 40}
out = []
for position in POS_ROLE:
    d = eff[(eff["position"] == position)].rename(columns={"elo_z": "skill_composite"})
    d = d[d["plays"] >= MINP[position] * 0.6]
    medP = d["plays"].median()
    best = None
    for phi in (0.75, 0.85, 0.92, 0.97):
        for Q in (0.02, 0.05, 0.10, 0.20):
            for R0 in (0.4, 0.7, 1.0, 1.6, 2.5):
                mae, bmae = SK.one_step_score(d, phi, Q, R0, medP)
                if best is None or mae < best[0]: best = (mae, bmae, phi, Q, R0)
    mae, bmae, phi, Q, R0 = best
    print(f"  {position}: phi={phi} Q={Q} R0={R0} | {mae:.3f} vs {bmae:.3f} ({100*(bmae-mae)/bmae:+.0f}%)")
    for pid, g in d.groupby("player_id"):
        g = g.sort_values("season")
        y = g["skill_composite"].to_numpy(); yrs = g["season"].to_numpy()
        rv = R0 * medP / g["plays"].to_numpy()
        xf, xs, _ = SK.kalman_career(yrs, y, rv, phi, Q)
        for i in range(len(y)):
            out.append({"player_id": pid, "position": position, "season": int(yrs[i]),
                        "elo_obs": float(y[i]), "elo_true": float(xs[i]),
                        "elo_true_causal": float(xf[i]), "plays": int(g["plays"].iloc[i])})
res = pd.DataFrame(out)
res.to_sql("player_skill_elo_qb", con, if_exists="replace", index=False)
print(f"\nSaved player_skill_elo_qb: {len(res):,} rows")

# ---- head-to-head: v2 vs v1 vs unadjusted, predicting next RAW skill z ----
print("\n=== Predicting next season's RAW skill z (causal estimates) ===")
v1 = pd.read_sql("SELECT player_id, position, season, elo_true_causal AS v1 FROM player_skill_elo", con)
h = res.merge(raw[["player_id", "season", "position", "skill_true_causal"]],
              on=["player_id", "season", "position"])
h = h.merge(v1, on=["player_id", "season", "position"], how="left")
nx = raw[["player_id", "season", "position", "skill_z"]].copy(); nx["season"] -= 1
h = h.merge(nx, on=["player_id", "season", "position"]).dropna(subset=["skill_z", "v1"])
for position in POS_ROLE:
    d = h[h["position"] == position]
    r2 = pearsonr(d["elo_true_causal"], d["skill_z"])[0]
    r1 = pearsonr(d["v1"], d["skill_z"])[0]
    r0 = pearsonr(d["skill_true_causal"], d["skill_z"])[0]
    print(f"  {position}: QB-adj r={r2:.3f} | v1 r={r1:.3f} | unadjusted r={r0:.3f} (n={len(d)})")

# ---- QB effect sanity: 2025 passer effects seen by WR/TE fits ----
qeff["qb_id"] = qeff["qb_id"].str.replace("qb_", "", n=1)
qn = qeff.merge(names, left_on="qb_id", right_on="player_id", how="left")
q25 = qn[(qn["season"] == 2025) & (qn["position"] == "WR")].sort_values("qb_eff")
print("\n=== 2025 passer effects (WR fit): 5 worst / 5 best EPA-per-target ===")
for r in pd.concat([q25.head(5), q25.tail(5)]).itertuples():
    print(f"  {str(r.name):24s} {r.qb_eff:+.3f}")

# ---- 2025 WR leaders, v2 vs v1, + Chase career ----
print("\n=== 2025 WR Elo top 15: QB-adjusted (v2) ===")
r25 = res[(res["season"] == 2025) & (res["position"] == "WR")].merge(names, on="player_id", how="left")
v125 = pd.read_sql("""SELECT player_id, elo_true_causal AS v1c FROM player_skill_elo
                      WHERE season=2025 AND position='WR'""", con)
r25 = r25.merge(v125, on="player_id", how="left")
r25["rk_v1"] = r25["v1c"].rank(ascending=False)
for i, r in enumerate(r25.nlargest(15, "elo_true_causal").itertuples(), 1):
    print(f"  {i:2d}. {str(r.name):24s} {r.elo_true_causal:+.2f}  (v1 rank {r.rk_v1:.0f})")
ch = res.merge(names, on="player_id").query("name==\"Ja'Marr Chase\"").sort_values("season")
print("\nChase Elo by season, QB-adjusted (obs -> causal):")
print("  " + "  ".join(f"{r.season}:{r.elo_obs:+.2f}->{r.elo_true_causal:+.2f}" for r in ch.itertuples()))
con.close()
