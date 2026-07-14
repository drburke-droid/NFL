"""Forward predictive value of the true-skill trajectories for NEXT season.

Uses skill_true_causal (filtered estimate through season t, no future data).
1. Predicting next season's OBSERVED skill: causal estimate vs repeat-last-obs.
2. Predicting next season's PPG: partial correlation of causal level and causal
   trend with next_ppg / next dPPG, controlling prior PPG and age.
3. Decline flags: full-workload true-skill decliners vs risers - next dPPG,
   bust rate, by position and for age-28+ veterans (the CMC question).
4. Walk-forward PPG MAE: base features vs base + causal trajectory features.
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("ms", os.path.join(HERE, "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)
DB = os.path.join(os.path.dirname(HERE), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]

con = sqlite3.connect(DB)
st = pd.read_sql("SELECT * FROM player_skill_true", con)
sd = pd.read_sql("""SELECT player_id, season, position, age, prior_ppg, next_ppg
                    FROM season_dataset""", con)

st = st.sort_values(["player_id", "season"])
g = st.groupby("player_id")
st["causal_trend"] = np.where((st["season"] - g["season"].shift(1)) == 1,
                              st["skill_true_causal"] - g["skill_true_causal"].shift(1), np.nan)
nxt = st[["player_id", "season", "skill_obs"]].copy()
nxt["season"] -= 1
st = st.merge(nxt, on=["player_id", "season"], how="left", suffixes=("", "_next"))
# next-season outcomes live in season_dataset keyed by TARGET season = t+1
out = sd.rename(columns={"season": "tgt"})
out["season"] = out["tgt"] - 1
st = st.merge(out[["player_id", "season", "age", "prior_ppg", "next_ppg"]],
              on=["player_id", "season"], how="left")
medP = st.groupby("position")["plays"].transform("median")
st["full_load"] = st["plays"] >= 0.8 * medP

print("=== 1. Predicting NEXT season's observed skill z ===")
print(f"{'pos':4s} {'n':>5s} | causal-true MAE  r | repeat-obs MAE  r")
for pos in POS:
    d = st[(st["position"] == pos)].dropna(subset=["skill_obs_next"])
    m1 = mean_absolute_error(d["skill_obs_next"], d["skill_true_causal"])
    r1 = pearsonr(d["skill_true_causal"], d["skill_obs_next"])[0]
    m0 = mean_absolute_error(d["skill_obs_next"], d["skill_obs"])
    r0 = pearsonr(d["skill_obs"], d["skill_obs_next"])[0]
    print(f"{pos:4s} {len(d):5d} | {m1:.3f}  {r1:.3f}       | {m0:.3f}  {r0:.3f}")

print("\n=== 2. Does the trajectory say anything about NEXT PPG beyond prior PPG? ===")
def partial(d, xcol, ycol, ctrl):
    d = d.dropna(subset=[xcol, ycol] + ctrl)
    if len(d) < 60: return np.nan, len(d)
    A = np.column_stack([np.ones(len(d))] + [d[c] for c in ctrl])
    rx = d[xcol] - A @ np.linalg.lstsq(A, d[xcol], rcond=None)[0]
    ry = d[ycol] - A @ np.linalg.lstsq(A, d[ycol], rcond=None)[0]
    return pearsonr(rx, ry)[0], len(d)
st["age"] = st["age"].fillna(st["age"].median())
print(f"{'pos':4s} | level->nextPPG (ctrl priorPPG,age) | trend->nextPPG (ctrl level,priorPPG,age)")
for pos in POS:
    d = st[st["position"] == pos]
    r1, n1 = partial(d, "skill_true_causal", "next_ppg", ["prior_ppg", "age"])
    r2, n2 = partial(d, "causal_trend", "next_ppg", ["skill_true_causal", "prior_ppg", "age"])
    print(f"{pos:4s} | r={r1:+.3f} (n={n1})              | r={r2:+.3f} (n={n2})")

print("\n=== 3. Decline flags: full-workload causal trend, next-season outcome ===")
st["dppg"] = st["next_ppg"] - st["prior_ppg"]
st["bust"] = (st["next_ppg"] <= 0.7 * st["prior_ppg"]).astype(float)
def flagrow(d, lab):
    d = d.dropna(subset=["next_ppg", "prior_ppg"])
    if len(d) < 20: return
    print(f"  {lab:34s} n={len(d):4d}  next dPPG {d['dppg'].mean():+.2f}  "
          f"bust {d['bust'].mean():.0%}  next PPG {d['next_ppg'].mean():.1f}")
for pos in POS:
    d = st[(st["position"] == pos) & st["full_load"] & st["causal_trend"].notna()]
    print(f"{pos}:")
    flagrow(d[d["causal_trend"] <= -0.08], "decliners (trend <= -0.08)")
    flagrow(d[d["causal_trend"].abs() < 0.08], "stable")
    flagrow(d[d["causal_trend"] >= 0.08], "risers (trend >= +0.08)")
print("Age-28+ full-workload veterans (all positions):")
v = st[(st["age"] >= 28) & st["full_load"] & st["causal_trend"].notna()]
flagrow(v[v["causal_trend"] <= -0.08], "vet decliners")
flagrow(v[v["causal_trend"] > -0.08], "vet non-decliners")

print("\n=== 4. Walk-forward PPG MAE: base vs base + causal trajectory ===")
df = pd.read_sql("SELECT * FROM season_dataset", con)
ns = pd.read_sql("SELECT player_id, season, games FROM nflv_season", con).drop_duplicates(["player_id","season"])
g2 = ns.rename(columns={"games": "prior2_games"}); g2["season"] += 2
df = df.merge(g2, on=["player_id","season"], how="left")
df = MS.add_injury_features(df)
blk = st[["player_id", "season", "skill_true_causal", "causal_trend", "skill_obs"]].copy()
blk.columns = ["player_id", "season", "p_true_skill", "p_true_trend", "p_obs_skill"]
blk["season"] += 1
df = df.merge(blk, on=["player_id","season"], how="left")
df = df[df["next_ppg"].notna() & (df["prior_games"] >= 3)]
A = MS.FEATURES + MS.INJURY_FEATURES
NEW = ["p_true_skill", "p_true_trend", "p_obs_skill"]
GB = dict(objective="regression_l1", n_estimators=300, learning_rate=0.04, num_leaves=20,
          min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
for pos in POS:
    d = df[df["position"] == pos]
    res = {}
    for lab, feats in [("base", A), ("+traj", A + NEW)]:
        preds = []
        for T in range(2016, 2026):
            tr, te = d[d["season"] < T], d[d["season"] == T]
            if len(te) == 0 or len(tr) < 60: continue
            m = lgb.LGBMRegressor(**GB).fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
            preds.append(pd.Series(m.predict(te[feats].astype(float).fillna(-1)), index=te.index))
        p = pd.concat(preds)
        res[lab] = mean_absolute_error(d.loc[p.index, "next_ppg"], p)
    print(f"  {pos}: base {res['base']:.3f}  +traj {res['+traj']:.3f}  "
          f"({100*(res['base']-res['+traj'])/res['base']:+.2f}%)")
con.close()
