"""Does true skill (Kalman causal estimate) predict NEXT season beyond preseason rankings?

Setup: target season S. Predictors available at draft time:
  - preseason ranking: ffa_points / ffa_pos_rank (nflv_ffa_proj, 2012-25),
    FantasyPros ECR / ADP (nflv_adp, 2021-25)
  - skill_true_causal through S-1 (player_skill_true), causal trend, alpha_skill
Outcome: next_ppg / next_pos_finish in season S (season_dataset row S).
Veterans only (skill requires a prior-season estimate; rookies excluded).

1. Spearman: ranking vs outcome, skill vs outcome (sanity).
2. Partial correlation: skill -> next PPG controlling the preseason ranking.
3. Walk-forward MAE: ranking-only model vs ranking + skill.
4. Tier test: within ADP tiers, do high-skill players beat their rank?
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]

con = sqlite3.connect(DB)
sd = pd.read_sql("""SELECT player_id, season, position, age, prior_ppg, prior_games,
                           next_ppg, next_games, next_pos_finish
                    FROM season_dataset WHERE next_ppg IS NOT NULL""", con)
st = pd.read_sql("SELECT player_id, season, skill_true_causal, plays FROM player_skill_true", con)
al = pd.read_sql("SELECT player_id, season, alpha_skill FROM player_skill_alpha", con)
ffa = pd.read_sql("""SELECT player_id, season, ffa_points, ffa_rank, ffa_pos_rank
                     FROM nflv_ffa_proj WHERE player_id IS NOT NULL""", con)
fp = pd.read_sql("""SELECT player_id, season, ecr, adp_overall, pos_rank
                    FROM nflv_adp WHERE player_id IS NOT NULL""", con)

# skill through S-1 -> usable for target season S
st = st.sort_values(["player_id", "season"])
g = st.groupby("player_id")
st["skill_trend"] = np.where((st["season"] - g["season"].shift(1)) == 1,
                             st["skill_true_causal"] - g["skill_true_causal"].shift(1), np.nan)
st = st.merge(al, on=["player_id", "season"], how="left")
st["season"] += 1
sd = sd.merge(st, on=["player_id", "season"], how="left")
sd = sd.merge(ffa, on=["player_id", "season"], how="left")
sd = sd.merge(fp.drop_duplicates(["player_id", "season"]), on=["player_id", "season"], how="left")
sd["age"] = sd["age"].fillna(sd["age"].median())

vet = sd[sd["skill_true_causal"].notna()].copy()
print(f"veteran rows with skill: {len(vet)}, with ffa rank: {vet['ffa_pos_rank'].notna().sum()}, "
      f"with FP ecr: {vet['ecr'].notna().sum()}")

def zin(d, col, by):
    return d.groupby(by)[col].transform(lambda s: (s - s.mean()) / (s.std() or 1))

def partial(d, xcol, ycol, ctrl):
    d = d.dropna(subset=[xcol, ycol] + ctrl)
    if len(d) < 50: return np.nan, len(d)
    A = np.column_stack([np.ones(len(d))] + [d[c] for c in ctrl])
    rx = d[xcol] - A @ np.linalg.lstsq(A, d[xcol], rcond=None)[0]
    ry = d[ycol] - A @ np.linalg.lstsq(A, d[ycol], rcond=None)[0]
    r, p = pearsonr(rx, ry)
    return r, len(d), p

print("\n=== 1. Spearman with next PPG (veterans, FFA sample) ===")
print(f"{'pos':4s} {'n':>5s} | ffa_pos_rank | skill_causal | alpha")
for pos in POS:
    d = vet[(vet["position"] == pos) & vet["ffa_pos_rank"].notna()]
    r_adp = spearmanr(-d["ffa_pos_rank"], d["next_ppg"])[0]
    r_sk = spearmanr(d["skill_true_causal"], d["next_ppg"])[0]
    da = d.dropna(subset=["alpha_skill"])
    r_al = spearmanr(da["alpha_skill"], da["next_ppg"])[0] if len(da) > 50 else np.nan
    print(f"{pos:4s} {len(d):5d} | {r_adp:+.3f}       | {r_sk:+.3f}       | {r_al:+.3f}")

print("\n=== 2. Partial correlation: skill -> next PPG | preseason ranking ===")
vet["log_posrank"] = np.log(vet["ffa_pos_rank"].clip(lower=1))
vet["log_ecr"] = np.log(vet["ecr"].clip(lower=1))
vet["log_adp"] = np.log(vet["adp_overall"].clip(lower=1))
vet["z_next"] = zin(vet, "next_ppg", ["season", "position"])
print("-- FFA sample (2012-25), ctrl = ffa_points + log pos_rank + age --")
print(f"{'pos':4s} | skill_causal          | skill_trend           | alpha_skill")
for pos in POS:
    d = vet[vet["position"] == pos]
    C = ["ffa_points", "log_posrank", "age"]
    r1 = partial(d, "skill_true_causal", "z_next", C)
    r2 = partial(d, "skill_trend", "z_next", C + ["skill_true_causal"])
    r3 = partial(d, "alpha_skill", "z_next", C)
    def fmt(t): return f"r={t[0]:+.3f} n={t[1]:4d} p={t[2]:.3f}" if len(t) == 3 and not np.isnan(t[0]) else "n/a"
    print(f"{pos:4s} | {fmt(r1)} | {fmt(r2)} | {fmt(r3)}")
print("-- FantasyPros sample (2021-25), ctrl = log ECR + log ADP + age --")
for pos in POS:
    d = vet[vet["position"] == pos]
    C = ["log_ecr", "log_adp", "age"]
    r1 = partial(d, "skill_true_causal", "z_next", C)
    r2 = partial(d, "skill_trend", "z_next", C + ["skill_true_causal"])
    r3 = partial(d, "alpha_skill", "z_next", C)
    def fmt(t): return f"r={t[0]:+.3f} n={t[1]:4d} p={t[2]:.3f}" if len(t) == 3 and not np.isnan(t[0]) else "n/a"
    print(f"{pos:4s} | {fmt(r1)} | {fmt(r2)} | {fmt(r3)}")

print("\n=== 2b. And beyond ranking + prior PPG (is skill just prior production?) ===")
for pos in POS:
    d = vet[vet["position"] == pos]
    C = ["ffa_points", "log_posrank", "age", "prior_ppg"]
    r1 = partial(d, "skill_true_causal", "z_next", C)
    def fmt(t): return f"r={t[0]:+.3f} n={t[1]:4d} p={t[2]:.3f}" if len(t) == 3 and not np.isnan(t[0]) else "n/a"
    print(f"{pos:4s} | {fmt(r1)}")

print("\n=== 3. Walk-forward next-PPG MAE: ranking-only vs + skill (FFA sample) ===")
GB = dict(objective="regression_l1", n_estimators=250, learning_rate=0.05, num_leaves=15,
          min_child_samples=25, subsample=0.8, colsample_bytree=0.9, random_state=0, verbosity=-1)
BASE = ["ffa_points", "log_posrank", "age"]
BASE2 = BASE + ["prior_ppg", "prior_games"]
SK = ["skill_true_causal", "skill_trend", "alpha_skill"]
dfw = vet[vet["ffa_pos_rank"].notna()]
for pos in POS:
    d = dfw[dfw["position"] == pos]
    res = {}
    for lab, feats in [("rank", BASE), ("rank+skill", BASE + SK),
                       ("rank+prior", BASE2), ("rank+prior+skill", BASE2 + SK)]:
        preds = []
        for T in range(2018, 2026):
            tr, te = d[d["season"] < T], d[d["season"] == T]
            if len(te) == 0 or len(tr) < 60: continue
            m = lgb.LGBMRegressor(**GB).fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
            preds.append(pd.Series(m.predict(te[feats].astype(float).fillna(-1)), index=te.index))
        if not preds: res[lab] = np.nan; continue
        p = pd.concat(preds)
        res[lab] = mean_absolute_error(d.loc[p.index, "next_ppg"], p)
    print(f"  {pos}: rank {res['rank']:.3f}  +skill {res['rank+skill']:.3f} "
          f"({100*(res['rank']-res['rank+skill'])/res['rank']:+.2f}%) | "
          f"rank+prior {res['rank+prior']:.3f}  +skill {res['rank+prior+skill']:.3f} "
          f"({100*(res['rank+prior']-res['rank+prior+skill'])/res['rank+prior']:+.2f}%)")

print("\n=== 4. Within ADP tier: does skill pick the outperformers? ===")
# skill above/below position-season median among similar-ranked players
d = vet[vet["ffa_pos_rank"].notna() & vet["next_pos_finish"].notna()].copy()
d["tier"] = pd.cut(d["ffa_pos_rank"], [0, 6, 12, 24, 48, 999],
                   labels=["1-6", "7-12", "13-24", "25-48", "49+"])
d["skill_hi"] = d["skill_true_causal"] > d.groupby(["season", "position", "tier"])["skill_true_causal"].transform("median")
d["beat"] = (d["next_pos_finish"] < d["ffa_pos_rank"]).astype(float)
print(f"{'pos':4s} {'tier':6s} | hi-skill: beat%  next_ppg | lo-skill: beat%  next_ppg | n")
for pos in POS:
    for tier in ["1-6", "7-12", "13-24", "25-48"]:
        t = d[(d["position"] == pos) & (d["tier"] == tier)]
        hi, lo = t[t["skill_hi"]], t[~t["skill_hi"]]
        if len(hi) < 15 or len(lo) < 15: continue
        print(f"{pos:4s} {tier:6s} | {hi['beat'].mean():.0%}  {hi['next_ppg'].mean():5.1f}      "
              f"| {lo['beat'].mean():.0%}  {lo['next_ppg'].mean():5.1f}      | {len(t)}")
con.close()
