"""
Model-design experiments for the late-breakout study (loads the frame pickle).
Question: what ranking of the CHEAP pool (adp>120 & aav<=$2) maximizes
precision@K on VORP+ hits, walk-forward 2019-2025?

Variants:
  A  classifier trained on cheap pool only (status quo)
  B  classifier trained on ALL players, scored on cheap
  C  regressor on VORP trained on ALL players, ranked
  D  blend: B prob rank + ffa_points rank
  E  ffa_points alone (the baseline to beat)
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "models")
df = pd.read_pickle(os.path.join(OUT, "late_breakout_frame.pkl"))
df = pd.concat([df, pd.get_dummies(df.position, prefix="pos")], axis=1)

FEATS = ["adp_f", "aav_f", "in_ffa", "ffa_points", "ffa_upside", "ffa_uncertainty",
         "age", "years_exp", "is_rookie", "draft_round", "draft_pick",
         "prior_games", "prior_ppg", "prior2_ppg", "prior_cv", "prior_snap_pct",
         "prior_target_share", "prior_wopr", "prior_receiving_epa", "prior_rushing_epa",
         "prior_carries_pg", "prior_targets_pg", "prior_tds", "team_change",
         "ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_d_tch", "ht_slope",
         "vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets",
         "pos_QB", "pos_RB", "pos_TE", "pos_WR"]
for c in FEATS:
    if c not in df.columns: df[c] = np.nan

P = dict(n_estimators=350, learning_rate=0.03, num_leaves=31, min_child_samples=40,
         subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
YEARS = range(2019, 2026)


def eval_scores(bt, score, ks=(5, 10, 20)):
    out = {}
    for k in ks:
        hits, vorps, aavs = [], [], []
        for y, g in bt.groupby("season"):
            top = g.sort_values(score, ascending=False).head(k)
            hits.append(top.hit.mean()); vorps.append(top.vorp.mean())
            aavs.append(top.next_aav.fillna(1).mean())
        out[f"p@{k}"] = np.mean(hits)
        if k == 10:
            out["vorp@10"] = np.mean(vorps); out["nextAAV@10"] = np.mean(aavs)
    return out


rows, bts = {}, {}
cheap = df[(df.cheap == 1) & (df.season <= 2025)]

# A: cheap-only classifier
preds = []
for Y in YEARS:
    tr = cheap[cheap.season < Y]; te = cheap[cheap.season == Y]
    m = lgb.LGBMClassifier(**P).fit(tr[FEATS], tr.hit)
    t = te.copy(); t["sA"] = m.predict_proba(te[FEATS])[:, 1]; preds.append(t)
btA = pd.concat(preds); rows["A cheap-only clf"] = eval_scores(btA, "sA"); bts["A"] = btA

# B: all-player classifier, scored on cheap
allp = df[df.season <= 2025]
preds = []
for Y in YEARS:
    tr = allp[allp.season < Y]
    te = cheap[cheap.season == Y]
    m = lgb.LGBMClassifier(**P).fit(tr[FEATS], tr.hit)
    t = te.copy(); t["sB"] = m.predict_proba(te[FEATS])[:, 1]; preds.append(t)
btB = pd.concat(preds); rows["B all-pop clf"] = eval_scores(btB, "sB"); bts["B"] = btB

# C: all-player VORP regressor
preds = []
for Y in YEARS:
    tr = allp[allp.season < Y]
    te = cheap[cheap.season == Y]
    m = lgb.LGBMRegressor(**P).fit(tr[FEATS], tr.vorp)
    t = te.copy(); t["sC"] = m.predict(te[FEATS]); preds.append(t)
btC = pd.concat(preds); rows["C all-pop vorp reg"] = eval_scores(btC, "sC"); bts["C"] = btC

# D: blend of B and ffa_points (rank average, ffa NaN -> worst)
btD = btB.copy()
btD["r_model"] = btD.groupby("season").sB.rank(pct=True)
btD["r_ffa"] = btD.groupby("season").ffa_points.rank(pct=True).fillna(0)
btD["sD"] = 0.5 * btD.r_model + 0.5 * btD.r_ffa
rows["D blend B+ffa"] = eval_scores(btD, "sD")

# E: ffa_points baseline
btE = btB.copy(); btE["sE"] = btE.ffa_points.fillna(-1)
rows["E ffa_points"] = eval_scores(btE, "sE")

# F: ensemble B prob * C vorp rank
btF = btB.copy(); btF["sC"] = btC["sC"].values
btF["sF"] = btF.groupby("season").sB.rank(pct=True) + btF.groupby("season").sC.rank(pct=True)
rows["F B+C ranks"] = eval_scores(btF, "sF")

res = pd.DataFrame(rows).T
print(res.round(3).to_string())

# face validity: top-10 names per year for best variant
best = max(rows, key=lambda k: rows[k]["p@10"])
print(f"\nbest variant: {best}")
key = {"A cheap-only clf": ("A", "sA"), "B all-pop clf": ("B", "sB"),
       "C all-pop vorp reg": ("C", "sC")}.get(best, ("B", "sB"))
bt = bts[key[0]]
for y, g in bt.groupby("season"):
    top = g.sort_values(key[1], ascending=False).head(10)
    names = [f"{r['name']}{'*' if r.hit else ''}" for _, r in top.iterrows()]
    print(f"{y}: " + ", ".join(names))
bt.to_pickle(os.path.join(OUT, "late_breakout_bt_best.pkl"))
print("\n(* = VORP-positive hit)")
