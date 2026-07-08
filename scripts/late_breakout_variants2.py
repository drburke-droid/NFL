"""
Round 2 experiments: predict next-draft price directly; QB exclusion; young-only.
G  regressor on next_aav (the literal objective: who gets a big bid next year)
H  G restricted to skill positions (no QB)
I  D-blend restricted to skill positions
J  young-only (years_exp <= 2) versions of the best ranker
Economics: for the chosen ranker, top-10/season -> hit rate, VORP, next AAV,
keeper surplus assuming $1 bid kept at bid+$5 escalation (league empirical).
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
allp = df[df.season <= 2025].copy()
allp["next_aav_t"] = allp.next_aav.fillna(0)
cheap = allp[allp.cheap == 1]


def eval_scores(bt, score, ks=(5, 10, 20)):
    out = {}
    for k in ks:
        h, v, a, y = [], [], [], []
        for _, g in bt.groupby("season"):
            top = g.sort_values(score, ascending=False).head(k)
            h.append(top.hit.mean()); v.append(top.vorp.mean())
            a.append(top.next_aav.fillna(1).mean()); y.append((top.years_exp <= 2).mean())
        out[f"p@{k}"] = np.mean(h)
        if k == 10:
            out.update({"vorp@10": np.mean(v), "nextAAV@10": np.mean(a), "young@10": np.mean(y)})
    return out


def wf(pool, target, kind="clf", train_pool=None):
    tr_all = train_pool if train_pool is not None else pool
    preds = []
    for Y in YEARS:
        tr = tr_all[tr_all.season < Y]; te = pool[pool.season == Y]
        if kind == "clf":
            m = lgb.LGBMClassifier(**P).fit(tr[FEATS], tr[target])
            s = m.predict_proba(te[FEATS])[:, 1]
        else:
            m = lgb.LGBMRegressor(**P).fit(tr[FEATS], tr[target])
            s = m.predict(te[FEATS])
        t = te.copy(); t["score"] = s; preds.append(t)
    return pd.concat(preds)


rows = {}
btG = wf(cheap, "next_aav_t", "reg", train_pool=allp)
rows["G next-AAV reg (all pos)"] = eval_scores(btG, "score")

sk = cheap[cheap.position != "QB"]
btH = wf(sk, "next_aav_t", "reg", train_pool=allp[allp.position != "QB"])
rows["H next-AAV reg (skill)"] = eval_scores(btH, "score")

btB = wf(sk, "hit", "clf", train_pool=allp[allp.position != "QB"])
btI = btB.copy()
btI["r_model"] = btI.groupby("season").score.rank(pct=True)
btI["r_ffa"] = btI.groupby("season").ffa_points.rank(pct=True).fillna(0)
btI["score"] = btI.r_model + btI.r_ffa
rows["I blend clf+ffa (skill)"] = eval_scores(btI, "score")

# blend of H + I (price reg + hit clf + ffa)
btJ = btI.copy()
btJ["r_price"] = btH.groupby("season").score.rank(pct=True).values
btJ["score"] = btJ.r_model + btJ.r_ffa + btJ.r_price
rows["J triple blend (skill)"] = eval_scores(btJ, "score")

# FFA baseline on skill pool
btE = sk.copy(); btE["score"] = btE.ffa_points.fillna(-1)
rows["E ffa_points (skill)"] = eval_scores(btE, "score")

# young-only pool with the best blend
yo = btJ[btJ.years_exp <= 2]
rows["J young<=2 only"] = eval_scores(yo, "score")
yoE = btE[btE.years_exp <= 2]
rows["E ffa young<=2"] = eval_scores(yoE, "score")

res = pd.DataFrame(rows).T
print(res.round(3).to_string())

# --- economics for chosen ranker (J triple blend, skill) --------------------
print("\ntop-10/season economics (J triple blend):")
econ = []
for y, g in btJ.groupby("season"):
    top = g.sort_values("score", ascending=False).head(10)
    surplus = np.maximum(0, top.next_aav.fillna(0) - 6)      # $1 bid kept at ~$6
    econ.append({"season": y, "hits": int(top.hit.sum()), "mean_vorp": top.vorp.mean(),
                 "mean_nextAAV": top.next_aav.fillna(0).mean(),
                 "keeper_surplus_sum": surplus.sum(),
                 "names": ", ".join(f"{r['name']}{'*' if r.hit else ''}"
                                     for _, r in top.iterrows())})
e = pd.DataFrame(econ)
print(e[["season", "hits", "mean_vorp", "mean_nextAAV", "keeper_surplus_sum"]].round(2).to_string(index=False))
for _, r in e.iterrows():
    print(f"{r.season}: {r.names}")

# random-dart benchmark economics
rnd = []
for y, g in sk.groupby("season"):
    surplus = np.maximum(0, g.next_aav.fillna(0) - 6)
    rnd.append({"hit": g.hit.mean(), "vorp": g.vorp.mean(),
                "nextAAV": g.next_aav.fillna(0).mean(), "surplus": surplus.mean()})
r = pd.DataFrame(rnd).mean()
print("\nrandom cheap dart (per pick):", r.round(3).to_dict())

btJ.to_pickle(os.path.join(OUT, "late_breakout_bt_best.pkl"))
print("saved btJ")
