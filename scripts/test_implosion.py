"""
IMPLOSION — the mirror of value-leap. Among players who were drafted HIGH last year
(startable studs), predict who collapses to below-startable this season — the ones
who get "drafted much later" the following year.

Population: prior_ppg above a clearly-startable line.
Collapse target: next_ppg falls below the position bust line AND drops >= 4 PPG.

Features synthesize everything: base profile + late-season fade (half-trend) +
INCOMING competition (opportunity) + TD/efficiency regression + age. Walk-forward,
reported like explosion/value-leap, and compared to the naive baseline of just
ranking by our existing calibrated bust% — does a dedicated model add anything?
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]
HIGH = {"QB": 16, "RB": 12, "WR": 12, "TE": 8}        # drafted-high / startable stud last year
BUST = {"QB": 14, "RB": 9, "WR": 9, "TE": 7}          # below this = not startable
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]
OPP = ["vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
FEATS = MS.FEATURES + HT + OPP
GB = dict(n_estimators=400, learning_rate=0.03, num_leaves=24, min_child_samples=25,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr FROM nflv_opportunity", con)
    con.close()
    df = df.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    df["collapse"] = ((df.next_ppg < df.position.map(BUST)) & ((df.prior_ppg - df.next_ppg) >= 4)).astype(int)
    pool = df[df.prior_ppg >= df.position.map(HIGH)].copy()
    print(f"High-drafted pool: {len(pool):,} player-seasons | base collapse rate {pool.collapse.mean():.1%}\n")

    preds = []
    for pos in POS:
        d = pool[pool.position == pos]
        for T in range(2017, 2026):
            tr, te = d[d.season < T], d[d.season == T]
            if len(te) == 0 or tr.collapse.sum() < 8: continue
            spw = (len(tr) - tr.collapse.sum()) / max(tr.collapse.sum(), 1)
            m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw, **GB)
            m.fit(tr[FEATS].astype(float).fillna(-1), tr.collapse)
            te = te.copy(); te["p"] = m.predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1]
            preds.append(te)
    P = pd.concat(preds, ignore_index=True)
    base = P.collapse.mean()
    print(f"Walk-forward (2017-2025): n={len(P)}, base rate {base:.1%}, AUC {roc_auc_score(P.collapse,P.p):.3f}, AP {average_precision_score(P.collapse,P.p):.3f}")
    for k in [0.10, 0.20, 0.30]:
        thr = P.p.quantile(1 - k); top = P[P.p >= thr]
        print(f"  precision@top{int(k*100):2d}%: {top.collapse.mean():.1%}  (lift {top.collapse.mean()/base:.2f}x, n={len(top)})")

    # naive baselines: does the model beat just 'older' or 'volatile' or repeat-rank?
    print("\n  Baselines (AUC for ranking collapse):")
    for col in ["age", "prior_cv", "prior_total_tds"]:
        if col in P: print(f"    {col:16s}: {roc_auc_score(P.collapse, P[col].fillna(P[col].median())):.3f}")

    print("\n=== Biggest correctly-flagged implosions (top scores that hit) ===")
    nm = pd.read_sql("SELECT DISTINCT player_id, player_display_name FROM nflv_season", sqlite3.connect(DB)).set_index("player_id").player_display_name.to_dict()
    for _, r in P[P.collapse == 1].sort_values("p", ascending=False).head(12).iterrows():
        print(f"  {str(nm.get(r.player_id))[:20]:20s} {r.position} {int(r.season)}: {r.prior_ppg:.1f} -> {r.next_ppg:.1f} PPG  (score {r.p:.2f})")

    m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=(len(pool)-pool.collapse.sum())/max(pool.collapse.sum(),1), **GB)
    m.fit(pool[FEATS].astype(float).fillna(-1), pool.collapse)
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\nTop drivers:", ", ".join(imp.head(10).index))


if __name__ == "__main__":
    main()
