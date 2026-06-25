"""
VALUE LEAP — the season-level analog of the weekly explosion model.

Goal: among players who were CHEAP / undraftable-caliber last year, find the ones
who will break out into a startable asset this season (which is what makes them
"drafted much higher" the following year).

Population: prior_ppg below a draftable line (LOW) with >=3 games (had a cup of
coffee but wasn't a fantasy asset).
Leap target: next_ppg reaches startable (HI) AND jumps >= +4 PPG.

Features synthesize everything we built — base profile + late-season surge
(half-trend) + opportunity (vacated/incoming touches). Walk-forward, reported like
explosion: base rate, AUC, precision@topK, lift.
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
LOW = {"QB": 12, "RB": 7, "WR": 7, "TE": 5}      # cheap / undraftable-caliber last year
HI = {"QB": 16, "RB": 11, "WR": 11, "TE": 8}     # clearly startable this year
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]
OPP = ["vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
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
    df["low"] = df.prior_ppg < df.position.map(LOW)
    df["leap"] = ((df.next_ppg >= df.position.map(HI)) & ((df.next_ppg - df.prior_ppg) >= 4)).astype(int)
    pool = df[df.low].copy()
    print(f"Cheap-player pool: {len(pool):,} player-seasons | base leap rate {pool.leap.mean():.1%}\n")

    FEATS = MS.FEATURES + HT + OPP
    preds = []
    for pos in POS:
        d = pool[pool.position == pos]
        for T in range(2017, 2026):
            tr, te = d[d.season < T], d[d.season == T]
            if len(te) == 0 or tr.leap.sum() < 8: continue
            spw = (len(tr) - tr.leap.sum()) / max(tr.leap.sum(), 1)
            m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw, **GB)
            m.fit(tr[FEATS].astype(float).fillna(-1), tr.leap)
            te = te.copy(); te["p"] = m.predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1]
            preds.append(te)
    P = pd.concat(preds, ignore_index=True)
    base = P.leap.mean()
    auc = roc_auc_score(P.leap, P.p); ap = average_precision_score(P.leap, P.p)
    print(f"Walk-forward (2017-2025): n={len(P)}, base rate {base:.1%}, AUC {auc:.3f}, AP {ap:.3f}")
    for k in [0.05, 0.10, 0.20]:
        thr = P.p.quantile(1 - k); top = P[P.p >= thr]
        prec = top.leap.mean()
        print(f"  precision@top{int(k*100):2d}%: {prec:.1%}  (lift {prec/base:.2f}x, n={len(top)})")

    # how good were the players the model flagged? (mean next PPG of top decile vs pool)
    top10 = P[P.p >= P.p.quantile(0.90)]
    print(f"\n  Flagged top-10%: mean next PPG {top10.next_ppg.mean():.1f} vs pool {P.next_ppg.mean():.1f}")
    print("\n=== Biggest correctly-flagged leaps (top model scores that hit) ===")
    nm = pd.read_sql("SELECT DISTINCT player_id, player_display_name FROM nflv_season", sqlite3.connect(DB)).set_index("player_id").player_display_name.to_dict()
    hits = P[P.leap == 1].sort_values("p", ascending=False).head(12)
    for _, r in hits.iterrows():
        print(f"  {str(nm.get(r.player_id))[:20]:20s} {r.position} {int(r.season)}: prior {r.prior_ppg:.1f} -> {r.next_ppg:.1f} PPG  (score {r.p:.2f})")

    # feature importance (which signals drive the leap)
    m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=(len(pool)-pool.leap.sum())/max(pool.leap.sum(),1), **GB)
    m.fit(pool[FEATS].astype(float).fillna(-1), pool.leap)
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\nTop drivers:", ", ".join(f"{k}" for k in imp.head(10).index))


if __name__ == "__main__":
    main()
