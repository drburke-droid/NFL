"""
Head-to-head: FFA-anchored season model using the STANDARD-scored consensus
(nflv_ffa_proj, the current production anchor) vs the LEAGUE-scored consensus
(nflv_ffa_league, PPR + 6-pt pass TD) vs BOTH sets of features together.

Walk-forward by season (test 2017-2025; league history starts 2016), on the
identical player set (inner join of both tables), predicting next-season PPG.
Same protocol as test_ffa_value.py.
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from model_season import FEATURES, POS

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
PARAMS = dict(objective="regression_l1", n_estimators=400, learning_rate=0.03, num_leaves=24,
              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
FFA_COLS = ["ffa_points","ffa_vor","ffa_floor","ffa_ceiling","ffa_sd","ffa_uncertainty",
            "ffa_adp","ffa_pos_rank","ffa_rank","ffa_tier","ffa_dropoff"]


def load_ffa(con, table, suffix):
    f = pd.read_sql(f"SELECT * FROM {table} WHERE player_id IS NOT NULL", con)
    f = f.sort_values("ffa_points", ascending=False).drop_duplicates(["season","player_id"])
    f = f[["season","player_id"] + FFA_COLS]
    return f.rename(columns={c: c+suffix for c in FFA_COLS})


def main():
    con = sqlite3.connect(DB)
    ds = pd.read_sql("SELECT * FROM season_dataset", con)
    std = load_ffa(con, "nflv_ffa_proj", "_s")
    lg  = load_ffa(con, "nflv_ffa_league", "_l")
    con.close()

    df = ds.merge(std, on=["season","player_id"], how="inner") \
           .merge(lg,  on=["season","player_id"], how="inner")
    df = df[df["next_ppg"].notna()].copy()
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    print(f"common player-seasons: {len(df):,}  seasons {sorted(df.season.unique())}")

    STD  = FEATURES + [c+"_s" for c in FFA_COLS] + ["pos_id"]
    LGF  = FEATURES + [c+"_l" for c in FFA_COLS] + ["pos_id"]
    BOTH = FEATURES + [c+"_s" for c in FFA_COLS] + [c+"_l" for c in FFA_COLS] + ["pos_id"]

    def walk(feats, test_seasons):
        out=[]
        for T in test_seasons:
            tr=df[df.season<T]; te=df[df.season==T]
            if len(te)==0 or len(tr)<150: continue
            m=lgb.LGBMRegressor(**PARAMS).fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
            t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
        return pd.concat(out, ignore_index=True)

    TEST=list(range(2017,2026))
    runs = {"anchored-STANDARD": walk(STD,TEST),
            "anchored-LEAGUE":   walk(LGF,TEST),
            "anchored-BOTH":     walk(BOTH,TEST)}

    def ev(p):
        s=p.dropna(subset=["pred","next_ppg"])
        mae=mean_absolute_error(s["next_ppg"],s["pred"]); rho=spearmanr(s["pred"],s["next_ppg"])[0]
        fin=[spearmanr(g["pred"],-g["next_pos_finish"])[0] for (_,_),g in s.groupby(["season","position"]) if len(g)>=12]
        return mae,rho,np.mean(fin)

    print("\n=== Season-PPG models, walk-forward (test 2017-2025, identical players) ===")
    print(f"{'model':20s} {'MAE':>7s} {'rho(PPG)':>9s} {'rank-finish':>12s}")
    for name,p in runs.items():
        mae,rho,fin=ev(p); print(f"{name:20s} {mae:7.3f} {rho:9.3f} {fin:12.3f}")

    print("\n  MAE by position:")
    print(f"    {'pos':4s} {'STANDARD':>9s} {'LEAGUE':>9s} {'BOTH':>9s}  (best)")
    for pos in POS:
        maes={}
        for name,p in runs.items():
            g=p[p.position==pos]; maes[name]=mean_absolute_error(g["next_ppg"],g["pred"])
        best=min(maes,key=maes.get)
        print(f"    {pos:4s} {maes['anchored-STANDARD']:9.3f} {maes['anchored-LEAGUE']:9.3f} {maes['anchored-BOTH']:9.3f}  {best.replace('anchored-','')}")

    print("\n  rank-vs-finish by position (higher = better):")
    print(f"    {'pos':4s} {'STANDARD':>9s} {'LEAGUE':>9s} {'BOTH':>9s}")
    for pos in POS:
        row=[]
        for name,p in runs.items():
            g=p[p.position==pos]
            v=[spearmanr(x["pred"],-x["next_pos_finish"])[0] for (_,_),x in g.groupby(["season","position"]) if len(x)>=12]
            row.append(np.mean(v))
        print(f"    {pos:4s} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}")


if __name__ == "__main__":
    main()
