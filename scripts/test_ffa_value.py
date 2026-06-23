"""
Does the FantasyFootballAnalytics consensus add value to the season model?

Two questions:
  1. Benchmark — how well does FFA consensus rank/projection predict actual PPG,
     vs our prior-year model and vs ADP?
  2. Feature — does adding FFA features beat prior-year-only, and does our model
     add anything on top of the FFA consensus (FFA-anchored vs FFA-alone)?

Walk-forward by season over the FFA window. FFA has 12 full seasons (2014-15 sparse).
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from model_season import FEATURES, POS, project_games, add_finish_vbd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
PARAMS = dict(objective="regression_l1", n_estimators=400, learning_rate=0.03, num_leaves=24,
              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
FFA = ["ffa_points","ffa_vor","ffa_floor","ffa_ceiling","ffa_sd","ffa_uncertainty",
       "ffa_adp","ffa_pos_rank","ffa_rank","ffa_tier","ffa_dropoff"]
PRIOR = FEATURES + ["pos_id"]
FFA_ONLY = FFA + ["pos_id"]
ANCHOR = FEATURES + FFA + ["pos_id"]


def main():
    con = sqlite3.connect(DB)
    ds = pd.read_sql("SELECT * FROM season_dataset", con)
    ffa = pd.read_sql("SELECT * FROM nflv_ffa_proj WHERE player_id IS NOT NULL", con)
    adp = pd.read_sql("SELECT season,player_id,pos_rank adp_pos_rank FROM nflv_adp WHERE player_id IS NOT NULL", con)
    con.close()
    ffa = ffa.sort_values("ffa_points", ascending=False).drop_duplicates(["season","player_id"])
    ffa = ffa.drop(columns=[c for c in ["position","player"] if c in ffa.columns])

    df = ds.merge(ffa, on=["season","player_id"], how="inner")
    df = df[df["next_ppg"].notna() & df["ffa_points"].notna()].copy()
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    # exclude the two sparse FFA seasons from coverage stats
    print(f"FFA-overlap player-seasons: {len(df):,}  seasons {sorted(df.season.unique())}")
    print("per season:", df.groupby("season").size().to_dict())

    def walk(feats, test_seasons):
        out=[]
        for T in test_seasons:
            tr=df[df.season<T]; te=df[df.season==T]
            if len(te)==0 or len(tr)<150: continue
            m=lgb.LGBMRegressor(**PARAMS).fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
            t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
        return pd.concat(out, ignore_index=True)

    TEST=list(range(2016,2026))
    P=walk(PRIOR,TEST); F=walk(FFA_ONLY,TEST); A=walk(ANCHOR,TEST)

    def ev(p):
        s=p.dropna(subset=["pred","next_ppg"])
        mae=mean_absolute_error(s["next_ppg"],s["pred"]); rho=spearmanr(s["pred"],s["next_ppg"])[0]
        fin=[spearmanr(g["pred"],-g["next_pos_finish"])[0] for (_,_),g in s.groupby(["season","position"]) if len(g)>=12]
        return mae,rho,np.mean(fin)

    print("\n=== Season-PPG models, walk-forward (test 2016-2025, same players) ===")
    print(f"{'model':16s} {'MAE':>7s} {'rho(PPG)':>9s} {'rank-finish':>12s}")
    for name,p in [("prior-year",P),("FFA consensus",F),("FFA-anchored",A)]:
        mae,rho,fin=ev(p); print(f"{name:16s} {mae:7.3f} {rho:9.3f} {fin:12.3f}")

    # raw FFA (no model) ranking quality
    raw=[]
    for (_,_),g in df[df.season>=2016].groupby(["season","position"]):
        if len(g)>=12: raw.append(spearmanr(-g["ffa_pos_rank"], -g["next_pos_finish"])[0])
    print(f"\n  Raw FFA pos_rank vs actual finish (rank Spearman): {np.mean(raw):.3f}")

    # by position: FFA-anchored vs FFA-alone
    print("\n  MAE by position (FFA-alone -> FFA-anchored):")
    for pos in POS:
        f=F[F.position==pos]; a=A[A.position==pos]
        mf=mean_absolute_error(f['next_ppg'],f['pred']); ma=mean_absolute_error(a['next_ppg'],a['pred'])
        print(f"    {pos}: {mf:.2f} -> {ma:.2f} ({100*(mf-ma)/mf:+.1f}%)")

    # head-to-head vs ADP on the 2021-25 subset (where ADP exists)
    sub=df.merge(adp,on=["season","player_id"],how="inner")
    sub=sub[sub.season>=2021]
    if len(sub):
        print(f"\n=== ADP vs FFA head-to-head (2021-25 overlap, n={len(sub):,}) — rank vs actual finish ===")
        for name,col,sign in [("ADP pos_rank","adp_pos_rank",-1),("FFA pos_rank","ffa_pos_rank",-1),
                              ("FFA points","ffa_points",1)]:
            v=[spearmanr(sign*g[col],-g["next_pos_finish"])[0] for (_,_),g in sub.groupby(["season","position"]) if len(g)>=10]
            print(f"  {name:16s} {np.mean(v):.3f}")

    # persist FFA-anchored predictions (best season model) w/ finish + VBD
    con2=sqlite3.connect(DB)
    names=pd.read_sql("SELECT player_id,player_display_name FROM nflv_season",con2).drop_duplicates("player_id")
    a=A.copy(); a["proj_games"]=project_games(a)
    a=add_finish_vbd(a.rename(columns={"pred":"pred_ppg"}),"pred_ppg","proj_games","pred").merge(names,on="player_id",how="left")
    keep=["player_id","player_display_name","position","season","team","ffa_points","ffa_pos_rank",
          "prior_ppg","pred_ppg","next_ppg","finish_pred","next_pos_finish","vbd_pred"]
    a[[c for c in keep if c in a.columns]].to_sql("season_ffa_predictions",con2,if_exists="replace",index=False)
    con2.close()

    # feature importance of anchored
    mfull=lgb.LGBMRegressor(**PARAMS).fit(df[ANCHOR].astype(float).fillna(-1), df["next_ppg"])
    imp=pd.DataFrame({"feature":ANCHOR,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"ffa_anchored_importance.csv"),index=False)
    print("\n  Top 12 features in FFA-anchored model:")
    print(imp.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
