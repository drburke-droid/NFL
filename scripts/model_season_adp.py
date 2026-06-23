"""
ADP-anchored season model — can we beat the market?

The benchmark showed preseason ADP beats the prior-year-only model. Here we test
the one configuration that could beat ADP: use ADP as the anchor feature AND let
the model apply prior-year regression/role deltas (TD regression, CV, age, snap
share, 2yr history) on top.

Three models, same GB, same OOS test (walk-forward over ADP years 2021-25,
test 2023/24/25), on the SAME veteran player-seasons (have both ADP + priors):
  M_market   : ECR + pos_rank only            (pure market)
  M_prior    : prior-year features only        (no market)
  M_anchored : ADP + prior-year features       (market + deltas)

If M_anchored beats M_market, the deltas add value over the market.
"""
import os, sqlite3
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from model_season import FEATURES, project_games, add_finish_vbd, POS

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
PARAMS = dict(objective="regression_l1", n_estimators=400, learning_rate=0.03, num_leaves=24,
              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

MARKET = ["ecr", "pos_rank", "pos_id"]
PRIOR  = FEATURES + ["pos_id"]
ANCHOR = FEATURES + ["ecr", "pos_rank", "pos_id"]


def main():
    con = sqlite3.connect(DB)
    ds = pd.read_sql("SELECT * FROM season_dataset", con)
    adp = pd.read_sql("SELECT season,player_id,ecr,pos_rank,adp_overall FROM nflv_adp WHERE player_id IS NOT NULL", con)
    con.close()

    df = ds.merge(adp, on=["season","player_id"], how="inner")
    df = df[df["next_ppg"].notna()].copy()
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    df["proj_games"] = project_games(df)
    print(f"ADP-overlap veteran player-seasons: {len(df):,}  seasons {sorted(df.season.unique())}")

    def walk(feats):
        out=[]
        for T in [2023,2024,2025]:
            tr=df[df.season<T]; te=df[df.season==T]
            if len(te)==0: continue
            m=lgb.LGBMRegressor(**PARAMS)
            m.fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
            t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
        return pd.concat(out, ignore_index=True)

    preds={name:walk(f) for name,f in [("market",MARKET),("prior",PRIOR),("anchored",ANCHOR)]}

    def evals(p):
        s=p.dropna(subset=["pred","next_ppg"])
        mae=mean_absolute_error(s["next_ppg"],s["pred"])
        rho=spearmanr(s["pred"],s["next_ppg"])[0]
        # within season/pos rank-vs-actual-finish
        fin=[]
        for (yr,pos),g in s.groupby(["season","position"]):
            if len(g)<12: continue
            fin.append(spearmanr(g["pred"], -g["next_pos_finish"])[0])
        return mae, rho, np.mean(fin)

    print("\n=== OOS comparison (test 2023-25, same players) ===")
    print(f"{'model':12s} {'MAE':>7s} {'rho(PPG)':>9s} {'rank-vs-finish':>15s}")
    rows={}
    for name in ["market","prior","anchored"]:
        mae,rho,fin=evals(preds[name]); rows[name]=(mae,rho,fin)
        print(f"{name:12s} {mae:7.3f} {rho:9.3f} {fin:15.3f}")
    dm = 100*(rows['market'][0]-rows['anchored'][0])/rows['market'][0]
    print(f"\nanchored vs market: MAE {dm:+.1f}%, rho {rows['anchored'][1]-rows['market'][1]:+.3f}, "
          f"rank {rows['anchored'][2]-rows['market'][2]:+.3f}")

    # by position (anchored vs market MAE)
    print("\n  MAE by position (market -> anchored):")
    for p in POS:
        mm=mean_absolute_error(preds['market'].query("position==@p")['next_ppg'],
                               preds['market'].query("position==@p")['pred'])
        aa=mean_absolute_error(preds['anchored'].query("position==@p")['next_ppg'],
                               preds['anchored'].query("position==@p")['pred'])
        print(f"    {p}: {mm:.2f} -> {aa:.2f} ({100*(mm-aa)/mm:+.1f}%)")

    # top-N finish hit rate
    print("\n  Top-N hit rate (market | anchored):")
    for p,N in [("QB",12),("RB",24),("WR",36),("TE",12)]:
        def hit(pr):
            h=t=0
            d=pr[pr.position==p]
            for yr,g in d.groupby("season"):
                act=set(g.nsmallest(N,"next_pos_finish")["player_id"])
                if not act: continue
                h+=len(set(g.nlargest(N,"pred")["player_id"])&act); t+=len(act)
            return h/t if t else float("nan")
        print(f"    {p} top-{N}: {hit(preds['market']):.1%}  {hit(preds['anchored']):.1%}")

    # importance of anchored model (which deltas it uses on top of ADP)
    full=lgb.LGBMRegressor(**PARAMS).fit(df[ANCHOR].astype(float).fillna(-1), df["next_ppg"])
    imp=pd.DataFrame({"feature":ANCHOR,"importance":full.feature_importances_}).sort_values("importance",ascending=False)
    os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"season_adp_importance.csv"),index=False)
    print("\n  Top features in anchored model:")
    print(imp.head(12).to_string(index=False))

    # persist anchored predictions w/ finish/VBD
    a=preds["anchored"].rename(columns={"pred":"pred_ppg"})
    a=add_finish_vbd(a,"pred_ppg","proj_games","pred")
    keep=["player_id","player_display_name","position","season","team","ecr","pos_rank",
          "prior_ppg","pred_ppg","next_ppg","finish_pred","next_pos_finish","vbd_pred"]
    con2=sqlite3.connect(DB); a[[c for c in keep if c in a.columns]].to_sql(
        "season_adp_predictions",con2,if_exists="replace",index=False); con2.close()
    print("\nSaved season_adp_predictions + importance.")


if __name__ == "__main__":
    main()
