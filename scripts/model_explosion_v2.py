"""
Avenue 2 explosion (tail) model — now on the FULL 2012-2025 window.

Uses nflv_weekly + nflv_game_lines (historical spread/total) + xFP. Explosion =
PPR >= rolling_mean + 2*rolling_std (rolling mean >= 3). Tests whether the new
game-environment + xFP-variance signals improve TAIL detection, with ~5x the
training data vs the 3-season version. Reports full window + 2023-25 subset.

Writes explosion_v2_predictions + metrics.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
POS = ["QB","RB","WR","TE"]
START_TEST = 2014


def main():
    con = sqlite3.connect(DB)
    wk = pd.read_sql("""SELECT player_id,player_display_name,position,team,opponent_team opponent,
                        season,week,targets,carries,fantasy_points_ppr
                        FROM nflv_weekly WHERE season_type='REG' AND season>=2012
                        AND position IN ('QB','RB','WR','TE')""", con)
    gl = pd.read_sql("""SELECT season,week,team,team_spread,game_total,implied_team_total
                        FROM nflv_game_lines WHERE game_type='REG'""", con)
    xfp = pd.read_sql("SELECT player_id,CAST(season AS INT) season,week,total_fantasy_points_exp xfp FROM nflv_ff_opp", con)
    con.close()

    df = wk.merge(gl, on=["season","week","team"], how="left").merge(xfp, on=["player_id","season","week"], how="left")
    df["order"] = df["season"]*100 + df["week"]
    df = df.sort_values(["player_id","order"]).reset_index(drop=True)
    df["opp_ct"] = df["targets"].fillna(0) + df["carries"].fillna(0)

    g=df.groupby("player_id")
    df["roll_mean"]=g["fantasy_points_ppr"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_std"] =g["fantasy_points_ppr"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std())
    df["roll_opp"] =g["opp_ct"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_xfp"] =g["xfp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_xfp_std"]=g["xfp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std())
    df["luck"]=df["fantasy_points_ppr"]-df["xfp"]
    df["roll_luck"]=g["luck"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["cv"]=df["roll_std"]/df["roll_mean"].replace(0,np.nan)
    df["pos_id"]=df["position"].map({p:i for i,p in enumerate(POS)})
    df["is_qb"]=(df["position"]=="QB").astype(int)
    df["ix_total_x_std"]=df["game_total"]*df["roll_std"]
    df["ix_itt_x_opp"]=df["implied_team_total"]*df["roll_opp"]

    df=df[df["roll_mean"]>=3].dropna(subset=["roll_std"]).copy()
    df["explosion"]=(df["fantasy_points_ppr"]>=df["roll_mean"]+2*df["roll_std"]).astype(int)
    print(f"rows: {len(df):,}  seasons {int(df.season.min())}-{int(df.season.max())}  base rate {df['explosion'].mean():.3f}")

    BASE=["roll_mean","roll_std","cv","roll_opp","pos_id"]
    FULL=BASE+["roll_xfp","roll_xfp_std","roll_luck","game_total","implied_team_total",
               "team_spread","ix_total_x_std","ix_itt_x_opp","is_qb"]
    params=dict(objective="binary",n_estimators=400,learning_rate=0.03,num_leaves=31,
                min_child_samples=60,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)

    def walk(feats):
        out=[]
        for S in range(START_TEST,2026):
            for W in range(6,23):
                te=df[(df.season==S)&(df.week==W)]; tr=df[df.order<S*100+W]
                if len(te)<10 or len(tr)<2000 or tr["explosion"].sum()<40: continue
                spw=(len(tr)-tr["explosion"].sum())/max(tr["explosion"].sum(),1)
                m=lgb.LGBMClassifier(scale_pos_weight=spw,**params)
                m.fit(tr[feats].astype(float).fillna(-1),tr["explosion"])
                t=te.copy(); t["prob"]=m.predict_proba(te[feats].astype(float).fillna(-1))[:,1]; out.append(t)
        return pd.concat(out,ignore_index=True)

    def metrics(p):
        y=p["explosion"]; s=p["prob"]; k=int(0.1*len(p))
        return (roc_auc_score(y,s), average_precision_score(y,s),
                p.nlargest(k,"prob")["explosion"].mean(), p.nlargest(k,"prob")["explosion"].mean()/y.mean())

    res={}
    base=walk(BASE); full=walk(FULL)
    for name,p in [("BASE",base),("FULL",full)]:
        auc,ap,prec,lift=metrics(p); res[name]={"auc":round(auc,4),"ap":round(ap,4),"prec@10%":round(prec,3),"lift":round(lift,2),"n":len(p)}
        print(f"  {name}: AUC={auc:.4f} AP={ap:.4f} precision@top10%={prec:.3f} ({lift:.2f}x)")
    # 2023-25 subset
    for name,p in [("BASE",base),("FULL",full)]:
        sub=p[p.season>=2023]; auc,ap,prec,lift=metrics(sub)
        print(f"  [2023-25] {name}: AUC={auc:.4f} AP={ap:.4f} prec@10%={prec:.3f}")

    keep=["player_id","player_display_name","position","team","opponent","season","week",
          "game_total","implied_team_total","roll_mean","prob","explosion","fantasy_points_ppr"]
    con2=sqlite3.connect(DB); full[keep].to_sql("explosion_v2_predictions",con2,if_exists="replace",index=False); con2.close()
    mfull=lgb.LGBMClassifier(scale_pos_weight=(len(df)-df.explosion.sum())/df.explosion.sum(),**params)
    mfull.fit(df[FULL].astype(float).fillna(-1),df["explosion"])
    imp=pd.DataFrame({"feature":FULL,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"explosion_v2_importance.csv"),index=False)
    json.dump(res,open(os.path.join(OUT,"explosion_v2_metrics.json"),"w"),indent=2)
    print("  top features:", ", ".join(imp.head(8)["feature"]))
    print("Saved explosion_v2_predictions + metrics. (3-season ref: BASE AUC 0.655 / FULL 0.662)")


if __name__ == "__main__":
    main()
