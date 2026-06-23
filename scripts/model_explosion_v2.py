"""
Avenue 2 tail model: weekly EXPLOSION (ceiling) prediction — the GPP edge.

Explosion = actual PPR >= rolling_mean + 2*rolling_std (among DFS-relevant players,
rolling mean >= 3). Walk-forward 2023-25. Tests whether the new game-environment +
xFP-variance signals improve TAIL detection (ROC AUC / avg precision / precision@k)
over a rolling-only baseline — where the mean-point model showed no lift.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
POS = ["QB","RB","WR","TE"]
ABBR = {"Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
 "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
 "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
 "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
 "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
 "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
 "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
 "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"}


def main():
    con = sqlite3.connect(DB)
    ps = pd.read_sql("""SELECT event_id,player_id,player_display_name,position,team,opponent,season,week,
                        targets,carries,fantasy_points_ppr FROM player_stats
                        WHERE position IN ('QB','RB','WR','TE')""", con)
    tot = pd.read_sql("SELECT event_id, AVG(point) game_total FROM game_odds WHERE market='totals' GROUP BY event_id", con)
    spr = pd.read_sql("""SELECT event_id,outcome_name,AVG(point) team_spread FROM game_odds
                         WHERE market='spreads' GROUP BY event_id,outcome_name""", con)
    spr["team"]=spr["outcome_name"].map(ABBR)
    odds=spr.merge(tot,on="event_id").dropna(subset=["team"])
    odds["implied_team_total"]=odds["game_total"]/2 - odds["team_spread"]/2
    xfp=pd.read_sql("SELECT player_id,CAST(season AS INT) season,week,total_fantasy_points_exp xfp FROM nflv_ff_opp",con)
    con.close()

    df=ps.merge(odds[["event_id","team","team_spread","game_total","implied_team_total"]],on=["event_id","team"],how="left") \
         .merge(xfp,on=["player_id","season","week"],how="left")
    df["order"]=df["season"]*100+df["week"]
    df=df.sort_values(["player_id","order"]).reset_index(drop=True)
    df["opp"]=df["targets"].fillna(0)+df["carries"].fillna(0)

    g=df.groupby("player_id")
    df["roll_mean"]=g["fantasy_points_ppr"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_std"] =g["fantasy_points_ppr"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std())
    df["roll_opp"] =g["opp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_xfp"] =g["xfp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_xfp_std"]=g["xfp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std())
    df["luck"]=df["fantasy_points_ppr"]-df["xfp"]
    df["roll_luck"]=g["luck"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["cv"]=df["roll_std"]/df["roll_mean"].replace(0,np.nan)
    df["pos_id"]=df["position"].map({p:i for i,p in enumerate(POS)})
    df["is_qb"]=(df["position"]=="QB").astype(int)
    df["ix_total_x_std"]=df["game_total"]*df["roll_std"]
    df["ix_itt_x_opp"]=df["implied_team_total"]*df["roll_opp"]

    # explosion target (DFS-relevant only)
    df=df[df["roll_mean"]>=3].dropna(subset=["roll_std"]).copy()
    df["explosion"]=(df["fantasy_points_ppr"]>=df["roll_mean"]+2*df["roll_std"]).astype(int)
    print(f"rows: {len(df):,}  base explosion rate: {df['explosion'].mean():.3f}")

    BASE=["roll_mean","roll_std","cv","roll_opp","pos_id"]
    FULL=BASE+["roll_xfp","roll_xfp_std","roll_luck","game_total","implied_team_total",
               "team_spread","ix_total_x_std","ix_itt_x_opp","is_qb"]
    params=dict(objective="binary",n_estimators=400,learning_rate=0.03,num_leaves=31,
                min_child_samples=60,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)

    def walk(feats):
        out=[]
        for S in [2023,2024,2025]:
            for W in range(6,23):
                te=df[(df.season==S)&(df.week==W)]; tr=df[df.order<S*100+W]
                if len(te)<10 or len(tr)<800 or tr["explosion"].sum()<20: continue
                spw=(len(tr)-tr["explosion"].sum())/max(tr["explosion"].sum(),1)
                m=lgb.LGBMClassifier(scale_pos_weight=spw,**params)
                m.fit(tr[feats].astype(float).fillna(-1),tr["explosion"])
                t=te.copy(); t["prob"]=m.predict_proba(te[feats].astype(float).fillna(-1))[:,1]; out.append(t)
        return pd.concat(out,ignore_index=True)

    res={}
    for name,feats in [("BASE",BASE),("FULL",FULL)]:
        p=walk(feats); y=p["explosion"]; s=p["prob"]
        auc=roc_auc_score(y,s); ap=average_precision_score(y,s)
        # precision@ top decile of predicted prob
        k=int(0.1*len(p)); topk=p.nlargest(k,"prob")
        prec=topk["explosion"].mean(); lift=prec/y.mean()
        res[name]={"auc":round(auc,4),"ap":round(ap,4),"prec@10%":round(prec,3),"lift":round(lift,2),"n":len(p)}
        print(f"  {name}: AUC={auc:.4f}  AP={ap:.4f}  precision@top10%={prec:.3f} ({lift:.2f}x base)")
        if name=="FULL":
            keep=["player_id","player_display_name","position","team","opponent","season","week",
                  "game_total","implied_team_total","roll_mean","prob","explosion","fantasy_points_ppr"]
            con2=sqlite3.connect(DB); p[keep].to_sql("explosion_v2_predictions",con2,if_exists="replace",index=False); con2.close()
            mfull=lgb.LGBMClassifier(scale_pos_weight=(len(df)-df.explosion.sum())/df.explosion.sum(),**params)
            mfull.fit(df[FULL].astype(float).fillna(-1),df["explosion"])
            imp=pd.DataFrame({"feature":FULL,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
            os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"explosion_v2_importance.csv"),index=False)
            print("  top features:", ", ".join(imp.head(8)["feature"]))
    json.dump(res,open(os.path.join(OUT,"explosion_v2_metrics.json"),"w"),indent=2)
    print("\nExisting model (PROJECT_SUMMARY) reference: AUC 0.679, AP 0.152.")


if __name__ == "__main__":
    main()
