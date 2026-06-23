"""
Does adding the now-full-window ARCHETYPE + GAME-SCRIPT features help the
explosion model beyond the rolling+xFP+game-environment baseline?

Compares NOARCH (the explosion_v2 FULL feature set) vs ARCH (+ team off/def
archetype tiers + predicted game-script probabilities) on 2012-2025.
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
OFF_TIER = {"Elite":4,"Pass-First":3,"Pass-Heavy":3,"Run-Heavy":2,"Bottom-Tier":1}
DEF_TIER = {"Elite":4,"Above-Average":3,"Middle-of-Pack":2,"Bottom-Tier":1}


def ou_bucket(v):
    if pd.isna(v): return None
    if v<=38: return "Low (≤38)"
    if v<=42: return "Med-Low (38-42)"
    if v<=46: return "Medium (42-46)"
    if v<=50: return "Med-High (46-50)"
    return "High (50+)"
def sp_bucket(v):
    if pd.isna(v): return None
    v=abs(v)
    if v<=2.5: return "Pickem (0-2.5)"
    if v<=5: return "Close (3-5)"
    if v<=8: return "Moderate (5.5-8)"
    if v<=15: return "Big (8.5-15)"
    return "Huge (15+)"

HIGH={"Shootout","High-Scoring Pulled Away"}
BLOW={"Wire-to-Wire Blowout","2nd Half Blowout"}
LOWS={"Defensive Slugfest","Low-Scoring Seesaw"}


def main():
    con=sqlite3.connect(DB)
    wk=pd.read_sql("""SELECT player_id,player_display_name,position,team,opponent_team opponent,
                      season,week,targets,carries,fantasy_points_ppr FROM nflv_weekly
                      WHERE season_type='REG' AND season>=2012 AND position IN ('QB','RB','WR','TE')""",con)
    gl=pd.read_sql("SELECT season,week,team,team_spread,game_total,implied_team_total FROM nflv_game_lines WHERE game_type='REG'",con)
    xfp=pd.read_sql("SELECT player_id,CAST(season AS INT) season,week,total_fantasy_points_exp xfp FROM nflv_ff_opp",con)
    ta=pd.read_sql("SELECT team,season,unit,archetype FROM team_archetypes",con)
    fac=pd.read_sql("SELECT factor_type,factor_value,game_script,probability FROM script_prediction_factors",con)
    con.close()

    df=wk.merge(gl,on=["season","week","team"],how="left").merge(xfp,on=["player_id","season","week"],how="left")
    df["order"]=df["season"]*100+df["week"]; df=df.sort_values(["player_id","order"]).reset_index(drop=True)
    df["opp_ct"]=df["targets"].fillna(0)+df["carries"].fillna(0)
    g=df.groupby("player_id")
    df["roll_mean"]=g["fantasy_points_ppr"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_std"]=g["fantasy_points_ppr"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std())
    df["roll_opp"]=g["opp_ct"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_xfp"]=g["xfp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["roll_xfp_std"]=g["xfp"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std())
    df["luck"]=df["fantasy_points_ppr"]-df["xfp"]
    df["roll_luck"]=g["luck"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    df["cv"]=df["roll_std"]/df["roll_mean"].replace(0,np.nan)
    df["pos_id"]=df["position"].map({p:i for i,p in enumerate(POS)})
    df["is_qb"]=(df["position"]=="QB").astype(int)
    df["ix_total_x_std"]=df["game_total"]*df["roll_std"]
    df["ix_itt_x_opp"]=df["implied_team_total"]*df["roll_opp"]

    # --- archetype tiers ---
    off=ta[ta.unit=="offense"][["team","season","archetype"]].rename(columns={"archetype":"own_off_arch"})
    deff=ta[ta.unit=="defense"][["team","season","archetype"]].rename(columns={"archetype":"opp_def_arch"})
    df=df.merge(off,on=["team","season"],how="left").merge(deff,left_on=["opponent","season"],right_on=["team","season"],how="left",suffixes=("","_d"))
    df["own_off_tier"]=df["own_off_arch"].map(OFF_TIER).fillna(2)
    df["opp_def_tier"]=df["opp_def_arch"].map(DEF_TIER).fillna(2)

    # --- game-script probabilities (blend O/U + spread lookups) ---
    ou={}; sp={}
    for _,r in fac.iterrows():
        (ou if r.factor_type=="over_under" else sp)[(r.factor_value,r.game_script)]=r.probability
    df["ou_bkt"]=df["game_total"].apply(ou_bucket); df["sp_bkt"]=df["team_spread"].apply(sp_bucket)
    def gprob(row,scripts):
        tot=0.0
        for sc in scripts:
            a=ou.get((row["ou_bkt"],sc)); b=sp.get((row["sp_bkt"],sc))
            vals=[v for v in (a,b) if v is not None]
            tot+= (sum(vals)/len(vals)) if vals else 0.125
        return tot
    df["gs_high"]=df.apply(lambda r:gprob(r,HIGH),axis=1)
    df["gs_blowout"]=df.apply(lambda r:gprob(r,BLOW),axis=1)
    df["gs_lowscore"]=df.apply(lambda r:gprob(r,LOWS),axis=1)
    df["ix_gshigh_x_ppr"]=df["gs_high"]*df["roll_mean"]

    df=df[df["roll_mean"]>=3].dropna(subset=["roll_std"]).copy()
    df["explosion"]=(df["fantasy_points_ppr"]>=df["roll_mean"]+2*df["roll_std"]).astype(int)
    print(f"rows: {len(df):,}  base rate {df['explosion'].mean():.3f}")

    NOARCH=["roll_mean","roll_std","cv","roll_opp","pos_id","roll_xfp","roll_xfp_std","roll_luck",
            "game_total","implied_team_total","team_spread","ix_total_x_std","ix_itt_x_opp","is_qb"]
    ARCH=NOARCH+["own_off_tier","opp_def_tier","gs_high","gs_blowout","gs_lowscore","ix_gshigh_x_ppr"]
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
        return roc_auc_score(y,s),average_precision_score(y,s),p.nlargest(k,"prob")["explosion"].mean()/y.mean()

    res={}
    for name,feats in [("NOARCH",NOARCH),("ARCH",ARCH)]:
        p=walk(feats); auc,ap,lift=metrics(p); res[name]={"auc":round(auc,4),"ap":round(ap,4),"lift@10%":round(lift,2),"n":len(p)}
        print(f"  {name}: AUC={auc:.4f} AP={ap:.4f} lift@top10%={lift:.2f}x")
    mfull=lgb.LGBMClassifier(scale_pos_weight=(len(df)-df.explosion.sum())/df.explosion.sum(),**params)
    mfull.fit(df[ARCH].astype(float).fillna(-1),df["explosion"])
    imp=pd.DataFrame({"feature":ARCH,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"explosion_arch_importance.csv"),index=False)
    json.dump(res,open(os.path.join(OUT,"explosion_arch_metrics.json"),"w"),indent=2)
    print("  arch feature ranks:", {f:int(imp[imp.feature==f].index[0]) for f in ["own_off_tier","opp_def_tier","gs_high","gs_blowout","ix_gshigh_x_ppr"]})
    print("  top 10:", ", ".join(imp.head(10)["feature"]))


if __name__ == "__main__":
    main()
