"""
Rookie season model — the anchored approach extended to players with no prior
NFL season. Inputs are pre-NFL signals only: draft capital, combine athleticism,
age, and LANDING SPOT (prior-year team offense + position-group environment).
Where rookie ADP exists (2021-25), we test ADP-alone vs model vs ADP-anchored.

Target: rookie-year PPG. Walk-forward by season.
"""
import os, sqlite3
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
POS = ["QB","RB","WR","TE"]
PARAMS = dict(objective="regression_l1", n_estimators=350, learning_rate=0.03, num_leaves=20,
              min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

BASE = ["draft_pick","draft_round","age","forty","vertical","broad_jump","cone","shuttle",
        "bench","height","weight","team_off_ppg_prior","team_pos_ppg_prior",
        "team_pass_att_prior","team_rush_att_prior","pos_id"]
MARKET = ["ecr","pos_rank","pos_id"]
ANCHOR = BASE + ["ecr","pos_rank"]


def build(con):
    seas = pd.read_sql("""SELECT player_id,player_display_name,position,recent_team team,season,
                          games,attempts,carries,fantasy_points_ppr FROM nflv_season
                          WHERE position IN ('QB','RB','WR','TE')""", con)
    ros = pd.read_sql("SELECT gsis_id player_id,season,years_exp,birth_date,height,weight,draft_number FROM nflv_rosters", con)
    ros = ros.sort_values("years_exp").drop_duplicates(["player_id","season"])
    draft = pd.read_sql("SELECT gsis_id player_id,round draft_round,pick draft_pick FROM nflv_draft", con).dropna(subset=["player_id"]).drop_duplicates("player_id")
    comb = pd.read_sql("SELECT pfr_id,forty,vertical,broad_jump,cone,shuttle,bench FROM nflv_combine", con).drop_duplicates("pfr_id")
    pfrmap = pd.read_sql("SELECT gsis_id player_id,pfr_id FROM nflv_rosters WHERE pfr_id IS NOT NULL", con).drop_duplicates("player_id")
    adp = pd.read_sql("SELECT season,player_id,ecr,pos_rank FROM nflv_adp WHERE player_id IS NOT NULL", con)

    # landing spot: prior-year team offense + position environment
    team_all = seas.groupby(["team","season"]).agg(
        team_off_ppg=("fantasy_points_ppr","sum"),
        team_pass_att=("attempts","sum"), team_rush_att=("carries","sum")).reset_index()
    team_pos = seas.groupby(["team","season","position"]).agg(
        team_pos_ppg=("fantasy_points_ppr","sum")).reset_index()
    team_all["season"] += 1; team_pos["season"] += 1   # shift to act as PRIOR year
    for c in ["team_off_ppg","team_pass_att","team_rush_att"]:
        team_all[c] = team_all[c]/17.0
    team_pos["team_pos_ppg"] = team_pos["team_pos_ppg"]/17.0

    # rookies = first NFL year
    rook = ros[ros["years_exp"] == 0][["player_id","season","birth_date"]]
    df = seas.merge(rook, on=["player_id","season"], how="inner")
    df["ppg"] = df["fantasy_points_ppr"]/df["games"].clip(lower=1)
    df["age"] = df["season"] - pd.to_datetime(df["birth_date"], errors="coerce").dt.year

    df = df.merge(draft, on="player_id", how="left")
    df["draft_pick"] = df["draft_pick"].fillna(262); df["draft_round"] = df["draft_round"].fillna(8)
    df = df.merge(pfrmap, on="player_id", how="left").merge(comb, on="pfr_id", how="left")
    ros_sz = ros[["player_id","season","height","weight"]]
    df = df.merge(ros_sz, on=["player_id","season"], how="left")
    df = df.merge(team_all, on=["team","season"], how="left").rename(columns={
        "team_off_ppg":"team_off_ppg_prior","team_pass_att":"team_pass_att_prior",
        "team_rush_att":"team_rush_att_prior"})
    df = df.merge(team_pos, on=["team","season","position"], how="left").rename(columns={"team_pos_ppg":"team_pos_ppg_prior"})
    df = df.merge(adp, on=["season","player_id"], how="left")
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    return df


def walk(df, feats, test_seasons):
    out=[]
    for T in test_seasons:
        tr=df[df.season<T]; te=df[df.season==T]
        if len(te)==0 or len(tr)<150: continue
        m=lgb.LGBMRegressor(**PARAMS).fit(tr[feats].astype(float).fillna(-1), tr["ppg"])
        t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ev(p):
    s=p.dropna(subset=["pred","ppg"])
    return mean_absolute_error(s["ppg"],s["pred"]), spearmanr(s["pred"],s["ppg"])[0], len(s)


def main():
    con=sqlite3.connect(DB); df=build(con)
    print(f"Rookie skill seasons: {len(df):,} (2011-2025)")
    print("by position:", df["position"].value_counts().to_dict())

    # ---- full model vs baselines (all rookies, test 2015-2025) ----
    test=list(range(2015,2026))
    full=walk(df, BASE, test)
    pick=walk(df, ["draft_pick","draft_round","pos_id"], test)
    # position-mean baseline (prior seasons mean by pos)
    posmean=[]
    for T in test:
        tr=df[df.season<T]; te=df[df.season==T].copy()
        mp=tr.groupby("position")["ppg"].mean()
        te["pred"]=te["position"].map(mp); posmean.append(te)
    posmean=pd.concat(posmean, ignore_index=True)

    print("\n=== Full rookie model (test 2015-2025) ===")
    for name,p in [("position-mean",posmean),("draft-pick only",pick),("full model",full)]:
        mae,rho,n=ev(p); print(f"  {name:16s} MAE={mae:.3f} rho={rho:.3f} (n={n})")
    print("\n  Full-model MAE by position:")
    for pos in POS:
        d=full[full.position==pos];
        if len(d): print(f"    {pos}: MAE {mean_absolute_error(d['ppg'],d['pred']):.2f}  rho {spearmanr(d['pred'],d['ppg'])[0]:.3f} (n={len(d)})")

    # ---- ADP subset: market vs model vs anchored (2021-25) ----
    sub=df[df["ecr"].notna()].copy()
    print(f"\n=== Rookie ADP subset: {len(sub):,} (2021-2025) ===")
    tt=[2023,2024,2025]
    res={}
    for name,feats in [("market(ADP)",MARKET),("model(no ADP)",BASE),("anchored",ANCHOR)]:
        p=walk(sub, feats, tt); res[name]=ev(p)
    print(f"{'model':16s} {'MAE':>7s} {'rho':>7s} {'n':>5s}")
    for name in ["market(ADP)","model(no ADP)","anchored"]:
        mae,rho,n=res[name]; print(f"{name:16s} {mae:7.3f} {rho:7.3f} {n:5d}")

    # importance of full model
    mfull=lgb.LGBMRegressor(**PARAMS).fit(df[BASE].astype(float).fillna(-1), df["ppg"])
    imp=pd.DataFrame({"feature":BASE,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"rookie_importance.csv"),index=False)
    print("\n  Top features:"); print(imp.head(10).to_string(index=False))

    # save full predictions
    save = full[["player_id","player_display_name","position","team","season","draft_pick","pred","ppg"]] \
            .rename(columns={"pred":"pred_ppg","ppg":"actual_ppg"})
    save.to_sql("rookie_predictions", con, if_exists="replace", index=False)
    con.close()
    print("\nSaved rookie_predictions + importance.")
    print("\nBest rookie seasons the model ranked highest (test years):")
    print(full.nlargest(12,"pred")[["season","player_display_name","position","team","draft_pick","pred","ppg"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
