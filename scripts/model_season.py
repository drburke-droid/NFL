"""
Avenue 1 model: predict next-season PPG, then derive positional finish + VBD.

Season-blocked walk-forward: for each target season T, train on all seasons < T
(honest out-of-sample). LightGBM regression on the EDA-confirmed feature set.
Finish/VBD derived by projecting games and ranking projected season totals.

Writes table season_predictions; prints evaluation; saves 2026 projections.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
os.makedirs(OUT, exist_ok=True)
POS = ["QB","RB","WR","TE"]
REPL_RANK = {"QB":14,"RB":30,"WR":36,"TE":14}

FEATURES = [
    "prior_ppg","prior2_ppg","prior_games","prior_ppr_std","prior_cv","prior_off_pct",
    "prior_target_share","prior_air_yards_share","prior_wopr","prior_total_tds",
    "prior_rec_td_rate","prior_rush_td_rate","prior_passing_epa","prior_rushing_epa",
    "prior_receiving_epa","prior_targets_pg","prior_receptions_pg","prior_receiving_yards_pg",
    "prior_carries_pg","prior_rushing_yards_pg","prior_attempts_pg","prior_passing_yards_pg",
    "age","years_exp","draft_round","draft_pick","weight","forty","team_change",
]


def project_games(df):
    """Simple, honest games projection: shrink prior games toward position mean."""
    pos_mean = df.groupby("position")["prior_games"].transform("mean")
    pg = 0.55*df["prior_games"].fillna(pos_mean) + 0.45*pos_mean
    return pg.clip(1, 17)


def add_finish_vbd(d, ppg_col, games_col, tag):
    d = d.copy()
    d[f"proj_total_{tag}"] = d[ppg_col]*d[games_col]
    d[f"finish_{tag}"] = d.groupby(["season","position"])[f"proj_total_{tag}"].rank(
        ascending=False, method="min")
    repl=[]
    for (s,p),g in d.groupby(["season","position"]):
        gg=g.sort_values(f"proj_total_{tag}",ascending=False).reset_index(drop=True)
        idx=min(REPL_RANK[p]-1,len(gg)-1)
        repl.append({"season":s,"position":p,f"repl_{tag}":gg.loc[idx,ppg_col]})
    d=d.merge(pd.DataFrame(repl),on=["season","position"],how="left")
    d[f"vbd_{tag}"]=(d[ppg_col]-d[f"repl_{tag}"])*d[games_col]
    return d


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con).copy()
    df = df[df["next_ppg"].notna()]
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    df["proj_games"] = project_games(df)

    params = dict(objective="regression_l1", n_estimators=500, learning_rate=0.03,
                  num_leaves=31, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                  random_state=0, verbosity=-1)

    preds=[]
    for T in range(2016, 2026):
        tr = df[df["season"] < T]; te = df[df["season"] == T]
        if len(te)==0: continue
        Xtr = tr[FEATURES+["pos_id"]].astype(float).fillna(-1)
        Xte = te[FEATURES+["pos_id"]].astype(float).fillna(-1)
        m = lgb.LGBMRegressor(**params); m.fit(Xtr, tr["next_ppg"])
        p = te.copy(); p["pred_ppg"] = m.predict(Xte)
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True)

    # baseline = repeat prior ppg
    pred["base_ppg"] = pred["prior_ppg"].fillna(pred["prior_ppg"].median())

    # derive finish/VBD for model, baseline, and actual
    pred = add_finish_vbd(pred, "pred_ppg", "proj_games", "pred")
    pred = add_finish_vbd(pred, "base_ppg", "proj_games", "base")
    pred = add_finish_vbd(pred, "next_ppg", "next_games", "act")

    # ---------- evaluation ----------
    print("=== Avenue 1 season model — out-of-sample (2016-2025) ===")
    print(f"rows: {len(pred):,}")
    def ppg_eval(d,col):
        s=d[[col,"next_ppg"]].dropna()
        return mean_absolute_error(s["next_ppg"],s[col]), r2_score(s["next_ppg"],s[col]), \
               spearmanr(s[col],s["next_ppg"])[0]
    for label,col in [("model","pred_ppg"),("baseline(repeat)","base_ppg")]:
        mae,r2,rho=ppg_eval(pred,col)
        print(f"  PPG {label:18s} MAE={mae:.3f} R2={r2:.3f} rho={rho:.3f}")
    print("\n  PPG MAE by position (model vs baseline):")
    for p in POS:
        d=pred[pred["position"]==p]
        mm=mean_absolute_error(d["next_ppg"],d["pred_ppg"]); bb=mean_absolute_error(d["next_ppg"],d["base_ppg"])
        print(f"    {p}: model {mm:.2f}  baseline {bb:.2f}  ({100*(bb-mm)/bb:+.1f}%)")

    # finish quality: rank correlation of projected finish vs actual finish, by position
    print("\n  Finish rank correlation (proj finish vs actual finish; lower rank=better):")
    for p in POS:
        d=pred[pred["position"]==p].dropna(subset=["finish_pred","finish_act"])
        rm=spearmanr(d["finish_pred"],d["finish_act"])[0]
        rb=spearmanr(d["finish_base"],d["finish_act"])[0]
        print(f"    {p}: model {rm:.3f}  baseline {rb:.3f}")

    # top-N hit rate (did we identify the actual top tier?)
    print("\n  Top-N identification hit rate (model | baseline):")
    for p,N in [("QB",12),("RB",24),("WR",36),("TE",12)]:
        d=pred[pred["position"]==p]
        hits_m=hits_b=tot=0
        for s,g in d.groupby("season"):
            act_top=set(g.nsmallest(N,"finish_act")["player_id"])
            if not act_top: continue
            hits_m+=len(set(g.nsmallest(N,"finish_pred")["player_id"])&act_top)
            hits_b+=len(set(g.nsmallest(N,"finish_base")["player_id"])&act_top)
            tot+=len(act_top)
        print(f"    {p} top-{N}: model {hits_m/tot:.1%}  baseline {hits_b/tot:.1%}")

    # ---------- persist ----------
    names = pd.read_sql("SELECT player_id, player_display_name FROM nflv_season",
                        con).drop_duplicates("player_id")
    pred = pred.merge(names, on="player_id", how="left")
    keep=["player_id","player_display_name","position","season","team","prior_season",
          "prior_ppg","pred_ppg","next_ppg","proj_games","next_games",
          "proj_total_pred","finish_pred","vbd_pred","finish_act","vbd_act"]
    pred[keep].to_sql("season_predictions", con, if_exists="replace", index=False)
    json.dump({"rows":int(len(pred))}, open(os.path.join(OUT,"season_model_meta.json"),"w"))

    # feature importance from a full-data fit (for the report)
    full = df[df["season"]<=2025]
    mf = lgb.LGBMRegressor(**params).fit(full[FEATURES+["pos_id"]].astype(float).fillna(-1),
                                         full["next_ppg"])
    imp = pd.DataFrame({"feature":FEATURES+["pos_id"],"importance":mf.feature_importances_}) \
            .sort_values("importance",ascending=False)
    imp.to_csv(os.path.join(OUT,"season_model_importance.csv"), index=False)
    print("\nTop 12 features:"); print(imp.head(12).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
