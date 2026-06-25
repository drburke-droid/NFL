"""
Avenue 1 model: predict next-season PPG, then derive positional finish + VBD.

Season-blocked walk-forward: for each target season T, train on all seasons < T
(honest out-of-sample). LightGBM regression. The production feature set is
FFA-ANCHORED: prior-year production/role/draft signals fused with the
FantasyFootballAnalytics weighted-expert consensus (nflv_ffa_proj), which the
value test showed is the best configuration. FFA features are NaN (-> -1) for
players without a consensus projection, so all rows are still scored.
Finish/VBD derived by projecting games and ranking projected season totals.

Writes table season_predictions; prints evaluation.
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

# FantasyFootballAnalytics consensus features (the season market anchor)
FFA_FEATURES = [
    "ffa_points","ffa_vor","ffa_floor","ffa_ceiling","ffa_sd","ffa_uncertainty",
    "ffa_adp","ffa_pos_rank","ffa_rank","ffa_tier","ffa_dropoff",
]
MODEL_FEATURES = FEATURES + FFA_FEATURES + ["pos_id"]

# Injury / games-aware prior — a player coming off an injury-shortened season is
# under-rated when anchored on prior_ppg alone. These down-weight a short recent
# year toward the healthy baseline. Validated (test_injury_prior.py): improves
# overall MAE (QB -0.08) and the injury-return subgroup without hurting the pool.
INJURY_FEATURES = ["prior2_games", "gw_prior", "healthy_prior", "short_season", "bounce", "games_trend"]


def add_injury_features(df):
    """Add injury/games-aware prior features. Needs prior_ppg, prior2_ppg,
    prior_games, prior2_games (prior2_games filled with NaN if absent)."""
    d = df.copy()
    if "prior2_games" not in d.columns: d["prior2_games"] = np.nan
    pg = d["prior_games"].fillna(0); p2g = d["prior2_games"].fillna(0)
    pp = d["prior_ppg"]; p2p = d["prior2_ppg"] if "prior2_ppg" in d.columns else pd.Series(np.nan, index=d.index)
    d["gw_prior"] = ((pp.fillna(0)*pg + p2p.fillna(0)*p2g) / (pg+p2g).replace(0, np.nan)).fillna(pp)
    d["healthy_prior"] = pd.concat([pp, p2p], axis=1).max(axis=1)
    d["short_season"] = (pg <= 11).astype(int)
    d["bounce"] = ((pg <= 11) & (p2g >= 14) & ((p2p - pp) >= 3)).astype(int)
    d["games_trend"] = pg - p2g
    return d


def attach_ffa(df, con):
    """Left-join FFA consensus projections onto a (player_id, season) frame."""
    ffa = pd.read_sql("SELECT * FROM nflv_ffa_proj WHERE player_id IS NOT NULL", con)
    ffa = ffa.sort_values("ffa_points", ascending=False).drop_duplicates(["season","player_id"])
    ffa = ffa.drop(columns=[c for c in ["position","player"] if c in ffa.columns])
    return df.merge(ffa, on=["season","player_id"], how="left")


def project_games(df):
    """Games projection, shrunk toward the position mean. Games-played has low
    year-over-year persistence (r~0.42), so when a SECOND prior year is available
    we blend it in — this keeps a single injury-shortened season from tanking the
    projection (validated: lower next-games MAE overall AND for injury-returns)."""
    pos_mean = df.groupby("position")["prior_games"].transform("mean")
    if "prior2_games" in df.columns:
        p2 = df["prior2_games"].fillna(df["prior_games"]).fillna(pos_mean)
        pg = 0.35*df["prior_games"].fillna(pos_mean) + 0.30*p2 + 0.35*pos_mean
    else:
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
    df = attach_ffa(df, con)
    df["has_ffa"] = df["ffa_points"].notna().astype(int)
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    df["proj_games"] = project_games(df)

    params = dict(objective="regression_l1", n_estimators=500, learning_rate=0.03,
                  num_leaves=31, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                  random_state=0, verbosity=-1)

    preds=[]
    for T in range(2016, 2026):
        tr = df[df["season"] < T]; te = df[df["season"] == T]
        if len(te)==0: continue
        Xtr = tr[MODEL_FEATURES].astype(float).fillna(-1)
        Xte = te[MODEL_FEATURES].astype(float).fillna(-1)
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
    for label,col in [("model(FFA-anchored)","pred_ppg"),("baseline(repeat)","base_ppg")]:
        mae,r2,rho=ppg_eval(pred,col)
        print(f"  PPG {label:20s} MAE={mae:.3f} R2={r2:.3f} rho={rho:.3f}")
    cov = pred["has_ffa"].mean()
    fc = pred[pred["has_ffa"]==1]; nc = pred[pred["has_ffa"]==0]
    print(f"  FFA coverage: {cov:.0%} of scored rows | model MAE  with-FFA "
          f"{mean_absolute_error(fc['next_ppg'],fc['pred_ppg']):.2f}  "
          f"no-FFA {mean_absolute_error(nc['next_ppg'],nc['pred_ppg']):.2f}")
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
          "prior_ppg","ffa_points","has_ffa","pred_ppg","next_ppg","proj_games","next_games",
          "proj_total_pred","finish_pred","vbd_pred","finish_act","vbd_act"]
    pred[keep].to_sql("season_predictions", con, if_exists="replace", index=False)
    json.dump({"rows":int(len(pred)),"ffa_coverage":float(pred["has_ffa"].mean())},
              open(os.path.join(OUT,"season_model_meta.json"),"w"))

    # feature importance from a full-data fit (for the report)
    full = df[df["season"]<=2025]
    mf = lgb.LGBMRegressor(**params).fit(full[MODEL_FEATURES].astype(float).fillna(-1),
                                         full["next_ppg"])
    imp = pd.DataFrame({"feature":MODEL_FEATURES,"importance":mf.feature_importances_}) \
            .sort_values("importance",ascending=False)
    imp.to_csv(os.path.join(OUT,"season_model_importance.csv"), index=False)
    print("\nTop 12 features:"); print(imp.head(12).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
