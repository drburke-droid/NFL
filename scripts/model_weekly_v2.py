"""
Avenue 2 weekly PPR model — now on the FULL 2012-2025 window.

Uses nflv_weekly (2011-25) + nflv_game_lines (historical spread/total) + xFP +
snaps, instead of the 3-season Odds-API tables. Compares a rolling-stats BASE to
a FULL set (xFP, luck, directional implied team total, QB interactions). Reports
the full window and the 2023-25 subset (to see if more history helped recent years).

Writes weekly_v2_predictions + metrics.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
POS = ["QB","RB","WR","TE"]
START_TEST = 2014   # need 2012-2013 as initial history


def ewm_prior(df, col, alpha=0.5):
    return df.groupby("player_id")[col].transform(lambda s: s.shift(1).ewm(alpha=alpha).mean())


def main():
    con = sqlite3.connect(DB)
    wk = pd.read_sql("""SELECT player_id,player_display_name,position,team,opponent_team opponent,
                        season,week,targets,carries,fantasy_points_ppr
                        FROM nflv_weekly WHERE season_type='REG' AND season>=2012
                        AND position IN ('QB','RB','WR','TE')""", con)
    gl = pd.read_sql("""SELECT season,week,team,team_spread,game_total,implied_team_total,is_home
                        FROM nflv_game_lines WHERE game_type='REG'""", con)
    xfp = pd.read_sql("SELECT player_id,CAST(season AS INT) season,week,total_fantasy_points_exp xfp FROM nflv_ff_opp", con)
    snaps = pd.read_sql("SELECT pfr_player_id,season,week,offense_pct FROM nflv_snaps WHERE game_type='REG'", con)
    pmap = pd.read_sql("SELECT gsis_id player_id,pfr_id,season FROM nflv_rosters WHERE pfr_id IS NOT NULL", con) \
            .drop_duplicates(["pfr_id","season"])
    con.close()
    snaps = snaps.merge(pmap, left_on=["pfr_player_id","season"], right_on=["pfr_id","season"], how="left") \
                 .dropna(subset=["player_id"])[["player_id","season","week","offense_pct"]]

    df = wk.merge(gl, on=["season","week","team"], how="left") \
           .merge(xfp, on=["player_id","season","week"], how="left") \
           .merge(snaps, on=["player_id","season","week"], how="left")
    df["order"] = df["season"]*100 + df["week"]
    df = df.sort_values(["player_id","order"]).reset_index(drop=True)
    df["opportunities"] = df["targets"].fillna(0) + df["carries"].fillna(0)
    df["luck"] = df["fantasy_points_ppr"] - df["xfp"]

    df["roll_ppr"]     = ewm_prior(df, "fantasy_points_ppr")
    df["roll_opp"]     = ewm_prior(df, "opportunities")
    df["roll_targets"] = ewm_prior(df, "targets")
    df["roll_offpct"]  = ewm_prior(df, "offense_pct")
    df["roll_xfp"]     = ewm_prior(df, "xfp")
    df["roll_luck"]    = ewm_prior(df, "luck")
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    df["is_qb"]  = (df["position"]=="QB").astype(int)
    df["ix_total_x_ppr"]  = df["game_total"] * df["roll_ppr"]
    df["ix_itt_x_offpct"] = df["implied_team_total"] * df["roll_offpct"]
    df["ix_qb_x_total"]   = df["is_qb"] * df["game_total"]

    BASE = ["roll_ppr","roll_opp","roll_targets","roll_offpct","game_total","team_spread","pos_id"]
    FULL = BASE + ["roll_xfp","roll_luck","implied_team_total","ix_total_x_ppr",
                   "ix_itt_x_offpct","ix_qb_x_total","is_qb"]

    df = df[df["roll_ppr"].notna()].copy()
    print(f"rows: {len(df):,}  seasons {int(df.season.min())}-{int(df.season.max())}")
    params = dict(objective="regression_l1", n_estimators=400, learning_rate=0.04, num_leaves=31,
                  min_child_samples=60, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

    def walk(feats):
        out=[]
        for S in range(START_TEST, 2026):
            for W in range(2,23):
                te=df[(df.season==S)&(df.week==W)]
                if len(te)<10: continue
                tr=df[df.order < S*100+W]
                if len(tr)<2000: continue
                m=lgb.LGBMRegressor(**params); m.fit(tr[feats].astype(float).fillna(-1), tr["fantasy_points_ppr"])
                t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
        return pd.concat(out, ignore_index=True)

    base=walk(BASE); full=walk(FULL)
    def rep(d):
        s=d.dropna(subset=["pred","fantasy_points_ppr"])
        return mean_absolute_error(s["fantasy_points_ppr"],s["pred"]), spearmanr(s["pred"],s["fantasy_points_ppr"])[0]
    bm,bc=rep(base); fm,fc=rep(full)
    print("\n=== Weekly PPR (walk-forward, FULL window 2014-2025) ===")
    print(f"  BASE  MAE={bm:.3f} corr={bc:.3f} (n={len(base):,})")
    print(f"  FULL  MAE={fm:.3f} corr={fc:.3f} ({100*(bm-fm)/bm:+.1f}% MAE)")
    # 2023-25 subset for comparison to prior 3-season run
    b23=base[base.season>=2023]; f23=full[full.season>=2023]
    bm2,_=rep(b23); fm2,_=rep(f23)
    print(f"  [2023-25 subset] BASE MAE={bm2:.3f}  FULL MAE={fm2:.3f}  ({100*(bm2-fm2)/bm2:+.1f}%)")
    print("\n  By position (FULL window, MAE base->full):")
    for p in POS:
        b=base[base.position==p]; f=full[full.position==p]
        print(f"    {p}: {mean_absolute_error(b['fantasy_points_ppr'],b['pred']):.2f} -> {mean_absolute_error(f['fantasy_points_ppr'],f['pred']):.2f}")

    keep=["player_id","player_display_name","position","team","opponent","season","week",
          "game_total","implied_team_total","roll_ppr","roll_xfp","pred","fantasy_points_ppr"]
    con2=sqlite3.connect(DB); full[keep].rename(columns={"pred":"pred_ppr","fantasy_points_ppr":"actual_ppr"}) \
        .to_sql("weekly_v2_predictions", con2, if_exists="replace", index=False); con2.close()
    mfull=lgb.LGBMRegressor(**params).fit(df[FULL].astype(float).fillna(-1), df["fantasy_points_ppr"])
    imp=pd.DataFrame({"feature":FULL,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    os.makedirs(OUT,exist_ok=True); imp.to_csv(os.path.join(OUT,"weekly_v2_importance.csv"),index=False)
    json.dump({"window":"2014-2025","base_mae":bm,"full_mae":fm,"base_corr":bc,"full_corr":fc,
               "n":len(full),"sub2325_base":bm2,"sub2325_full":fm2},
              open(os.path.join(OUT,"weekly_v2_metrics.json"),"w"), indent=2)
    print("\nTop features:"); print(imp.head(10).to_string(index=False))
    print("Saved weekly_v2_predictions + metrics.")


if __name__ == "__main__":
    main()
