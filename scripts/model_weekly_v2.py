"""
Avenue 2 model: weekly PPR with the EDA-confirmed new signals layered onto a
rolling-stats baseline. Quantifies the incremental lift of:
  - expected fantasy points (xFP, rolling)
  - luck correction (rolling actual - expected)
  - opportunity share (rolling)
  - directional game environment (implied team total, game total x role)
  - QB-aware interactions

Walk-forward by (season, week) over the odds window (2023-2025). Compares a BASE
feature set to the FULL set. Writes weekly_v2_predictions + metrics.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
os.makedirs(OUT, exist_ok=True)
POS = ["QB","RB","WR","TE"]

ABBR = {"Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL",
 "Buffalo Bills":"BUF","Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN",
 "Cleveland Browns":"CLE","Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET",
 "Green Bay Packers":"GB","Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
 "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA",
 "Miami Dolphins":"MIA","Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO",
 "New York Giants":"NYG","New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT",
 "San Francisco 49ers":"SF","Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN",
 "Washington Commanders":"WAS"}


def ewm_prior(df, col, alpha=0.5):
    """Exponentially-weighted mean of PRIOR weeks (shifted to exclude current)."""
    return df.groupby("player_id")[col].transform(lambda s: s.shift(1).ewm(alpha=alpha).mean())


def main():
    con = sqlite3.connect(DB)
    ps = pd.read_sql("""SELECT event_id, player_id, player_display_name, position, team, opponent,
                        season, week, targets, carries, receptions, receiving_yards, rushing_yards,
                        fantasy_points_ppr FROM player_stats WHERE position IN ('QB','RB','WR','TE')""", con)

    # ---- consensus odds per event/team ----
    tot = pd.read_sql("SELECT event_id, AVG(point) game_total FROM game_odds WHERE market='totals' GROUP BY event_id", con)
    spr = pd.read_sql("""SELECT event_id, outcome_name, AVG(point) team_spread
                         FROM game_odds WHERE market='spreads' GROUP BY event_id, outcome_name""", con)
    spr["team"] = spr["outcome_name"].map(ABBR)
    odds = spr.merge(tot, on="event_id", how="inner")
    odds["implied_team_total"] = odds["game_total"]/2 - odds["team_spread"]/2
    odds = odds.dropna(subset=["team"])[["event_id","team","team_spread","game_total","implied_team_total"]]

    # ---- xFP ----
    xfp = pd.read_sql("""SELECT player_id, CAST(season AS INT) season, week,
                         total_fantasy_points_exp xfp FROM nflv_ff_opp""", con)
    # ---- snaps -> offense_pct (join via pfr) ----
    snaps = pd.read_sql("SELECT pfr_player_id, season, week, offense_pct FROM nflv_snaps WHERE game_type='REG'", con)
    pmap = pd.read_sql("SELECT gsis_id player_id, pfr_id, season FROM nflv_rosters WHERE pfr_id IS NOT NULL", con) \
            .drop_duplicates(["pfr_id","season"])
    snaps = snaps.merge(pmap, left_on=["pfr_player_id","season"], right_on=["pfr_id","season"], how="left") \
                 .dropna(subset=["player_id"])[["player_id","season","week","offense_pct"]]
    con.close()

    df = ps.merge(odds, on=["event_id","team"], how="left") \
           .merge(xfp, on=["player_id","season","week"], how="left") \
           .merge(snaps, on=["player_id","season","week"], how="left")
    df["order"] = df["season"]*100 + df["week"]
    df = df.sort_values(["player_id","order"]).reset_index(drop=True)

    df["opportunities"] = df["targets"].fillna(0) + df["carries"].fillna(0)
    df["luck"] = df["fantasy_points_ppr"] - df["xfp"]

    # ---- rolling (prior-only) features ----
    df["roll_ppr"]    = ewm_prior(df, "fantasy_points_ppr")
    df["roll_opp"]    = ewm_prior(df, "opportunities")
    df["roll_targets"]= ewm_prior(df, "targets")
    df["roll_offpct"] = ewm_prior(df, "offense_pct")
    df["roll_xfp"]    = ewm_prior(df, "xfp")
    df["roll_luck"]   = ewm_prior(df, "luck")
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    df["is_qb"] = (df["position"]=="QB").astype(int)
    # interactions
    df["ix_total_x_ppr"]  = df["game_total"] * df["roll_ppr"]
    df["ix_itt_x_offpct"] = df["implied_team_total"] * df["roll_offpct"]
    df["ix_qb_x_total"]   = df["is_qb"] * df["game_total"]

    BASE = ["roll_ppr","roll_opp","roll_targets","roll_offpct","game_total","team_spread","pos_id"]
    FULL = BASE + ["roll_xfp","roll_luck","implied_team_total","ix_total_x_ppr",
                   "ix_itt_x_offpct","ix_qb_x_total","is_qb"]

    df = df[df["roll_ppr"].notna()].copy()  # need history
    params = dict(objective="regression_l1", n_estimators=400, learning_rate=0.04,
                  num_leaves=31, min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
                  random_state=0, verbosity=-1)

    def walk(feats):
        out=[]
        for S in [2023,2024,2025]:
            for W in range(5,23):
                te = df[(df["season"]==S)&(df["week"]==W)]
                if len(te)<10: continue
                tr = df[df["order"] < S*100+W]
                if len(tr)<500: continue
                m=lgb.LGBMRegressor(**params)
                m.fit(tr[feats].astype(float).fillna(-1), tr["fantasy_points_ppr"])
                t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
        return pd.concat(out, ignore_index=True)

    base = walk(BASE); full = walk(FULL)

    def rep(name,d):
        s=d.dropna(subset=["pred","fantasy_points_ppr"])
        return mean_absolute_error(s["fantasy_points_ppr"],s["pred"]), \
               spearmanr(s["pred"],s["fantasy_points_ppr"])[0]
    bm,bc=rep("base",base); fm,fc=rep("full",full)
    print("=== Avenue 2 weekly v2 — walk-forward 2023-25 ===")
    print(f"  BASE  MAE={bm:.3f}  corr={bc:.3f}  (n={len(base):,})")
    print(f"  FULL  MAE={fm:.3f}  corr={fc:.3f}  ({100*(bm-fm)/bm:+.1f}% MAE)")

    print("\n  By position (MAE base -> full):")
    for p in POS:
        b=base[base["position"]==p]; f=full[full["position"]==p]
        bmae=mean_absolute_error(b["fantasy_points_ppr"],b["pred"])
        fmae=mean_absolute_error(f["fantasy_points_ppr"],f["pred"])
        print(f"    {p}: {bmae:.2f} -> {fmae:.2f} ({100*(bmae-fmae)/bmae:+.1f}%)")

    print("\n  By game total bucket (the shootout under-prediction target):")
    for lab,lo,hi in [("<=42",0,42),("42-46",42,46),("46-50",46,50),("50+",50,99)]:
        b=base[(base["game_total"]>lo)&(base["game_total"]<=hi)]
        f=full[(full["game_total"]>lo)&(full["game_total"]<=hi)]
        if len(b)<30: continue
        bb=mean_absolute_error(b["fantasy_points_ppr"],b["pred"])
        ff=mean_absolute_error(f["fantasy_points_ppr"],f["pred"])
        bias_b=(b["fantasy_points_ppr"]-b["pred"]).mean(); bias_f=(f["fantasy_points_ppr"]-f["pred"]).mean()
        print(f"    O/U {lab}: MAE {bb:.2f}->{ff:.2f}  bias {bias_b:+.2f}->{bias_f:+.2f}")

    # persist full predictions + importance
    keep=["player_id","player_display_name","position","team","opponent","season","week",
          "game_total","implied_team_total","roll_ppr","roll_xfp","pred","fantasy_points_ppr"]
    con2=sqlite3.connect(DB); full[keep].rename(columns={"pred":"pred_ppr","fantasy_points_ppr":"actual_ppr"}) \
        .to_sql("weekly_v2_predictions", con2, if_exists="replace", index=False); con2.close()
    mfull=lgb.LGBMRegressor(**params).fit(df[FULL].astype(float).fillna(-1), df["fantasy_points_ppr"])
    imp=pd.DataFrame({"feature":FULL,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    imp.to_csv(os.path.join(OUT,"weekly_v2_importance.csv"), index=False)
    json.dump({"base_mae":bm,"full_mae":fm,"base_corr":bc,"full_corr":fc},
              open(os.path.join(OUT,"weekly_v2_metrics.json"),"w"), indent=2)
    print("\nTop features:"); print(imp.head(12).to_string(index=False))
    print("Saved weekly_v2_predictions + metrics.")


if __name__ == "__main__":
    main()
