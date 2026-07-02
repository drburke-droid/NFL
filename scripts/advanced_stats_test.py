"""Do UNUSED advanced stats improve the actual season model? Blocks tested over MODEL_FEATURES
walk-forward (2016-2025): A) conversion ratios (cpoe/racr/pacr/aDOT)  B) xFP opportunity+gap
C) team-QB context for pass-catchers  D) NGS tracking (separation/cushion/TT/box)  E) OL sack rate."""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_s=importlib.util.spec_from_file_location("ms",os.path.join(ROOT,"scripts","model_season.py"))
MS=importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)
REL="https://github.com/nflverse/nflverse-data/releases/download/"
con=sqlite3.connect(MS.DB)
df=pd.read_sql("SELECT * FROM season_dataset",con); df=df[df.next_ppg.notna()]
df=MS.attach_ffa(df,con)
# A) conversion ratios from nflv_season (prior year -> +1)
adv=pd.read_sql("""SELECT player_id,season,passing_cpoe cpoe,pacr,racr,
    receiving_air_yards,targets,attempts FROM nflv_season""",con).drop_duplicates(["player_id","season"])
adv["adot"]=adv.receiving_air_yards/adv.targets.replace(0,np.nan)
adv["season"]+=1
# B) xFP
xf=pd.read_sql("""SELECT player_id,season,SUM(total_fantasy_points_exp) e,SUM(total_fantasy_points_diff) g,
    COUNT(DISTINCT week) wk FROM nflv_ff_opp GROUP BY player_id,season""",con)
xf["xfp_pg"]=xf.e/xf.wk; xf["xgap_pg"]=xf.g/xf.wk; xf["season"]=xf.season.astype(int)+1
# C) team QB context (prior yr team QB quality)
qb=pd.read_sql("""SELECT season,recent_team tm,SUM(passing_epa)/SUM(attempts) qb_epa,
    AVG(passing_cpoe) qb_cpoe FROM nflv_season WHERE position='QB' AND attempts>=100
    GROUP BY season,recent_team""",con)
qb["season"]+=1
# E) OL sack rate (team, prior yr)
sk=pd.read_sql("""SELECT season,team tm,SUM(sacks_suffered) s,COUNT(*) n FROM nflv_weekly
    WHERE position='QB' GROUP BY season,team""",con)
sk["sk_pg"]=sk.s/sk.n; sk["season"]=sk.season+1
con.close()
# D) NGS season aggregates (week 0), prior yr -> +1
ngr=pd.read_parquet(REL+"nextgen_stats/ngs_receiving.parquet")
ngr=ngr[ngr.week==0][["season","player_gsis_id","avg_separation","avg_cushion","catch_percentage",
    "percent_share_of_intended_air_yards","avg_yac_above_expectation"]].rename(columns={"player_gsis_id":"player_id"})
ngp=pd.read_parquet(REL+"nextgen_stats/ngs_passing.parquet")
ngp=ngp[ngp.week==0][["season","player_gsis_id","avg_time_to_throw","aggressiveness",
    "avg_air_yards_to_sticks","completion_percentage_above_expectation"]].rename(columns={"player_gsis_id":"player_id"})
ngu=pd.read_parquet(REL+"nextgen_stats/ngs_rushing.parquet")
ngu=ngu[ngu.week==0][["season","player_gsis_id","efficiency","percent_attempts_gte_eight_defenders",
    "rush_yards_over_expected_per_att"]].rename(columns={"player_gsis_id":"player_id"})
for t in (ngr,ngp,ngu): t["season"]+=1
df=df.merge(adv[["player_id","season","cpoe","pacr","racr","adot"]],on=["player_id","season"],how="left") \
     .merge(xf[["player_id","season","xfp_pg","xgap_pg"]],on=["player_id","season"],how="left") \
     .merge(qb,left_on=["season","prior_team"],right_on=["season","tm"],how="left") \
     .merge(sk[["season","tm","sk_pg"]].rename(columns={"tm":"tm2"}),left_on=["season","prior_team"],right_on=["season","tm2"],how="left") \
     .merge(ngr,on=["player_id","season"],how="left").merge(ngp,on=["player_id","season"],how="left") \
     .merge(ngu,on=["player_id","season"],how="left")
df["pos_id"]=df.position.map({p:i for i,p in enumerate(MS.POS)})
BASE=MS.MODEL_FEATURES
BL={"A conv ratios (cpoe/racr/pacr/adot)":["cpoe","pacr","racr","adot"],
    "B xFP opp+gap":["xfp_pg","xgap_pg"],
    "C team-QB context":["qb_epa","qb_cpoe"],
    "D NGS tracking":["avg_separation","avg_cushion","catch_percentage","percent_share_of_intended_air_yards",
                      "avg_yac_above_expectation","avg_time_to_throw","aggressiveness","avg_air_yards_to_sticks",
                      "completion_percentage_above_expectation","efficiency","percent_attempts_gte_eight_defenders",
                      "rush_yards_over_expected_per_att"],
    "E OL sack rate":["sk_pg"]}
params=dict(objective="regression_l1",n_estimators=500,learning_rate=0.03,num_leaves=31,
            min_child_samples=40,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
def walk(feats):
    errs=[]
    for T in range(2016,2026):
        tr,te=df[df.season<T],df[df.season==T]
        if not len(te): continue
        m=lgb.LGBMRegressor(**params).fit(tr[feats].astype(float).fillna(-1),tr.next_ppg)
        errs+=list(np.abs(te.next_ppg.values-m.predict(te[feats].astype(float).fillna(-1))))
    return np.mean(errs)
print(f"rows {len(df):,} | NGS cov {df.avg_separation.notna().mean():.0%} rec / {df.avg_time_to_throw.notna().mean():.0%} pass | xFP cov {df.xfp_pg.notna().mean():.0%} | cpoe cov {df.cpoe.notna().mean():.0%}")
b=walk(BASE); print(f"\nBASELINE (production features)      MAE {b:.4f}")
allf=[]
for lbl,fs in BL.items():
    fs=[f for f in fs if f in df.columns]; allf+=fs
    m=walk(BASE+fs); print(f"+{lbl:35s} MAE {m:.4f}  ({b-m:+.4f})")
m=walk(BASE+allf); print(f"+ALL blocks                          MAE {m:.4f}  ({b-m:+.4f})")
