"""Deep research: what predicts a $1-3 player becoming a STAR next season?
Cohort:  preseason auction value <= $3 (ffa_aav<=3, or absent from FFA = deep $1 bin).
Star:    next_pos_finish in the upper half of the position's startable VORP ranks
         (QB<=6, RB<=12, WR<=12, TE<=6).  Also tracked: 'startable' (QB12/RB24/WR24/TE12).
Method:  univariate signal lift table + walk-forward LGBM (test 2016-2025) + 2026 ranked list."""
import os, sqlite3, warnings, sys, io, contextlib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(name):
    s=importlib.util.spec_from_file_location(name,os.path.join(ROOT,"scripts",name+".py"))
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
MS=load("model_season")
DB=MS.DB; POS=["QB","RB","WR","TE"]
STAR={"QB":6,"RB":12,"WR":12,"TE":6}; STARTABLE={"QB":12,"RB":24,"WR":24,"TE":12}
con=sqlite3.connect(DB)
df=pd.read_sql("SELECT * FROM season_dataset",con)
ffa=pd.read_sql("SELECT season,player_id,ffa_aav,ffa_points FROM nflv_ffa_proj WHERE player_id IS NOT NULL",con).drop_duplicates(["season","player_id"])
ht=pd.read_sql("SELECT * FROM nflv_half_trend",con)
opp=pd.read_sql("SELECT player_id,season,vac_rb_carries,inc_rb_carries,vac_pc_targets,inc_pc_targets FROM nflv_opportunity",con)
xf=pd.read_sql("""SELECT player_id, season, SUM(total_fantasy_points_exp) xfp, SUM(total_fantasy_points_diff) xgap,
                  COUNT(DISTINCT week) wk FROM nflv_ff_opp GROUP BY player_id, season""",con)
names=pd.read_sql("SELECT player_id,player_display_name FROM nflv_season",con).drop_duplicates("player_id")
con.close()
xf["xfp_pg"]=xf.xfp/xf.wk; xf["xgap_pg"]=xf.xgap/xf.wk
xf["season"]=xf.season.astype(int)+1                     # prior-season xFP -> keyed to target season
df=df.merge(ffa,on=["season","player_id"],how="left").merge(ht,on=["player_id","season"],how="left") \
     .merge(opp,on=["player_id","season"],how="left").merge(xf[["player_id","season","xfp_pg","xgap_pg"]],on=["player_id","season"],how="left")
df["cheap"]=((df.ffa_aav<=3)|df.ffa_aav.isna()).astype(int)
df["star"]=(df.next_pos_finish<=df.position.map(STAR)).astype(int)
df["startable"]=(df.next_pos_finish<=df.position.map(STARTABLE)).astype(int)
d=df[(df.cheap==1)&df.next_pos_finish.notna()&(df.season>=2014)].copy()
print(f"CHEAP cohort ($1-3 or unranked), 2014-2025: n={len(d):,}")
print("Base rates: star {:.1%}   startable {:.1%}".format(d.star.mean(),d.startable.mean()))
print("By pos: "+"  ".join(f"{p}:{d[d.position==p].star.mean():.1%}(n={len(d[d.position==p])})" for p in POS))
# ---------- 1) UNIVARIATE SIGNAL LIFT ----------
print("\n=== SIGNAL LIFT: P(star | signal) vs base ===")
def lift(name,mask):
    m=d[mask]; 
    if len(m)<80: return
    print(f"  {name:38s} n={len(m):4d}  star {m.star.mean():5.1%} ({m.star.mean()/d.star.mean():4.1f}x)  startable {m.startable.mean():5.1%}")
lift("years_exp <= 2 (yr-2/yr-3 player)",d.years_exp<=2)
lift("age <= 24",d.age<=24)
lift("draft_pick <= 75 (day-1/2 capital)",d.draft_pick<=75)
lift("draft_pick <= 75 AND exp <= 2",(d.draft_pick<=75)&(d.years_exp<=2))
lift("late surge: H2 snap +10pp",d.ht_d_snap>=10)
lift("H2 target share >= 15%",d.ht_h2_tgtsh>=0.15)
lift("H2 ppg >= 8",d.ht_h2_ppg>=8)
lift("vacated targets >= 80",d.vac_pc_targets>=80)
lift("vacated RB carries >= 120",d.vac_rb_carries>=120)
lift("prior snap% >= 50",d.prior_off_pct>=0.5)
lift("prior target share >= 12%",d.prior_target_share>=0.12)
lift("unlucky: xFP gap <= -1.5/wk",d.xgap_pg<=-1.5)
lift("over-produced: xFP gap >= +1.5/wk",d.xgap_pg>=1.5)
lift("team change",d.team_change==1)
lift("prior_ppg >= 7 (was semi-relevant)",d.prior_ppg>=7)
lift("COMBO: capital+young+H2 role",(d.draft_pick<=105)&(d.years_exp<=2)&((d.ht_d_snap>=8)|(d.ht_h2_tgtsh>=0.14)))
# ---------- 2) WALK-FORWARD MODEL ----------
FE=["age","years_exp","draft_pick","draft_round","prior_ppg","prior2_ppg","prior_games","prior_off_pct",
    "prior_target_share","prior_air_yards_share","prior_wopr","prior_targets_pg","prior_carries_pg",
    "prior_cv","prior_receiving_epa","team_change","weight","forty","ffa_aav",
    "ht_d_snap","ht_h2_snap","ht_d_tgtsh","ht_h2_tgtsh","ht_d_ppg","ht_h2_ppg","ht_slope",
    "vac_pc_targets","vac_rb_carries","inc_pc_targets","inc_rb_carries","xfp_pg","xgap_pg"]
FE=[f for f in FE if f in d.columns]; d["pos_id"]=d.position.map({p:i for i,p in enumerate(POS)}); FE+=["pos_id"]
GB=dict(objective="binary",n_estimators=250,learning_rate=0.04,num_leaves=15,min_child_samples=25,
        subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
preds=[]
for T in range(2016,2026):
    tr,te=d[d.season<T],d[d.season==T]
    if not len(te) or tr.star.sum()<12: continue
    m=lgb.LGBMClassifier(**GB).fit(tr[FE].astype(float).fillna(-1),tr.star)
    t=te.copy(); t["p"]=m.predict_proba(te[FE].astype(float).fillna(-1))[:,1]; preds.append(t)
P=pd.concat(preds,ignore_index=True)
auc=roc_auc_score(P.star,P.p)
top10=P.sort_values("p",ascending=False).groupby("season").head(10)
print(f"\n=== WALK-FORWARD (2016-2025, n={len(P):,}) ===")
print(f"  AUC {auc:.3f} | base {P.star.mean():.1%} | precision@10/season: star {top10.star.mean():.1%} ({top10.star.mean()/P.star.mean():.1f}x), startable {top10.startable.mean():.1%}")
mf=lgb.LGBMClassifier(**GB).fit(d[FE].astype(float).fillna(-1),d.star)
imp=pd.DataFrame({"f":FE,"i":mf.feature_importances_}).sort_values("i",ascending=False)
print("  top features: "+", ".join(f"{r.f}" for r in imp.head(10).itertuples()))
print("\n  Historical top-10 hits (would-have-been lottery wins):")
hits=top10[top10.star==1].merge(names,on="player_id",how="left")
for r in hits.sort_values("season").itertuples():
    print(f"    {r.season} {r.position:3s} {r.player_display_name[:22]:22s} p={r.p:.2f} -> finished #{int(r.next_pos_finish)}")
d.attrs={}; P.to_pickle(os.path.join(ROOT,"outputs","lottery_wf.pkl"))

# ---------- 3) 2026 LOTTERY LIST (score the current $1-3 pool) ----------
print("\n=== 2026 LOTTERY TICKETS (exp price <= $3 on our board) ===")
bsd=load("build_season_dataset")
con=sqlite3.connect(DB); sf=bsd.build_season_frame(con); con.close()
prior=sf[sf.season==2025][bsd.FEAT_COLS].copy()
prior.columns=["player_id","player_display_name","position","prior_season","prior_team"]+["prior_"+c for c in bsd.FEAT_COLS[5:]]
prior["season"]=2026
p2=sf[sf.season==2024][["player_id","ppg"]].rename(columns={"ppg":"prior2_ppg"})
ctx=sf[sf.season==2025][["player_id","age","years_exp","weight","forty","draft_round","draft_pick"]]
ctx=ctx.copy(); ctx["age"]+=1; ctx["years_exp"]+=1
v26=prior.merge(p2,on="player_id",how="left").merge(ctx,on="player_id",how="left")
con=sqlite3.connect(DB)
r26=pd.read_sql("SELECT player_id, team AS t26 FROM nflv_rosters_2026",con).drop_duplicates("player_id"); con.close()
v26=v26.merge(r26,on="player_id",how="left")
v26["team_change"]=(v26.t26.notna()&v26.prior_team.notna()&(v26.t26!=v26.prior_team)).astype(int)
v26=v26.merge(ht,on=["player_id","season"],how="left").merge(opp,on=["player_id","season"],how="left") \
       .merge(xf[["player_id","season","xfp_pg","xgap_pg"]],on=["player_id","season"],how="left")
v26["ffa_aav"]=np.nan; v26["pos_id"]=v26.position.map({p:i for i,p in enumerate(POS)})
for f in FE:
    if f not in v26.columns: v26[f]=np.nan
v26=v26[v26.position.isin(POS)&(v26.prior_games.fillna(0)>=1)].copy()
v26["p_star"]=mf.predict_proba(v26[FE].astype(float).fillna(-1))[:,1]
# expected price from the league bid-curve engine (predict_keepers)
pk=None
with contextlib.redirect_stdout(io.StringIO()): pk=load("predict_keepers")
exp_by_name={}
for pl in pk.P:
    exp_by_name[(pk.norm(pl["name"]),pl["position"])]=pk.exp_price.get(pk.pidof(pl),1)
import re
nrm=lambda s: re.sub(r"\s+"," ",re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",re.sub(r"[^a-z ]","",str(s).lower()))).strip()
v26["expd"]=[exp_by_name.get((nrm(n),p),None) for n,p in zip(v26.player_display_name,v26.position)]
cheap26=v26[v26["expd"].notna()&(v26["expd"]<=3)].sort_values("p_star",ascending=False)
print(f"pool: {len(cheap26)} players at exp <= $3\n")
def why(r):
    w=[]
    if (r.prior_target_share or 0)>=0.12: w.append(f"tgt%{r.prior_target_share:.0%}")
    if (r.ht_h2_ppg or 0)>=8: w.append(f"H2ppg{r.ht_h2_ppg:.0f}")
    if (r.prior_off_pct or 0)>=0.5: w.append(f"snap{r.prior_off_pct:.0%}")
    if (r.draft_pick or 300)<=75: w.append(f"pick{int(r.draft_pick)}")
    if (r.xgap_pg or 0)>=1.0: w.append("xFP+")
    if (r.years_exp or 9)<=2: w.append("yr%d"%(r.years_exp+0))
    if r.team_change==1: w.append("MOVED(neg)")
    return ",".join(w) or "-"
for r in cheap26.head(24).itertuples():
    print(f"  {r.position:3s} {r.player_display_name[:23]:23s} {str(r.t26 or r.prior_team):4s} exp ${int(r.expd):>2}  p(star) {r.p_star:5.1%}  [{why(r)}]")
