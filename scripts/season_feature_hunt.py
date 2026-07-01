"""Feature hunt over the ACTUAL season model (model_season.py walk-forward, 2016-2025).
1) Candidate feature blocks never wired in: INJURY (validated but absent from MODEL_FEATURES),
   half-season trend (nflv_half_trend), career comps (nflv_comp_features), opportunity (nflv_opportunity).
2) Residual autopsy: systematic bias pockets by segment.
3) Blend-alpha sweep: optimal model-vs-FFA weight per position (board BLEND_W default)."""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s=importlib.util.spec_from_file_location("ms",os.path.join(os.path.dirname(os.path.abspath(__file__)),"model_season.py"))
MS=importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)
DB=MS.DB; POS=MS.POS
con=sqlite3.connect(DB)
df=pd.read_sql("SELECT * FROM season_dataset",con); df=df[df.next_ppg.notna()]
df=MS.attach_ffa(df,con)
ht=pd.read_sql("SELECT * FROM nflv_half_trend",con)
cf=pd.read_sql("SELECT * FROM nflv_comp_features",con)
opp=pd.read_sql("SELECT player_id,season,vac_rb_carries,inc_rb_carries,vac_pc_targets,inc_pc_targets FROM nflv_opportunity",con)
g2=pd.read_sql("SELECT player_id,season,games FROM nflv_season",con).drop_duplicates(["player_id","season"]).rename(columns={"games":"prior2_games"}); g2["season"]+=2
con.close()
df=df.merge(ht,on=["player_id","season"],how="left").merge(cf,on=["player_id","season"],how="left") \
     .merge(opp,on=["player_id","season"],how="left").merge(g2,on=["player_id","season"],how="left")
df=MS.add_injury_features(df)
df["pos_id"]=df.position.map({p:i for i,p in enumerate(POS)})
BASE=MS.MODEL_FEATURES
HT=["ht_d_ppg","ht_h2_ppg","ht_d_snap","ht_h2_snap","ht_d_tch","ht_d_tgtsh","ht_h2_tgtsh","ht_slope"]
CF=["comp_prime","comp_bust","comp_elite","comp_dist","comp_n"]
OPP=["vac_rb_carries","inc_rb_carries","vac_pc_targets","inc_pc_targets"]
INJ=MS.INJURY_FEATURES
params=dict(objective="regression_l1",n_estimators=500,learning_rate=0.03,num_leaves=31,
            min_child_samples=40,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
def walk(feats):
    out=[]
    for T in range(2016,2026):
        tr=df[df.season<T]; te=df[df.season==T]
        if not len(te): continue
        m=lgb.LGBMRegressor(**params).fit(tr[feats].astype(float).fillna(-1),tr.next_ppg)
        t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
    return pd.concat(out,ignore_index=True)
def rep(p,label,base=None):
    mae=mean_absolute_error(p.next_ppg,p.pred); rho=spearmanr(p.pred,p.next_ppg)[0]
    d=f"  ({base-mae:+.4f})" if base else ""
    bypos=" ".join(f"{q}:{mean_absolute_error(p[p.position==q].next_ppg,p[p.position==q].pred):.2f}" for q in POS)
    print(f"  {label:28s} MAE {mae:.4f}{d}  rho {rho:.3f}  [{bypos}]"); return mae
print(f"rows={len(df):,}  seasons {df.season.min()}-{df.season.max()}  HT cov {df.ht_slope.notna().mean():.0%}  CF cov {df.comp_prime.notna().mean():.0%}")
print("\n=== 1) FEATURE BLOCKS over actual walk-forward (2016-2025) ===")
p0=walk(BASE); m0=rep(p0,"PRODUCTION (MODEL_FEATURES)")
for lbl,fs in [("+INJURY",BASE+INJ),("+HALF-TREND",BASE+HT),("+COMPS",BASE+CF),("+OPPORTUNITY",BASE+OPP),
               ("+INJ+HT",BASE+INJ+HT),("+ALL",BASE+INJ+HT+CF+OPP)]:
    rep(walk(fs),lbl,m0)
print("\n=== 2) RESIDUAL AUTOPSY (production preds; resid=actual-pred, + = under-projected) ===")
p0["resid"]=p0.next_ppg-p0.pred
def seg(name,mask):
    d=p0[mask]
    if len(d)<60: return
    t=d.resid.mean()/(d.resid.std()/np.sqrt(len(d)))
    flag=" <<<" if abs(t)>3 else ""
    print(f"  {name:34s} n={len(d):4d}  bias {d.resid.mean():+.2f} (t={t:+.1f})  MAE {d.resid.abs().mean():.2f}{flag}")
for a,b in [(21,24),(24,27),(27,30),(30,36)]: seg(f"age {a}-{b}",(p0.age>=a)&(p0.age<b))
for q in POS: seg(f"pos {q}",p0.position==q)
seg("team change",p0.team_change==1); seg("same team",p0.team_change==0)
seg("short prior season (<=11 g)",p0.prior_games<=11); seg("full prior season",p0.prior_games>=15)
seg("2nd-yr player",p0.years_exp==1); seg("yr 3-4",p0.years_exp.isin([2,3]))
seg("no FFA",p0.ffa_points.isna()); seg("FFA elite (pos_rank<=6)",p0.ffa_pos_rank<=6)
seg("FFA mid (rank 7-24)",(p0.ffa_pos_rank>=7)&(p0.ffa_pos_rank<=24))
cvq=p0.prior_cv.quantile([.75]); seg("volatile prior (cv top-25%)",p0.prior_cv>=cvq.iloc[0])
seg("high prior ppg (>=14)",p0.prior_ppg>=14); seg("low prior ppg (<7)",p0.prior_ppg<7)
print("\n=== 3) BLEND ALPHA SWEEP (model vs FFA, per-position rescale like the board) ===")
d=p0[p0.ffa_points.notna()].copy()
d["ffa_z"]=d.groupby(["season","position"]).ffa_points.transform(lambda x:(x-x.mean())/(x.std() or 1))
grp=d.groupby(["season","position"]).pred
d["ffa_ppg"]=d.ffa_z*grp.transform("std")+grp.transform("mean")
for q in POS+["ALL"]:
    dd=d if q=="ALL" else d[d.position==q]
    maes={a:mean_absolute_error(dd.next_ppg,a*dd.pred+(1-a)*dd.ffa_ppg) for a in np.arange(0,1.01,.1)}
    best=min(maes,key=maes.get)
    print(f"  {q:3s} n={len(dd):4d}  alpha 1.0(model)={maes[1.0]:.3f}  0.0(FFA)={maes[0.0]:.3f}  BEST a={best:.1f} MAE={maes[best]:.3f}")

print("\n=== 4) WALK-FORWARD SEGMENT BIAS CORRECTION (age30+/team-change/star fade) ===")
# For each test season T: estimate each segment's mean residual on PRIOR test seasons only, subtract.
p0=p0.sort_values("season").reset_index(drop=True)
SEGS={"age30":lambda x:x.age>=30,"tmchg":lambda x:x.team_change==1,"star":lambda x:x.prior_ppg>=14}
for name,fn in list(SEGS.items())+[("all3","COMBO")]:
    errb,errc=[],[]
    for T in range(2019,2026):
        past=p0[p0.season<T]; te=p0[p0.season==T].copy()
        adj=np.zeros(len(te))
        for nm2,f2 in SEGS.items():
            if name not in (nm2,"all3"): continue
            m=f2(past); off=past[m].resid.mean() if m.sum()>=100 else 0.0
            adj=adj+np.where(f2(te),off,0.0)
        errb+=list((te.next_ppg-te.pred).abs()); errc+=list((te.next_ppg-(te.pred+adj)).abs())
    print(f"  {name:6s} base MAE {np.mean(errb):.4f} -> corrected {np.mean(errc):.4f}  ({np.mean(errb)-np.mean(errc):+.4f})")

print("\n=== 5) ROBUSTNESS ===")
# (a) per-year delta of the all3 correction
for T in range(2019,2026):
    past=p0[p0.season<T]; te=p0[p0.season==T].copy()
    adj=np.zeros(len(te))
    for nm2,f2 in SEGS.items():
        m=f2(past); off=past[m].resid.mean() if m.sum()>=100 else 0.0
        adj=adj+np.where(f2(te),off,0.0)
    b=(te.next_ppg-te.pred).abs().mean(); c=(te.next_ppg-(te.pred+adj)).abs().mean()
    print(f"  {T}: base {b:.3f} -> corr {c:.3f} ({b-c:+.4f})  touched {int((adj!=0).sum())}/{len(te)}")
# (b) blend: current default 0.5 vs optimal 0.7 FFA
for q in POS+["ALL"]:
    dd=d if q=="ALL" else d[d.position==q]
    m5=mean_absolute_error(dd.next_ppg,.5*dd.pred+.5*dd.ffa_ppg)
    m7=mean_absolute_error(dd.next_ppg,.3*dd.pred+.7*dd.ffa_ppg)
    print(f"  blend {q:3s}: 50/50={m5:.3f}  30model/70ffa={m7:.3f}  ({m5-m7:+.4f})")
