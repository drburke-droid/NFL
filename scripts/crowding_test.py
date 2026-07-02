"""Teammate-quality crowding: does a strong WR1/TE1/RB1 depress a teammate's next-year production,
and does adding explicit 'competition for touches' features beat the actual season model?
Features (target season T, from T-1 usage of the players on the T roster):
  mates_tgtsh  = sum of teammates' prior target shares (competition for the target pie)
  mates_top    = the biggest teammate's prior target share (alpha dominance)
  pecking      = player's rank among teammates by prior target share
  mates_rbcar  = teammates' prior carries/g (RB competition)"""
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
con=sqlite3.connect(MS.DB)
df=pd.read_sql("SELECT * FROM season_dataset",con); df=df[df.next_ppg.notna()]
df=MS.attach_ffa(df,con); con.close()
df["pos_id"]=df.position.map({p:i for i,p in enumerate(MS.POS)})
# team composition in target season = players who logged stats for that team in T (season_dataset.team)
g=df.groupby(["team","season"])
df["team_tgtsh_sum"]=g.prior_target_share.transform(lambda x:x.fillna(0).sum())
df["mates_tgtsh"]=df.team_tgtsh_sum-df.prior_target_share.fillna(0)
tmax=g.prior_target_share.transform("max")
df["mates_top"]=np.where(df.prior_target_share.fillna(0)>=tmax,
    g.prior_target_share.transform(lambda x:x.fillna(0).nlargest(2).min() if len(x)>1 else 0),tmax)
df["pecking"]=g.prior_target_share.rank(ascending=False,method="min")
car=df.prior_carries_pg.fillna(0)
df["team_car_sum"]=g.prior_carries_pg.transform(lambda x:x.fillna(0).sum())
df["mates_rbcar"]=df.team_car_sum-car
CROWD=["mates_tgtsh","mates_top","pecking","mates_rbcar"]
# ---- 1) descriptive: is crowding REAL? partial regression next_ppg ~ own prior + crowding ----
print("=== 1) IS CROWDING REAL? (WR/TE/RB, next_ppg ~ prior_ppg + prior_tgtsh + age + signal) ===")
d=df[df.position.isin(["WR","TE","RB"])].dropna(subset=["next_ppg","prior_ppg","age"]).copy()
d["tsh"]=d.prior_target_share.fillna(0)
for sig in CROWD:
    x=d.dropna(subset=[sig])
    X=np.column_stack([np.ones(len(x)),x.prior_ppg,x.tsh,x.age,x[sig]])
    b,*_=np.linalg.lstsq(X,x.next_ppg.values,rcond=None)
    res=x.next_ppg.values-X@b
    se=np.sqrt((res@res/(len(x)-5))*np.linalg.pinv(X.T@X)[-1,-1])
    print(f"  {sig:14s} coef {b[-1]:+.3f} (t={b[-1]/se:+.1f})  n={len(x)}")
# the user's exact scenario: WR2s (pecking=2 among pass-catchers) split by teammate strength
wr2=d[(d.position=="WR")&(d.pecking==2)]
hi=wr2[wr2.mates_top>=0.26]; lo=wr2[wr2.mates_top<=0.20]
print(f"\n  WR2 next-yr PPG: alpha-WR1 team (top share>=26%) {hi.next_ppg.mean():.1f} (n={len(hi)}) vs modest WR1 {lo.next_ppg.mean():.1f} (n={len(lo)})")
print(f"  ...controlling for own prior: alpha {(hi.next_ppg-hi.prior_ppg).mean():+.1f} chg vs modest {(lo.next_ppg-lo.prior_ppg).mean():+.1f} chg")
# ---- 2) does the ACTUAL model already price it? ----
print("\n=== 2) OVER THE ACTUAL MODEL (walk-forward 2016-2025) ===")
params=dict(objective="regression_l1",n_estimators=500,learning_rate=0.03,num_leaves=31,
            min_child_samples=40,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
def walk(feats):
    errs=[]; segs={}
    for T in range(2016,2026):
        tr,te=df[df.season<T],df[df.season==T]
        if not len(te): continue
        m=lgb.LGBMRegressor(**params).fit(tr[feats].astype(float).fillna(-1),tr.next_ppg)
        p=m.predict(te[feats].astype(float).fillna(-1))
        errs+=list(np.abs(te.next_ppg.values-p))
    return np.mean(errs)
b0=walk(MS.MODEL_FEATURES); b1=walk(MS.MODEL_FEATURES+CROWD)
print(f"  baseline {b0:.4f} | +crowding {b1:.4f}  ({b0-b1:+.4f})")
# residual check: does the model err on crowded WR2s?
tr,parts=df[df.season<2016],[]
for T in range(2016,2026):
    trn,te=df[df.season<T],df[df.season==T]
    if not len(te): continue
    m=lgb.LGBMRegressor(**params).fit(trn[MS.MODEL_FEATURES].astype(float).fillna(-1),trn.next_ppg)
    t=te.copy(); t["pred"]=m.predict(te[MS.MODEL_FEATURES].astype(float).fillna(-1)); parts.append(t)
P=pd.concat(parts); P["resid"]=P.next_ppg-P.pred
w2=P[(P.position=="WR")&(P.pecking==2)]
print(f"  model residual on WR2s w/ alpha WR1 (share>=26%): {w2[w2.mates_top>=0.26].resid.mean():+.2f} (n={len(w2[w2.mates_top>=0.26])})")
print(f"  model residual on WR2s w/ modest WR1 (<=20%):     {w2[w2.mates_top<=0.20].resid.mean():+.2f} (n={len(w2[w2.mates_top<=0.20])})")

# ---- 3) production-style validation: segment correction (crowded vs uncrowded WR2) ----
print("\n=== 3) SEGMENT CORRECTION over production-style scorer (quantile, BASE+INJ+FFA) ===")
con=sqlite3.connect(MS.DB)
ns=pd.read_sql("SELECT player_id,season,games FROM nflv_season",con).drop_duplicates(["player_id","season"]); con.close()
g2=ns.rename(columns={"games":"prior2_games"}); g2["season"]+=2
dd=df.merge(g2,on=["player_id","season"],how="left"); dd=MS.add_injury_features(dd)
dd=dd[dd.prior_games>=3].copy()
FE=MS.FEATURES+MS.INJURY_FEATURES+MS.FFA_FEATURES
GB=dict(objective="quantile",alpha=0.5,n_estimators=300,learning_rate=0.04,num_leaves=20,
        min_child_samples=30,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
parts=[]
for pos in MS.POS:
    d2=dd[dd.position==pos]
    for T in range(2016,2026):
        tr,te=d2[d2.season<T],d2[d2.season==T]
        if not len(te) or len(tr)<60: continue
        m=lgb.LGBMRegressor(**GB).fit(tr[FE].astype(float).fillna(-1),tr.next_ppg)
        t=te.copy(); t["pred"]=m.predict(te[FE].astype(float).fillna(-1)); parts.append(t)
Q=pd.concat(parts,ignore_index=True); Q["resid"]=Q.next_ppg-Q.pred
SEG={"crowdedWR2":lambda x:(x.position=="WR")&(x.pecking==2)&(x.mates_top>=0.26),
     "openWR2":lambda x:(x.position=="WR")&(x.pecking==2)&(x.mates_top<=0.20)}
for nm,f in SEG.items(): print(f"  {nm}: full-period bias {Q[f(Q)].resid.mean():+.2f} (n={f(Q).sum()})")
errb,errc=[],[]
for T in range(2019,2026):
    past=Q[Q.season<T]; te=Q[Q.season==T].copy()
    adj=np.zeros(len(te))
    for nm,f in SEG.items():
        m_=f(past); off=past[m_].resid.mean() if m_.sum()>=40 else 0.0
        adj+=np.where(f(te),off,0.0)
    b=(te.next_ppg-te.pred).abs(); c=(te.next_ppg-(te.pred+adj)).abs()
    errb+=list(b); errc+=list(c)
    print(f"  {T}: {b.mean():.3f} -> {c.mean():.3f} ({b.mean()-c.mean():+.4f})  touched {int((adj!=0).sum())}")
print(f"  TOTAL: {np.mean(errb):.4f} -> {np.mean(errc):.4f} ({np.mean(errb)-np.mean(errc):+.4f})")

# ---- 4) per-year consistency of adding CROWD features to the mean model ----
print("\n=== 4) +CROWD features per-year (l1 model) ===")
wins=0; tot=0
for T in range(2016,2026):
    tr,te=df[df.season<T],df[df.season==T]
    if not len(te): continue
    m0=lgb.LGBMRegressor(**params).fit(tr[MS.MODEL_FEATURES].astype(float).fillna(-1),tr.next_ppg)
    m1=lgb.LGBMRegressor(**params).fit(tr[MS.MODEL_FEATURES+CROWD].astype(float).fillna(-1),tr.next_ppg)
    e0=np.abs(te.next_ppg.values-m0.predict(te[MS.MODEL_FEATURES].astype(float).fillna(-1))).mean()
    e1=np.abs(te.next_ppg.values-m1.predict(te[MS.MODEL_FEATURES+CROWD].astype(float).fillna(-1))).mean()
    wins+=e1<e0; tot+=1
    print(f"  {T}: {e0:.3f} -> {e1:.3f} ({e0-e1:+.4f})")
print(f"  wins {wins}/{tot}")
