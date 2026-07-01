"""Validate segment bias correction over the PRODUCTION scorer (projection_overhaul: per-position
quantile-0.5 LGBM, BASE+INJ+FFA). Segments (from season_feature_hunt.py autopsy): age>=30, team_change,
prior_ppg>=14. Walk-forward: offsets from mean residual of PRIOR test seasons only."""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s=importlib.util.spec_from_file_location("ms",os.path.join(os.path.dirname(os.path.abspath(__file__)),"model_season.py"))
MS=importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)
POS=["QB","RB","WR","TE"]
GB=dict(n_estimators=300,learning_rate=0.04,num_leaves=20,min_child_samples=30,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
con=sqlite3.connect(MS.DB)
df=pd.read_sql("SELECT * FROM season_dataset",con)
ns=pd.read_sql("SELECT player_id,season,games FROM nflv_season",con).drop_duplicates(["player_id","season"])
df=MS.attach_ffa(df,con); con.close()
g2=ns.rename(columns={"games":"prior2_games"}); g2["season"]+=2
df=df.merge(g2,on=["player_id","season"],how="left"); df=MS.add_injury_features(df)
df=df[df.next_ppg.notna()&(df.prior_games>=3)].copy()
FEATS=MS.FEATURES+MS.INJURY_FEATURES+MS.FFA_FEATURES
parts=[]
for pos in POS:
    d=df[df.position==pos]
    for T in range(2016,2026):
        tr,te=d[d.season<T],d[d.season==T]
        if not len(te) or len(tr)<60: continue
        m=lgb.LGBMRegressor(objective="quantile",alpha=0.5,**GB).fit(tr[FEATS].astype(float).fillna(-1),tr.next_ppg)
        t=te.copy(); t["pred"]=m.predict(te[FEATS].astype(float).fillna(-1)); parts.append(t)
p=pd.concat(parts,ignore_index=True); p["resid"]=p.next_ppg-p.pred
SEGS={"age30":lambda x:x.age>=30,"tmchg":lambda x:x.team_change==1,"star":lambda x:x.prior_ppg>=14}
print("Production-scorer walk-forward residual by segment (full period):")
for nm,f in SEGS.items():
    m=f(p); print(f"  {nm:6s} n={m.sum():4d}  bias {p[m].resid.mean():+.3f}")
print("\nWalk-forward correction (offsets from prior test seasons only):")
tot_b=tot_c=[]
errb,errc,yr=[],[],{}
for T in range(2019,2026):
    past=p[p.season<T]; te=p[p.season==T].copy()
    adj=np.zeros(len(te))
    for nm,f in SEGS.items():
        m=f(past); off=past[m].resid.mean() if m.sum()>=100 else 0.0
        adj+=np.where(f(te),off,0.0)
    b=(te.next_ppg-te.pred).abs(); c=(te.next_ppg-(te.pred+adj)).abs()
    errb+=list(b); errc+=list(c); yr[T]=(b.mean(),c.mean())
for T,(b,c) in yr.items(): print(f"  {T}: {b:.3f} -> {c:.3f} ({b-c:+.4f})")
print(f"  TOTAL: {np.mean(errb):.4f} -> {np.mean(errc):.4f} ({np.mean(errb)-np.mean(errc):+.4f})")
print("\nFinal production offsets (mean residual, all 2016-2025):")
for nm,f in SEGS.items():
    m=f(p); print(f"  {nm}: {p[m].resid.mean():+.3f} (n={m.sum()})")
