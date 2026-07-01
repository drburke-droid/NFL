"""Build a real 'incumbent competition' feature (returning players' prior carries/targets locked up
ahead of the rookie) and test whether it improves rookie projections."""
import warnings; warnings.filterwarnings("ignore")
import sqlite3, numpy as np, pandas as pd, importlib.util, os
import lightgbm as lgb
spec=importlib.util.spec_from_file_location("mr","scripts/model_rookie.py"); mr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mr)
con=sqlite3.connect("db/nfl_odds.db"); df=mr.build(con)
seas=pd.read_sql("SELECT player_id,season,recent_team team,position,carries,targets FROM nflv_season WHERE position IN ('RB','WR','TE')",con); con.close()
seas[["carries","targets"]]=seas[["carries","targets"]].fillna(0)
prior={(r.player_id,r.season+1):(r.carries,r.targets) for r in seas.itertuples()}   # a player's PRIOR-yr usage
# incumbent = returning players on the rookie's team-year: their prior-year usage (competition ahead)
inc={}
for r in seas.itertuples():
    pc,pt=prior.get((r.player_id,r.season),(0,0))                                    # this player's prior usage (=returning if >0)
    key=(r.team,r.season,r.position); d=inc.setdefault(key,[0,0]); d[0]+=pc; d[1]+=pt
df["inc_carries"]=[inc.get((t,s,"RB"),[0,0])[0] for t,s in zip(df.team,df.season)]
df["inc_targets"]=[inc.get((t,s,p),[0,0])[1] for t,s,p in zip(df.team,df.season,df.position)]
BASE=mr.BASE; NEW=BASE+["inc_carries","inc_targets"]
def wf(feats):
    out=[]
    for T in range(2018,2025):
        tr=df[df.season<T]; te=df[df.season==T]
        if len(te)<10 or len(tr)<150: continue
        m=lgb.LGBMRegressor(**mr.PARAMS).fit(tr[feats].astype(float).fillna(-1),tr["ppg"])
        t=te.copy(); t["pred"]=m.predict(te[feats].astype(float).fillna(-1)); out.append(t)
    r=pd.concat(out); return np.abs(r.pred-r.ppg).mean(),r
mb,rb=wf(BASE); mo,ro=wf(NEW)
print("Rookie model walk-forward MAE (2018-24, n=%d):"%len(rb))
print("  BASE:                         %.3f"%mb)
print("  + incumbent competition:      %.3f   (Δ %.3f)"%(mo,mb-mo))
for pos in ["RB","WR","TE"]:
    a=rb[rb.position==pos]; b=ro[ro.position==pos]
    print("  %s: BASE %.3f -> +inc %.3f (Δ %+.3f, n=%d)"%(pos,np.abs(a.pred-a.ppg).mean(),np.abs(b.pred-b.ppg).mean(),np.abs(a.pred-a.ppg).mean()-np.abs(b.pred-b.ppg).mean(),len(a)))
r=df[df.position=='RB']; print("\n  corr(incumbent RB carries, rookie RB ppg): %.2f"%r[["inc_carries","ppg"]].corr().iloc[0,1])
m=lgb.LGBMRegressor(**mr.PARAMS).fit(df[NEW].astype(float).fillna(-1),df["ppg"]); imp=pd.Series(m.feature_importances_,index=NEW).sort_values(ascending=False)
print("  inc_carries importance rank: %d/%d"%(list(imp.index).index("inc_carries")+1,len(NEW)))
