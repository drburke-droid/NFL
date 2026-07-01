"""Validate the rookie-WR uptick (historical_trends.md #5): are recent rookie WRs UNDER-predicted by the
ACTUAL rookie model, and does a recency bias-correction beat it out-of-sample? Discipline: only 'edge' if
it cuts walk-forward MAE."""
import warnings; warnings.filterwarnings("ignore")
import os, sqlite3, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
spec=importlib.util.spec_from_file_location("mr", os.path.join("scripts","model_rookie.py"))
mr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mr)
con=sqlite3.connect(mr.DB); df=mr.build(con); con.close()
test=list(range(2016,2026))
full=mr.walk(df, mr.BASE, test)              # actual model, walk-forward
full["resid"]=full.ppg-full.pred             # +ve = model UNDER-predicts
wr=full[full.position=="WR"].copy()
print(f"Rookie WR walk-forward (test {test[0]}-{test[-1]}, n={len(wr)})")
# 1) residual trend over time (is under-prediction growing?)
b=np.polyfit(wr.season, wr.resid, 1)
x=np.column_stack([np.ones(len(wr)),wr.season-wr.season.mean()]); yb,*_=np.linalg.lstsq(x,wr.resid.values,rcond=None)
se=np.sqrt((((wr.resid.values-x@yb)**2).sum()/(len(wr)-2))*np.linalg.pinv(x.T@x)[1,1])
print(f"  mean resid by era: 2016-20 {wr[wr.season<=2020].resid.mean():+.2f} | 2021-25 {wr[wr.season>=2021].resid.mean():+.2f}")
print(f"  resid~season slope {b[0]:+.3f} PPG/yr (t={yb[1]/se:+.1f})  [+ => model increasingly under-predicts rookie WRs]")
print(f"  rho by era: 2016-20 {wr[wr.season<=2020][['pred','ppg']].corr().iloc[0,1]:.2f} | 2021-25 {wr[wr.season>=2021][['pred','ppg']].corr().iloc[0,1]:.2f}")
# 2) correction test: trailing-3yr mean WR residual as bias offset, applied to each test season
def wf_corrected(trail):
    err_base,err_corr=[],[]
    for T in range(2019,2026):
        te=wr[wr.season==T]
        if len(te)==0: continue
        past=wr[(wr.season<T)&(wr.season>=T-trail)]
        off=past.resid.mean() if len(past) else 0.0
        err_base+=list(np.abs(te.resid)); err_corr+=list(np.abs(te.resid-off))
    return np.mean(err_base),np.mean(err_corr),off
for trail in (3,5):
    mb,mc,off=wf_corrected(trail)
    print(f"  bias-correct (trail {trail}y): base MAE {mb:.3f} -> corrected {mc:.3f}  ({mb-mc:+.3f}); last offset {off:+.2f}")
# 3) is it a right-tail (variance) story? hit-rate calibration recent era
rec=wr[wr.season>=2021]
print(f"  2021-25: actual >=12 PPG rate {100*(rec.ppg>=12).mean():.0f}% vs model-predicted-mean {rec.pred.mean():.1f}; actual mean {rec.ppg.mean():.1f}")
