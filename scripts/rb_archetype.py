"""RB archetype thesis (Gdula/4for4): pass-game specialists fading, medium-volume dual-threats rising.
Confirm the trend in our data, then test if the archetype predicts NEXT-year value beyond a baseline."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nfl_data_py as nfl
w=nfl.import_weekly_data(list(range(2016,2025))); w=w[w.season_type=="REG"].copy()
for c in ["carries","targets","receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds"]: w[c]=w.get(c,0).fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
rb=w[w.position=="RB"].groupby(["player_id","season"]).agg(g=("fp","size"),fp=("fp","sum"),
   car=("carries","sum"),tgt=("targets","sum"),rec=("receptions","sum")).reset_index()
rb=rb[rb.g>=8].copy(); rb["ppg"]=rb.fp/rb.g; rb["cpg"]=rb.car/rb.g; rb["tpg"]=rb.tgt/rb.g
rel=rb[rb.ppg>=10]
print("=== TREND CONFIRMATION (our data) ===")
print("  50+ catch RBs/yr:      "+"  ".join("%d:%d"%(s,((rb.season==s)&(rb.rec>=50)).sum()) for s in range(2016,2026) if s<2025 or (rb.season==s).any()))
print("  pass-specialists (<=8.5 cpg & >=10 ppg)/yr: "+"  ".join("%d:%d"%(s,((rel.season==s)&(rel.cpg<=8.5)).sum()) for s in range(2016,2025)))
# archetype: medium-volume (8.5-15 cpg) dual-threat (>=2.5 tpg)
rb["arch"]=np.where(rb.cpg>=15,"high",np.where(rb.cpg>=8.5,"med","low"))
med=rel[(rel.cpg>8.5)&(rel.cpg<=15)]; low=rel[rel.cpg<=8.5]
print("  medium-vol relevant/yr:"+"  ".join("%d:%d"%(s,((med.season==s)).sum()) for s in range(2016,2025)))
print("  medium-vol avg tpg early(16-20) %.1f vs recent(21-24) %.1f"%(med[med.season<=2020].tpg.mean(),med[med.season>=2021].tpg.mean()))
# PREDICTIVE: does year-Y archetype predict year Y+1 ppg beyond last-yr ppg?
nxt={(r.player_id,r.season):r.ppg for r in rb.itertuples()}; rb["next"]=[nxt.get((p,s+1),np.nan) for p,s in zip(rb.player_id,rb.season)]
d=rb.dropna(subset=["next"]).copy()
d["is_med_dt"]=((d.cpg>8.5)&(d.cpg<=15)&(d.tpg>=2.5)).astype(float)
d["is_spec"]=((d.cpg<=8.5)&(d.tpg>=3)).astype(float)
def partial(df,sig,extra=("ppg","cpg")):
    x=df.dropna(subset=["next",sig]+list(extra))
    X=np.column_stack([np.ones(len(x))]+[x[c].values for c in extra]+[x[sig].values]); y=x.next.values
    b,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@b
    se=np.sqrt((res@res/(len(x)-X.shape[1]))*np.linalg.pinv(X.T@X)[-1,-1]); return b[-1],b[-1]/se,len(x)
print("\n=== PREDICTIVE: next-yr PPG beyond [last-yr ppg + carries/game] ===")
for nm,sig in [("medium-vol dual-threat flag","is_med_dt"),("pass-specialist flag","is_spec"),("targets/game","tpg")]:
    c,t,n=partial(d,sig); print("  %-30s coef %+.2f  t %+.1f  (n=%d)"%(nm,c,t,n))
# recent vs early
for era,sub in [("2016-2020",d[d.season<=2020]),("2021-2024",d[d.season>=2021])]:
    c,t,n=partial(sub,"is_med_dt"); print("  med-dual-threat in %s: coef %+.2f t %+.1f n=%d"%(era,c,t,n))

# ============ RIGOR: does the archetype add over a RICH projection proxy, in recent years? ============
print("\n=== VALIDATE over rich proxy (walk-forward, RB only) ===")
# enrich panel with age + draft pick + prev2 ppg + team target share
pl2=nfl.import_players()[["gsis_id","birth_date","draft_pick"]]
age_ref=pd.Timestamp("2020-09-01")
pl2["dob"]=pd.to_datetime(pl2.birth_date,errors="coerce")
dp={g:(dn if pd.notna(dn) else 260) for g,dn in zip(pl2.gsis_id,pl2.draft_pick)}
dob={g:d for g,d in zip(pl2.gsis_id,pl2.dob)}
teamtgt=w[w.position.isin(["WR","RB","TE"])].groupby(["recent_team","season"]).targets.sum().to_dict()
rb["prev2"]=[nxt.get((p,s-1),np.nan) for p,s in zip(rb.player_id,rb.season)]  # ppg year-1 (reuse nxt map: (id,season)->ppg)
rb["age"]=[ (pd.Timestamp("%d-09-01"%s)-dob[p]).days/365.25 if dob.get(p)==dob.get(p) and dob.get(p) is not None and pd.notna(dob.get(p)) else np.nan for p,s in zip(rb.player_id,rb.season)]
rb["pick"]=[dp.get(p,260) for p in rb.player_id]
tt=w.groupby(["player_id","season"]).recent_team.agg(lambda s:s.mode().iloc[0] if len(s.mode()) else None).to_dict()
rb["tsh"]=[ rb.tgt.iloc[i]/teamtgt.get((tt.get((rb.player_id.iloc[i],rb.season.iloc[i])),rb.season.iloc[i]),np.nan) for i in range(len(rb))]
D=rb.dropna(subset=["next","ppg","cpg","tpg","age","prev2"]).copy()
D["prev2"]=D.prev2.fillna(D.ppg); D["tsh"]=D.tsh.fillna(D.tsh.median())
D["med_dt"]=((D.cpg>8.5)&(D.cpg<=15)&(D.tpg>=2.5)).astype(float)
D["spec"]=((D.cpg<=8.5)&(D.tpg>=3)).astype(float)
RICH=["ppg","prev2","cpg","tpg","tsh","age","pick"]
def wf(feats,test_years):
    err=[]
    for ty in test_years:
        tr=D[D.season<ty]; te=D[D.season==ty]
        if len(te)<5: continue
        X=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in feats]); y=tr.next.values
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        Xt=np.column_stack([np.ones(len(te))]+[te[c].values for c in feats])
        err+=list(np.abs(te.next.values-Xt@b))
    return np.mean(err),len(err)
for label,yrs in [("ALL test yrs 2019-2024",range(2019,2025)),("RECENT 2021-2024",range(2021,2025))]:
    b0,n=wf(RICH,yrs); b1,_=wf(RICH+["med_dt"],yrs); b2,_=wf(RICH+["spec"],yrs); b3,_=wf(RICH+["med_dt","spec"],yrs)
    print("  %s (n=%d): rich %.3f | +med_dt %.3f (%+.3f) | +spec %.3f (%+.3f) | +both %.3f (%+.3f)"%(
        label,n,b0,b1,b0-b1,b2,b0-b2,b3,b0-b3))
