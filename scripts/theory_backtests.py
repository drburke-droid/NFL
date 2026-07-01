"""Batch backtest of fantasy theories vs a last-year-PPG + age baseline (leading-indicator test).
partial(): regress next_ppg ~ [ppg, age, signal]; report signal coef/t and whether it cuts MAE.
Verdict SIGNAL if |t|>2 AND MAE improves; else weak/none. PPR, >=8 games, 2018->2024."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nfl_data_py as nfl
YRS=list(range(2017,2025))
w=nfl.import_weekly_data(YRS); w=w[w.season_type=="REG"].copy()
for c in ["targets","carries","receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds"]: w[c]=w.get(c,0).fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
w["td"]=w.receiving_tds+w.rushing_tds; w["yds"]=w.receiving_yards+w.rushing_yards; w["opp"]=w.targets+w.carries
g=w.groupby(["player_id","position","season"]).agg(fp=("fp","sum"),G=("fp","size"),tgt=("targets","sum"),
   car=("carries","sum"),rec=("receptions","sum"),yds=("yds","sum"),td=("td","sum"),opp=("opp","sum"),
   team=("recent_team",lambda s:s.mode().iat[0])).reset_index()
g=g[g.G>=8].copy(); g["ppg"]=g.fp/g.G
tt=w.groupby(["recent_team","season"]).targets.sum().rename("tt"); g=g.merge(tt,left_on=["team","season"],right_on=["recent_team","season"])
g["tshare"]=g.tgt/g.tt; g["tdrate"]=g.td/g.opp.clip(lower=1); g["ypg"]=g.yds/g.G; g["opg"]=g.opp/g.G
try: g=g.merge(nfl.import_seasonal_rosters(YRS)[["player_id","season","age"]].drop_duplicates(["player_id","season"]),on=["player_id","season"],how="left")
except Exception: g["age"]=np.nan
g["age"]=g.age.fillna(g.age.median())
nxt={(r.player_id,r.season):(r.ppg,r.G) for r in g.itertuples()}
g["next"]=[nxt.get((p,s+1),(np.nan,np.nan))[0] for p,s in zip(g.player_id,g.season)]
g["nextG"]=[nxt.get((p,s+1),(np.nan,np.nan))[1] for p,s in zip(g.player_id,g.season)]

def partial(df,sig,extra=("ppg","age"),tgt="next"):
    d=df.dropna(subset=[tgt,sig]+list(extra)).copy()
    if len(d)<40 or d[sig].std()==0: return (np.nan,np.nan,len(d),np.nan)
    X=np.column_stack([np.ones(len(d))]+[d[c].values for c in extra]+[d[sig].values]); y=d[tgt].values
    beta,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@beta
    try: se=np.sqrt((res@res/(len(d)-X.shape[1]))*np.linalg.pinv(X.T@X)[-1,-1]); t=beta[-1]/se if se else np.nan
    except Exception: t=np.nan
    Xb=np.column_stack([np.ones(len(d))]+[d[c].values for c in extra]); bb,*_=np.linalg.lstsq(Xb,y,rcond=None)
    return (beta[-1],t,len(d),np.abs(y-Xb@bb).mean()-np.abs(res).mean())
def rpt(name,r,note=""):
    coef,t,n,dmae=r; v="SIGNAL" if (abs(t)>2 and dmae>0.01) else ("weak" if abs(t)>2 else "none")
    print(f"  {name:36s} n={n:<4} coef={coef:+.3f} t={t:+.1f} ΔMAE={dmae:+.3f} -> {v}  {note}")

SK=g.position.isin(['RB','WR','TE'])
print("=== Leading-indicator tests (beyond last-yr PPG + age) ===")
rpt("Volume: target share (WR/TE)",partial(g[g.position.isin(['WR','TE'])],"tshare"))
rpt("Volume: opportunities/game (RB)",partial(g[g.position=='RB'],"opg"))
rpt("Yards/game (skill)",partial(g[SK],"ypg"))
rpt("TD regression: TD/opportunity",partial(g[SK],"tdrate"),"neg=high-TD regress down")
# draft capital persistence (pedigree beyond last yr): lower pick = better -> negative coef on pick
try:
    dp=nfl.import_draft_picks(list(range(2010,2025)))[["gsis_id","draft_ovr" if "draft_ovr" in nfl.import_draft_picks([2020]).columns else "pick"]] if False else nfl.import_draft_picks(list(range(2010,2025)))
    pk=dp.dropna(subset=["gsis_id"]).groupby("gsis_id")["pick"].min() if "pick" in dp else dp.dropna(subset=["gsis_id"]).groupby("gsis_id")["draft_ovr"].min()
    g["pick"]=g.player_id.map(pk); rpt("Draft capital: overall pick (skill)",partial(g[SK],"pick"),"neg=pedigree helps")
except Exception as e: print("  draft capital ERR",str(e)[:40])
# injury/availability persistence: games this yr -> games next yr (beyond ppg)
rpt("Availability: games -> next-yr games",partial(g[SK],"G",extra=("age",),tgt="nextG"),"does durability persist?")
print("\n=== Age curve: mean next-yr PPG change by age (skill starters, ppg>=8) ===")
a=g[SK & (g.ppg>=8)].dropna(subset=["next","age"]).copy(); a["d"]=a.next-a.ppg
for pos in ["RB","WR","TE"]:
    s=a[a.position==pos]
    print("  "+pos+": "+"  ".join(f"{lo}-{lo+1}:{s[(s.age>=lo)&(s.age<lo+2)].d.mean():+.1f}" for lo in [23,25,27,29,31]))
print("\n=== 2nd-year WR leap: PPG change from rookie yr to year 2 ===")
firstyr=g.sort_values("season").groupby("player_id").head(1).set_index("player_id").season.to_dict()
g["exp"]=[s-firstyr.get(p,s) for p,s in zip(g.player_id,g.season)]
wr2=g[(g.position=="WR")&(g.exp==0)].dropna(subset=["next"])   # rookie season, has yr2
print(f"  rookie WRs (n={len(wr2)}): rookie {wr2.ppg.mean():.1f} -> yr2 {wr2.next.mean():.1f} ({wr2.next.mean()-wr2.ppg.mean():+.1f} PPG leap); "
      f"rookie tshare predicts yr2: t={partial(wr2,'tshare')[1]:+.1f}")
