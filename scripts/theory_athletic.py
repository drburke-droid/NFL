"""Tier-3: do athletic thresholds (RB Speed Score; WR NGS separation + 40) add over the projection?
Same rich-proxy walk-forward MAE test, split by experience (young players = where breakouts live)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nfl_data_py as nfl
YRS=list(range(2016,2025))
w=nfl.import_weekly_data(YRS); w=w[w.season_type=="REG"].copy()
for c in ["targets","carries","receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds"]: w[c]=w.get(c,0).fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
w["yds"]=w.receiving_yards+w.rushing_yards; w["opp"]=w.targets+w.carries
norm=lambda s: __import__("re").sub(r"[^a-z ]","",str(s).lower()).strip()
g=w.groupby(["player_id","player_display_name","position","season"]).agg(fp=("fp","sum"),G=("fp","size"),tgt=("targets","sum"),
   yds=("yds","sum"),opp=("opp","sum"),team=("recent_team",lambda s:s.mode().iat[0])).reset_index()
g=g[g.G>=8].copy(); g["ppg"]=g.fp/g.G
tt=w.groupby(["recent_team","season"]).targets.sum().rename("tt"); g=g.merge(tt,left_on=["team","season"],right_on=["recent_team","season"])
g["tshare"]=g.tgt/g.tt; g["ypg"]=g.yds/g.G; g["opg"]=g.opp/g.G
for pos in ["RB","WR","TE"]: g[f"is_{pos}"]=(g.position==pos).astype(float)
try: g=g.merge(nfl.import_seasonal_rosters(YRS)[["player_id","season","age"]].drop_duplicates(["player_id","season"]),on=["player_id","season"],how="left")
except Exception: g["age"]=np.nan
g["age"]=g.age.fillna(g.age.median()); g["age2"]=g.age**2
try:
    dp=nfl.import_draft_picks(list(range(2005,2025))); pk=dp.dropna(subset=["gsis_id"]).groupby("gsis_id")["pick" if "pick" in dp else "draft_ovr"].min(); g["pick"]=g.player_id.map(pk).fillna(260)
except Exception: g["pick"]=260
g=g.sort_values("season"); lag={(r.player_id,r.season):r.ppg for r in g.itertuples()}; g["ppg_lag"]=[lag.get((p,s-1),np.nan) for p,s in zip(g.player_id,g.season)]; g["ppg_lag"]=g.ppg_lag.fillna(g.ppg)
nxt={(r.player_id,r.season):r.ppg for r in g.itertuples()}; g["next"]=[nxt.get((p,s+1),np.nan) for p,s in zip(g.player_id,g.season)]
first=g.groupby("player_id").season.min().to_dict(); g["exp"]=[s-first.get(p,s) for p,s in zip(g.player_id,g.season)]
# athletic: RB Speed Score from combine
cb=nfl.import_combine_data(list(range(2008,2025))); cb=cb.dropna(subset=["wt","forty"]); cb["ss"]=cb.wt*200/cb.forty**4; cb["nm"]=cb.player_name.map(norm)
ss=cb.groupby("nm").ss.max(); frt=cb.groupby("nm").forty.min()
g["nm"]=g.player_display_name.map(norm); g["ss"]=g.nm.map(ss); g["forty"]=g.nm.map(frt)
# WR NGS separation per season
try:
    ng=nfl.import_ngs_data("receiving",YRS); ng=ng[ng.week==0] if (ng.week==0).any() else ng
    sep=ng.groupby(["player_gsis_id" if "player_gsis_id" in ng else "player_display_name","season"]).avg_separation.mean()
    key="player_gsis_id" if "player_gsis_id" in ng else None
    if key: g["sep"]=[sep.get((pid,s),np.nan) for pid,s in zip(g.player_id,g.season)]
    else:
        sepn=ng.groupby([ng.player_display_name.map(norm),"season"]).avg_separation.mean(); g["sep"]=[sepn.get((n,s),np.nan) for n,s in zip(g.nm,g.season)]
except Exception as e: print("NGS err",str(e)[:40]); g["sep"]=np.nan
RICH=["ppg","ppg_lag","tshare","ypg","opg","age","age2","pick","G","is_RB","is_WR","is_TE"]
def wf(P,feats):
    m=[]
    for T in range(2019,2025):
        tr=P[P.season<T].dropna(subset=feats+["next"]); te=P[P.season==T].dropna(subset=feats+["next"])
        if len(te)<12 or len(tr)<60: continue
        Xt=np.column_stack([np.ones(len(tr))]+[tr[f].values for f in feats]); b,*_=np.linalg.lstsq(Xt,tr.next.values,rcond=None)
        Xe=np.column_stack([np.ones(len(te))]+[te[f].values for f in feats]); m.append(np.abs(te.next.values-Xe@b).mean())
    return np.mean(m) if m else np.nan
def test(name,df,sig):
    s=df.dropna(subset=RICH+["next",sig])
    if len(s)<60: print(f"  {name:42s} n={len(s)} too small"); return
    r0=wf(s,RICH); r1=wf(s,RICH+[sig]); v="ADDS" if (r0-r1)>0.01 else ("~0 (already in)" if abs(r0-r1)<=0.01 else "hurts")
    print(f"  {name:42s} n={len(s):<4} rich {r0:.3f}->+sig {r1:.3f}  ΔMAE {r0-r1:+.3f}  -> {v}")
print("Athletic thresholds — incremental MAE over the rich projection proxy:")
test("RB Speed Score (all RB)",g[g.position=='RB'],"ss")
test("RB Speed Score (young RB, exp<=2)",g[(g.position=='RB')&(g.exp<=2)],"ss")
test("WR 40 time (young WR, exp<=2)",g[(g.position=='WR')&(g.exp<=2)],"forty")
test("WR NGS avg separation (all WR)",g[g.position=='WR'],"sep")
test("WR NGS avg separation (young WR)",g[(g.position=='WR')&(g.exp<=2)],"sep")
