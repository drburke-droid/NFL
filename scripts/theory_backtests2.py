"""Athletic (combine), Vegas team total (odds DB), QB quality, scheme/coaching."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sqlite3, os, nfl_data_py as nfl
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YRS=list(range(2017,2025))
w=nfl.import_weekly_data(YRS); w=w[w.season_type=="REG"].copy()
for c in ["receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds","targets","carries"]: w[c]=w.get(c,0).fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
g=w.groupby(["player_id","player_display_name","position","season"]).agg(fp=("fp","sum"),G=("fp","size"),
   team=("recent_team",lambda s:s.mode().iat[0])).reset_index(); g=g[g.G>=8].copy(); g["ppg"]=g.fp/g.G
norm=lambda s: __import__("re").sub(r"[^a-z ]","",str(s).lower()).strip()
nxt={(r.player_id,r.season):r.ppg for r in g.itertuples()}; g["next"]=[nxt.get((p,s+1),np.nan) for p,s in zip(g.player_id,g.season)]

print("=== Athletic thresholds (combine) ===")
try:
    cb=nfl.import_combine_data(list(range(2010,2025)))
    fcol=[c for c in cb.columns if c in ("forty","forty_yd","x40yd")]; fcol=fcol[0] if fcol else None
    cb=cb.dropna(subset=["wt",fcol]) if fcol else cb
    cb["ss"]=(cb.wt*200)/(cb[fcol]**4) if fcol else np.nan     # RB Speed Score
    cb["nm"]=cb.player_name.map(norm); ss=dict(zip(cb.nm,cb.ss))
    g["ss"]=g.player_display_name.map(norm).map(ss)
    rb=g[(g.position=="RB")].dropna(subset=["next","ss"])
    hi=rb[rb.ss>=rb.ss.median()]; lo=rb[rb.ss<rb.ss.median()]
    print(f"  RB Speed Score (n={len(rb)}): high-SS next-yr {hi.next.mean():.1f} vs low-SS {lo.next.mean():.1f}  (Δ {hi.next.mean()-lo.next.mean():+.1f})")
except Exception as e: print("  combine ERR",str(e)[:60])

print("\n=== Vegas implied team total -> production (odds DB) ===")
try:
    c=sqlite3.connect(os.path.join(ROOT,"db","nfl_odds.db"))
    gl=pd.read_sql("select * from nflv_game_lines limit 3",c)
    print("  nflv_game_lines cols:",list(gl.columns)[:14])
    # try to build team-season implied total
    q="select * from nflv_game_lines"
    full=pd.read_sql(q,c)
    tc=[x for x in full.columns if 'team' in x.lower()]; toc=[x for x in full.columns if 'total' in x.lower() or 'implied' in x.lower()]
    print("  team cols:",tc," total/implied cols:",toc," seasons:",sorted(full.season.unique()) if 'season' in full else '?')
    c.close()
except Exception as e: print("  odds DB ERR",str(e)[:80])

print("\n=== Vegas / QB / scheme (leading indicators for next-yr PPG beyond last-yr) ===")
CAN={'GNB':'GB','KAN':'KC','LVR':'LV','OAK':'LV','NOR':'NO','NWE':'NE','SFO':'SF','TAM':'TB','SD':'LAC','STL':'LAR','LA':'LAR','WSH':'WAS','JAC':'JAX'}
can=lambda t: CAN.get(t,t)
w["pass"]=(w.get("attempts",0).fillna(0)>0).astype(int)  # crude; use attempts for pass rate below
qb=w[w.position=="QB"].groupby(["recent_team","season","player_id"]).agg(fp=("fp","sum"),G=("fp","size"),att=("attempts",lambda s:s.fillna(0).sum())).reset_index()
qbp=qb.sort_values("att").groupby(["recent_team","season"]).tail(1); qbp["qppg"]=qbp.fp/qbp.G.clip(lower=1)
qbppg={(can(r.recent_team),r.season):r.qppg for r in qbp.itertuples()}
# team pass rate
tr=w.groupby(["recent_team","season"]).agg(a=("attempts",lambda s:s.fillna(0).sum()),c=("carries",lambda s:s.fillna(0).sum())).reset_index()
tr["pr"]=tr.a/(tr.a+tr.c).clip(lower=1); passrate={(can(r.recent_team),r.season):r.pr for r in tr.itertuples()}
# vegas implied team total per team-season
c=sqlite3.connect(os.path.join(ROOT,"db","nfl_odds.db"))
gl=pd.read_sql("select season,team,game_type,implied_team_total from nflv_game_lines",c); c.close()
gl=gl[gl.game_type.isin(["REG","reg","regular"]) | gl.game_type.isna()]
gl["tm"]=gl.team.map(can); vt=gl.groupby(["tm","season"]).implied_team_total.mean(); vtot={(t,s):v for (t,s),v in vt.items()}
# next-year team for each player
g["tm"]=g.team.map(can); nextteam={(r.player_id,r.season):can(r.team) for r in g.itertuples()}
g["nteam"]=[nextteam.get((p,s+1)) for p,s in zip(g.player_id,g.season)]
g["nvt"]=[vtot.get((nt,s+1)) for nt,s in zip(g.nteam,g.season)]          # Y+1 team's implied total (preseason-knowable)
g["nqb"]=[qbppg.get((nt,s)) for nt,s in zip(g.nteam,g.season)]           # incoming QB's prior-yr ppg
g["npr"]=[passrate.get((nt,s+1)) for nt,s in zip(g.nteam,g.season)]      # Y+1 team pass rate
def part(df,sig):
    d=df.dropna(subset=["next","ppg",sig]); 
    if len(d)<40 or d[sig].std()==0: return (np.nan,np.nan,len(d),np.nan)
    X=np.column_stack([np.ones(len(d)),d.ppg.values,d[sig].values]); y=d.next.values
    b,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@b
    se=np.sqrt((res@res/(len(d)-3))*np.linalg.pinv(X.T@X)[-1,-1]); t=b[-1]/se if se else np.nan
    Xb=np.column_stack([np.ones(len(d)),d.ppg.values]); bb,*_=np.linalg.lstsq(Xb,y,rcond=None)
    return (b[-1],t,len(d),np.abs(y-Xb@bb).mean()-np.abs(res).mean())
def rp(n,r,note=""):
    coef,t,nn,dm=r; v="SIGNAL" if (abs(t)>2 and dm>0.01) else ("weak" if abs(t)>2 else "none")
    print(f"  {n:34s} n={nn:<4} coef={coef:+.3f} t={t:+.1f} ΔMAE={dm:+.3f} -> {v}  {note}")
SK=g.position.isin(['RB','WR','TE'])
rp("Vegas: Y+1 implied team total (skill)",part(g[SK],"nvt"))
rp("QB quality: incoming QB prior ppg (WR/TE)",part(g[g.position.isin(['WR','TE'])],"nqb"))
rp("Scheme: Y+1 team pass rate (WR/TE)",part(g[g.position.isin(['WR','TE'])],"npr"))
