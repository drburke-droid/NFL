"""Validate each theory signal's INCREMENTAL value over a rich projection proxy (mirrors what the
model already uses), walk-forward. If a signal adds ~0 over rich, the projection already accounts
for it; if it still helps, it's genuinely new. Same sample for rich vs rich+signal (fair)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sqlite3, os, nfl_data_py as nfl
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); YRS=list(range(2016,2025))
CAN={'GNB':'GB','KAN':'KC','LVR':'LV','OAK':'LV','NOR':'NO','NWE':'NE','SFO':'SF','TAM':'TB','SD':'LAC','STL':'LAR','LA':'LAR','WSH':'WAS','JAC':'JAX'}; can=lambda t:CAN.get(t,t)
w=nfl.import_weekly_data(YRS); w=w[w.season_type=="REG"].copy()
for c in ["targets","carries","receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds","attempts"]: w[c]=w.get(c,0).fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
w["td"]=w.receiving_tds+w.rushing_tds; w["yds"]=w.receiving_yards+w.rushing_yards; w["opp"]=w.targets+w.carries
g=w.groupby(["player_id","player_display_name","position","season"]).agg(fp=("fp","sum"),G=("fp","size"),tgt=("targets","sum"),
   yds=("yds","sum"),td=("td","sum"),opp=("opp","sum"),team=("recent_team",lambda s:s.mode().iat[0])).reset_index()
g=g[g.G>=8].copy(); g["ppg"]=g.fp/g.G
tt=w.groupby(["recent_team","season"]).targets.sum().rename("tt"); g=g.merge(tt,left_on=["team","season"],right_on=["recent_team","season"])
g["tshare"]=g.tgt/g.tt; g["ypg"]=g.yds/g.G; g["opg"]=g.opp/g.G; g["tdrate"]=g.td/g.opp.clip(lower=1); g["tm"]=g.team.map(can)
for pos in ["RB","WR","TE"]: g[f"is_{pos}"]=(g.position==pos).astype(float)
try: g=g.merge(nfl.import_seasonal_rosters(YRS)[["player_id","season","age"]].drop_duplicates(["player_id","season"]),on=["player_id","season"],how="left")
except Exception: g["age"]=np.nan
g["age"]=g.age.fillna(g.age.median()); g["age2"]=g.age**2
try:
    dp=nfl.import_draft_picks(list(range(2005,2025))); pkcol="pick" if "pick" in dp else "draft_ovr"
    pk=dp.dropna(subset=["gsis_id"]).groupby("gsis_id")[pkcol].min(); g["pick"]=g.player_id.map(pk).fillna(260)
except Exception: g["pick"]=260
g=g.sort_values("season")
lag={(r.player_id,r.season):r.ppg for r in g.itertuples()}; g["ppg_lag"]=[lag.get((p,s-1),np.nan) for p,s in zip(g.player_id,g.season)]
g["ppg_lag"]=g.ppg_lag.fillna(g.ppg)
nxt={(r.player_id,r.season):r.ppg for r in g.itertuples()}; g["next"]=[nxt.get((p,s+1),np.nan) for p,s in zip(g.player_id,g.season)]
# signals not in the rich model: Vegas Y+1 total, incoming QB, Y+1 pass rate
qb=w[w.position=="QB"].groupby(["recent_team","season","player_id"]).agg(fp=("fp","sum"),G=("fp","size"),att=("attempts","sum")).reset_index()
qbp=qb.sort_values("att").groupby(["recent_team","season"]).tail(1); qbppg={(can(r.recent_team),r.season):r.fp/max(r.G,1) for r in qbp.itertuples()}
tr=w.groupby(["recent_team","season"]).agg(a=("attempts","sum"),c=("carries","sum")).reset_index(); passrate={(can(r.recent_team),r.season):r.a/max(r.a+r.c,1) for r in tr.itertuples()}
c=sqlite3.connect(os.path.join(ROOT,"db","nfl_odds.db")); gl=pd.read_sql("select season,team,implied_team_total from nflv_game_lines",c); c.close()
gl["tm"]=gl.team.map(can); vtot=gl.groupby(["tm","season"]).implied_team_total.mean().to_dict()
nteam={(r.player_id,r.season):can(r.team) for r in g.itertuples()}; g["nteam"]=[nteam.get((p,s+1)) for p,s in zip(g.player_id,g.season)]
g["nvt"]=[vtot.get((nt,s+1)) for nt,s in zip(g.nteam,g.season)]
g["nqb"]=[qbppg.get((nt,s)) for nt,s in zip(g.nteam,g.season)]
g["npr"]=[passrate.get((nt,s+1)) for nt,s in zip(g.nteam,g.season)]

RICH=["ppg","ppg_lag","tshare","ypg","opg","age","age2","pick","G","is_RB","is_WR","is_TE"]
NAIVE=["ppg","age"]
def wf(P,feats):
    m=[]
    for T in range(2020,2025):
        tr=P[P.season<T]; te=P[P.season==T]
        if len(te)<15 or len(tr)<80: continue
        Xt=np.column_stack([np.ones(len(tr))]+[tr[f].values for f in feats]); b,*_=np.linalg.lstsq(Xt,tr.next.values,rcond=None)
        Xe=np.column_stack([np.ones(len(te))]+[te[f].values for f in feats]); m.append(np.abs(te.next.values-Xe@b).mean())
    return np.mean(m) if m else np.nan
base=g.dropna(subset=RICH+["next"])
print(f"Walk-forward MAE (n={len(base)}):  naive[ppg,age]={wf(base,NAIVE):.3f}   RICH(projection proxy)={wf(base,RICH):.3f}")
print(f"  -> the rich model (target share, yards, draft, age, durability...) is far better, i.e. those signals ARE used.\n")
print("Incremental MAE improvement of each signal OVER the rich projection proxy (same sample):")
for name,sig in [("Vegas: Y+1 implied team total","nvt"),("TD regression: TD/opportunity","tdrate"),
                 ("Scheme: Y+1 team pass rate","npr"),("QB quality: incoming QB prior ppg","nqb"),
                 ("(redundancy check) target share","tshare")]:
    s=g.dropna(subset=RICH+["next",sig])
    r0=wf(s,RICH); r1=wf(s,RICH+[sig]); v="ADDS (new)" if (r0-r1)>0.01 else ("~0 (already in)" if abs(r0-r1)<=0.01 else "hurts")
    print(f"  {name:36s} n={len(s):<4} rich {r0:.3f} -> +sig {r1:.3f}  ΔMAE {r0-r1:+.3f}  -> {v}")
