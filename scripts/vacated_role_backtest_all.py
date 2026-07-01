"""Clean-vacancy vacated-role backtest for WR (target share) AND RB (touch share). nfl_data_py, PPR."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nfl_data_py as nfl
CAN={'GNB':'GB','KAN':'KC','LVR':'LV','OAK':'LV','NOR':'NO','NWE':'NE','SFO':'SF','TAM':'TB','SD':'LAC','STL':'LAR','LA':'LAR','WSH':'WAS','JAC':'JAX'}
can=lambda t: CAN.get(t,t)
w=nfl.import_weekly_data(list(range(2018,2025))); w=w[w.season_type=="REG"].copy()
for c in ["targets","carries","receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds"]: w[c]=w[c].fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
w["tm"]=w.recent_team.map(can); w["touch"]=w.carries+w.targets
tt=w.groupby(["tm","season"]).agg(ttgt=("targets","sum"),ttouch=("touch","sum"))
pj=w.groupby(["player_id","position","season"]).agg(tgt=("targets","sum"),touch=("touch","sum"),fp=("fp","sum"),
   g=("fp","size"),team=("tm",lambda s:s.mode().iat[0])).reset_index().merge(tt,left_on=["team","season"],right_on=["tm","season"])
pj["ppg"]=pj.fp/pj.g
dp=nfl.import_draft_picks(list(range(2019,2025)))
def draft_set(pos): return {(can(r.team),int(r.season)) for r in dp[(dp.position==pos)&(dp["round"]<=2)].itertuples() if pd.notna(r.team)}

def run(pos, share_col, denom, VAC, LO, HI):
    pj["sh"]=pj[share_col]/pj[denom]
    sub=pj[(pj.position==pos)&(pj.g>=6)]; bk={(r.player_id,r.season):r for r in sub.itertuples()}
    tp={}
    for r in sub.itertuples(): tp.setdefault((r.team,r.season),[]).append(r)
    de=draft_set(pos); clean,cont,ctrl=[],[],[]
    for (A,Y),ros in tp.items():
        if Y+1>2024: continue
        hold=[p for p in ros if bk.get((p.player_id,Y+1)) and bk[(p.player_id,Y+1)].team==A]
        dep=[p for p in ros if p.sh>=VAC and (bk.get((p.player_id,Y+1)) is None or bk[(p.player_id,Y+1)].team!=A)]
        band=[p for p in hold if LO<=p.sh<=HI]
        if dep and band:
            m=max(band,key=lambda p:p.sh); n=bk[(m.player_id,Y+1)]
            (cont if (A,Y+1) in de else clean).append({"y":m.ppg,"y1":n.ppg})
        elif not dep:
            for p in band: ctrl.append({"y":p.ppg,"y1":bk[(p.player_id,Y+1)].ppg})
    C,X,K=pd.DataFrame(clean),pd.DataFrame(cont),pd.DataFrame(ctrl)
    e=lambda d:(d.y1.mean()-d.y.mean())
    print(f"{pos}: CLEAN n={len(C)} {C.y.mean():.1f}->{C.y1.mean():.1f} ({e(C):+.1f})  CONTESTED n={len(X)} ({e(X):+.1f})  CONTROL n={len(K)} ({e(K):+.1f})  |  clean edge {e(C)-e(K):+.2f} PPG, beat {(C.y1>C.y).mean()*100:.0f}% vs {(K.y1>K.y).mean()*100:.0f}%")

run("WR","tgt","ttgt",0.12,0.06,0.18)
run("RB","touch","ttouch",0.10,0.04,0.14)
