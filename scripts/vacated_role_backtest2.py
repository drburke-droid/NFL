"""Smarter vacated-role predictor: does conditioning on a CLEAN vacancy (team didn't spend early
draft capital on a competing WR) sharpen the ex-ante inheritor signal? nfl_data_py, PPR, 2018-24."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nfl_data_py as nfl
YEARS=list(range(2018,2025))
CAN={'GNB':'GB','KAN':'KC','LVR':'LV','OAK':'LV','NOR':'NO','NWE':'NE','SFO':'SF','TAM':'TB','SD':'LAC','STL':'LAR','LA':'LAR','WSH':'WAS','JAC':'JAX'}
can=lambda t: CAN.get(t,t)
w=nfl.import_weekly_data(YEARS); w=w[w.season_type=="REG"].copy()
for c in ["targets","receptions","receiving_yards","receiving_tds","rushing_yards","rushing_tds"]: w[c]=w[c].fillna(0)
w["fp"]=w.receptions+w.receiving_yards*.1+w.receiving_tds*6+w.rushing_yards*.1+w.rushing_tds*6
w["tm"]=w.recent_team.map(can)
tt=w.groupby(["tm","season"]).targets.sum().rename("tt")
pj=w.groupby(["player_id","position","season"]).agg(tgt=("targets","sum"),fp=("fp","sum"),g=("fp","size"),
   team=("tm",lambda s:s.mode().iat[0])).reset_index().merge(tt,left_on=["team","season"],right_on=["tm","season"])
pj["ts"]=pj.tgt/pj.tt; pj["ppg"]=pj.fp/pj.g
wr=pj[(pj.position=="WR")&(pj.g>=6)]; bk={(r.player_id,r.season):r for r in wr.itertuples()}
twr={}
for r in wr.itertuples(): twr.setdefault((r.team,r.season),[]).append(r)
dp=nfl.import_draft_picks(list(range(2019,2025)))
draft_early={(can(r.team),int(r.season)) for r in dp[(dp.position=="WR")&(dp["round"]<=2)].itertuples() if pd.notna(r.team)}
LO,HI=.06,.18
clean,contested,ctrl=[],[],[]
for (A,Y),roster in twr.items():
    if Y+1>2024: continue
    hold=[p for p in roster if bk.get((p.player_id,Y+1)) and bk[(p.player_id,Y+1)].team==A]
    dep=[p for p in roster if p.ts>=.12 and (bk.get((p.player_id,Y+1)) is None or bk[(p.player_id,Y+1)].team!=A)]
    band=[p for p in hold if LO<=p.ts<=HI]
    if dep and band:
        m=max(band,key=lambda p:p.ts); n=bk[(m.player_id,Y+1)]
        rec={"ppgY":m.ppg,"ppgY1":n.ppg,"dts":n.ts-m.ts}
        (contested if (A,Y+1) in draft_early else clean).append(rec)
    elif not dep:
        for p in band: n=bk[(p.player_id,Y+1)]; ctrl.append({"ppgY":p.ppg,"ppgY1":n.ppg,"dts":n.ts-p.ts})
def L(lbl,d):
    d=pd.DataFrame(d); print(f"  {lbl:38s} n={len(d):<4} ts {d.dts.mean()*100:+.1f}pp  PPG {d.ppgY.mean():.1f}->{d.ppgY1.mean():.1f} ({d.ppgY1.mean()-d.ppgY.mean():+.1f})  beat {(d.ppgY1>d.ppgY).mean()*100:.0f}%"); return d
print("Smarter vacated-role: inheritor split by whether team drafted a WR in rd1-2 the next spring\n")
C=L("CLEAN vacancy (no early WR drafted)",clean); X=L("CONTESTED (team drafted WR rd1-2)",contested); K=L("control (role-matched, no vacancy)",ctrl)
print(f"\n  CLEAN edge vs control: {(C.ppgY1.mean()-C.ppgY.mean())-(K.ppgY1.mean()-K.ppgY.mean()):+.2f} PPG")
up=(C.ppgY1.mean()-C.ppgY.mean())-(K.ppgY1.mean()-K.ppgY.mean())
print(f"  MAE clean inheritor Y+1 from last-yr PPG: {(C.ppgY1-C.ppgY).abs().mean():.2f}   from +{max(up,0):.1f} uplift: {(C.ppgY1-(C.ppgY+max(up,0))).abs().mean():.2f}")
