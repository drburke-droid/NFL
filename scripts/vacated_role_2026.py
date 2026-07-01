"""
2026 clean-vacancy vacated-role inheritors (WR + RB) -> docs/vacated_role.js.

Validated (scripts/vacated_role_backtest_all.py): a returning WR2/3 (target share) or RB2 (touch
share) on a team that lost a big-usage player AND did NOT draft that position rd1-2 rebounds ~+1.4-1.7
PPG vs control; contested vacancies decline. Shrunk bump: WR +1.0, RB +0.8.

Emits (a) VACATED_ROLE — default flags using 2026 NFL teams from the roster release; and (b) VAC_DATA
— the raw ingredients (2025 usage, 2026 draft, params, canonical names) so the board can RE-DERIVE the
flags from the FFA file's team column once you upload it (true rosters override the roster release).
2025 usage from play-by-play; teams/draft read directly (pkg gates 2025/26).
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, re
import pandas as pd
import nfl_data_py as nfl
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REL="https://github.com/nflverse/nflverse-data/releases/download/"
CAN={'GNB':'GB','KAN':'KC','LVR':'LV','OAK':'LV','NOR':'NO','NWE':'NE','SFO':'SF','TAM':'TB','SD':'LAC','STL':'LAR','LA':'LAR','WSH':'WAS','JAC':'JAX'}
can=lambda t: CAN.get(t,t)
norm=lambda s: re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",re.sub(r"[^a-z ]","",str(s).lower())).replace("  "," ").strip()
PARAMS={"WR":{"vac":0.12,"lo":0.06,"hi":0.18,"bump":1.0},"RB":{"vac":0.10,"lo":0.04,"hi":0.14,"bump":0.8}}

# 2025 usage from play-by-play: WR target share, RB touch(=carry+target) share
pbp=pd.read_parquet(REL+"pbp/play_by_play_2025.parquet",
    columns=["posteam","pass_attempt","receiver_player_id","rush_attempt","rusher_player_id","season_type"])
pbp=pbp[pbp.season_type=="REG"].copy(); pbp["tm"]=pbp.posteam.map(can)
tgt=pbp[(pbp.pass_attempt==1)&pbp.receiver_player_id.notna()]
car=pbp[(pbp.rush_attempt==1)&pbp.rusher_player_id.notna()]
team_tgt=tgt.groupby("tm").size(); team_touch=team_tgt.add(car.groupby("tm").size(),fill_value=0)
ptg=tgt.groupby(["receiver_player_id","tm"]).size().rename("t").reset_index()
pca=car.groupby(["rusher_player_id","tm"]).size().rename("c").reset_index()
pl=nfl.import_players()[["gsis_id","display_name","position"]]
nm=dict(zip(pl.gsis_id,pl.display_name)); ps=dict(zip(pl.gsis_id,pl.position))
# player primary team + touches
touch={}
for r in ptg.itertuples(): touch.setdefault(r.receiver_player_id,{}).setdefault(r.tm,[0,0])[0]+=r.t
for r in pca.itertuples(): touch.setdefault(r.rusher_player_id,{}).setdefault(r.tm,[0,0])[1]+=r.c
# 2026 team (roster release) + draft rd1-2
r26=pd.read_parquet(REL+"rosters/roster_2026.parquet",columns=["team","gsis_id"]); tm26={g:can(t) for g,t in zip(r26.gsis_id,r26.team) if pd.notna(g)}
dp=pd.read_parquet(REL+"draft_picks/draft_picks.parquet")
drafted={pos:{can(r.team) for r in dp[(dp.season==2026)&(dp.position==pos)&(dp["round"]<=2)].itertuples() if pd.notna(r.team)} for pos in ("WR","RB")}
# canonical data.js names
P=json.loads(open(os.path.join(ROOT,"docs","data.js"),encoding="utf-8").read().split("const PLAYERS = ")[1].rsplit(";",1)[0])
canon={norm(p["name"]):p["name"] for p in P}

usage={}                          # canonical name -> {pos, tm25, sh}
for g,tmd in touch.items():
    pos=ps.get(g); disp=nm.get(g)
    if pos not in ("WR","RB") or not disp: continue
    A=max(tmd,key=lambda t: sum(tmd[t]))                       # primary 2025 team
    t,c=tmd[A]; sh=(t/team_tgt.get(A,1)) if pos=="WR" else ((t+c)/team_touch.get(A,1))
    cn=canon.get(norm(disp))
    if cn: usage[cn]={"pos":pos,"tm":A,"sh":round(sh,4)}
gsis_by_canon={ (canon.get(norm(nm.get(g))) ): g for g in touch if nm.get(g)}

def compute(team26):
    byteam={}
    for cn,u in usage.items(): byteam.setdefault(u["tm"],[]).append(cn)
    out={}
    for A,names in byteam.items():
        for pos,P_ in PARAMS.items():
            dep=[n for n in names if usage[n]["pos"]==pos and usage[n]["sh"]>=P_["vac"] and team26.get(n,usage[n]["tm"])!=A]
            if not dep or A in drafted[pos]: continue
            hold=[n for n in names if usage[n]["pos"]==pos and team26.get(n,usage[n]["tm"])==A and P_["lo"]<=usage[n]["sh"]<=P_["hi"]]
            if not hold: continue
            inh=max(hold,key=lambda n: usage[n]["sh"]); dg=max(dep,key=lambda n: usage[n]["sh"])
            out[inh]={"bump":P_["bump"],"pos":pos,"team":A,"vac_share":round(usage[dg]["sh"]*100),"departed":dg}
    return out

# default team map from the roster release (by canonical name)
roster_team={cn:tm26.get(g) for cn,g in gsis_by_canon.items() if cn and tm26.get(g)}
default=compute(roster_team)
print("2026 clean-vacancy inheritors (default, roster-release teams):")
for n,d in sorted(default.items(),key=lambda x:-x[1]["vac_share"]):
    print(f"   {d['pos']} {n:22s} {d['team']:4s} +{d['bump']}  inherits {d['vac_share']}% from {d['departed']}")
# --- 2026 Vegas environment bump per team (implied team total; validated +0.063 MAE over projection) ---
import sqlite3
VCOEF, VCAP = 0.15, 2.0        # PPG per point of implied total above/below league mean (shrunk); capped
VEGAS={}
try:
    c=sqlite3.connect(os.path.join(ROOT,"db","nfl_odds.db"))
    gl=pd.read_sql("select team,implied_team_total from nflv_game_lines where season=2026",c); c.close()
    gl["tm"]=gl.team.map(can); vt=gl.groupby("tm").implied_team_total.mean(); mean=vt.mean()
    VEGAS={t:round(max(-VCAP,min(VCAP,VCOEF*(v-mean))),2) for t,v in vt.items()}
    hi=sorted(VEGAS.items(),key=lambda x:-x[1]); print(f"\n2026 Vegas env bump (mean total {mean:.1f}, {VCOEF}/pt, cap ±{VCAP}):")
    for t,b in hi[:4]+hi[-4:]: print(f"   {t:4s} total {vt[t]:.1f}  bump {b:+.2f} PPG")
except Exception as e: print("  Vegas ERR",str(e)[:60])
# default player -> 2026 team (canonical name) for the Vegas lookup / vacated fallback
ROSTER={}
for g_,tval in tm26.items():
    disp=nm.get(g_); cn=canon.get(norm(disp)) if disp else None
    if cn: ROSTER[cn]=tval

VAC={"usage":usage,"drafted":{k:sorted(v) for k,v in drafted.items()},"params":PARAMS}
js=("// AUTO-GENERATED by scripts/vacated_role_2026.py — vacated-role + Vegas env (validated).\n"
    "const VACATED_ROLE = "+json.dumps(default,separators=(",",":"))+";\n"
    "const VAC_DATA = "+json.dumps(VAC,separators=(",",":"))+";  // ingredients for FFA-roster recompute\n"
    "const VEGAS_2026 = "+json.dumps(VEGAS,separators=(",",":"))+";  // per-team implied-total PPG bump\n"
    "const ROSTER_2026 = "+json.dumps(ROSTER,separators=(",",":"))+";  // default 2026 team by player\n")
for d in (os.path.join(ROOT,"docs"),os.path.join(ROOT,"outputs","draft_tool")):
    try: open(os.path.join(d,"vacated_role.js"),"w",encoding="utf-8").write(js)
    except OSError: pass
print(f"\nWrote docs/vacated_role.js (vacated {len(default)}; VAC_DATA usage {len(usage)}; Vegas teams {len(VEGAS)}; roster {len(ROSTER)})")
