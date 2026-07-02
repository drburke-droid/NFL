"""2026 auction draft plan: assume every team keeps its predicted top-3 (+value) keepers, remove them
from the pool, recompute post-keeper expected prices from the league's positional bid curves, layer
owner tendencies + budgets, and emit a conditional target plan for MY team."""
import os, io, sys, json, contextlib, importlib.util
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location("pk",os.path.join(ROOT,"scripts","predict_keepers.py"))
pk=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(pk)
P,VAL,PMAP,norm,pidof=pk.P,pk.VAL,pk.PMAP,pk.norm,pk.pidof
CURVE=pk.bid_curve_pos; L=pk.L; MY=L.get("myTeamId")
teams=pk.teams_out
# --- 1) lock in predicted keepers ---
# FORCED: user-confirmed keeps the engine missed (it called Bowers 'redraft' on a $2 technicality —
# cost $19 vs exp $17 — but a human keeps him; his team had only 2 predicted keeps anyway).
FORCED={"Brock Bowers"}
kept=set(); tinfo={}
for t in teams:
    keeps=[c for c in t["candidates"] if c.get("predicted")]
    for c in t["candidates"]:
        if c["name"] in FORCED and not any(k["pid"]==c["pid"] for k in keeps):
            keeps=sorted(keeps+[c],key=lambda x:-x["savings"])[:3]
    spend=sum(c["cost"] for c in keeps)
    tinfo[t["id"]]={"name":t["name"],"keeps":keeps,"budget":200-spend,"tend":t["tend"]}
    kept.update(c["pid"] for c in keeps)
tot_keep=sum(200-v["budget"] for v in tinfo.values())
print(f"League: {L.get('name')} — {len(kept)} predicted keepers locked, ${tot_keep} of ${12*200} spent on keepers")
print(f"Post-keeper money in room: ${12*200-tot_keep}\n")
# --- 2) post-keeper pool + expected price (re-rank remaining players on positional curve) ---
pool={}
for pos in ("QB","RB","WR","TE"):
    arr=[p for p in P if p["position"]==pos and not pk.isK(p) and pidof(p) not in kept]
    arr.sort(key=lambda p:-(VAL.get((norm(p["name"]),pos)) or 0))
    C=CURVE.get(pos,[])
    rows=[]
    for r,p in enumerate(arr):
        v=VAL.get((norm(p["name"]),pos)) or 0
        e=(C[r] if r<len(C) else 1) if v>1 else 1
        rows.append({"name":p["name"],"team":p.get("team") or "","v":v,"exp":e,"edge":v-e})
    pool[pos]=rows
# --- 2b) KEEPER INFLATION: scale curve $ so the priced pool clears the room's actual money ---
n_slots=12*16-len(kept); room=12*200-tot_keep
exps=sorted([r["exp"] for pos in pool for r in pool[pos]],reverse=True)
exps=(exps+[1]*60)[:n_slots]                      # pad tail with $1 (K/DST/deep bench)
infl=max(0.85,min(1.8,room/sum(exps)))
for pos in pool:
    for r in pool[pos]:
        r["exp"]=max(1,round(1+(r["exp"]-1)*infl)); r["v"]=max(1,round(r["v"]*infl)); r["edge"]=r["v"]-r["exp"]
print(f"KEEPER INFLATION: raw curve sums ${sum(exps)} over {n_slots} slots vs ${room} in room -> x{infl:.2f}")
print("(values also restated in room-dollars so edges stay comparable)\n")
# --- 3) my team ---
me=tinfo[MY]
print(f"MY TEAM [{MY}] {me['name']} — budget after keepers: ${me['budget']}")
for c in me["keeps"]: print(f"  KEEP {c['pos']} {c['name']} @${c['cost']} (value ${c['value']})")
slots=L.get("roster_slots") or {}
print(f"  roster slots: {slots}")
# --- 4) rival bid pressure per position (tendency x budget) ---
print("\nRIVAL PRESSURE (tend mult >1.15 and budget rank):")
for pos in ("QB","RB","WR","TE"):
    riv=sorted([(v["tend"].get(pos,1),v["budget"],v["name"]) for k,v in tinfo.items() if k!=MY],reverse=True)
    hot=[f"{n} (x{t}, ${b})" for t,b,n in riv if t>=1.15][:4]
    print(f"  {pos}: "+("; ".join(hot) if hot else "none"))
flags=[(v["name"],v["tend"]) for k,v in tinfo.items() if k!=MY]
print("  eliteQB buyers: "+", ".join(n for n,t in flags if t.get("eliteQB")))
print("  payTE buyers:   "+", ".join(n for n,t in flags if t.get("payTE")))
print("  biggest-splash owners (top single bid): "+", ".join(f"{n} ${t['top']}" for n,t in sorted(flags,key=lambda x:-x[1]['top'])[:4]))
# --- 5) targets: best edge by tier per position ---
print("\nPOST-KEEPER POOL — top of each position (value / exp$ / edge):")
for pos in ("QB","RB","WR","TE"):
    print(f"  {pos}:")
    for r in pool[pos][:12]:
        print(f"    {r['name'][:24]:24s} {r['team']:4s} val ${r['v']:>3}  exp ${r['exp']:>3}  edge {r['edge']:+3d}")
print("\nBEST EDGES overall (val>=10):")
alle=[r|{"pos":pos} for pos in pool for r in pool[pos] if r["v"]>=10]
for r in sorted(alle,key=lambda x:-x["edge"])[:15]:
    print(f"  {r['pos']:3s} {r['name'][:24]:24s} val ${r['v']:>3} exp ${r['exp']:>3} edge {r['edge']:+3d}")

print("\nWR ranks 13-26 (my WR2/flex tier) & RB ranks 13-22:")
for pos,a,b in [("WR",12,26),("RB",12,22)]:
    for r in pool[pos][a:b]:
        print(f"  {pos} {r['name'][:24]:24s} {r['team']:4s} val ${r['v']:>3}  exp ${r['exp']:>3}  edge {r['edge']:+3d}")
