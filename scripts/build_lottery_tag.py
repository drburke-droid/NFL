"""Emit docs/lottery_2026.js — 'Lottery ticket' tags for the draft board.
Model: scripts/lottery_tickets.py (validated: AUC .789, top-10/season = 8% star = 6.7x base).
Tags players in the $1-3 expected-price pool with p(star) >= 2% (~1.5x+ base). Young players
(age <= 26) get the bright tag per user preference; 27+ emitted as muted 'vet' variant.
RERUN after FFA 2026 uploads (refresh_2026 chain) — ffa_aav is a model feature and the cheap
pool shifts once the market prices players."""
import os, sqlite3, warnings, sys, io, contextlib, json, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(name):
    s=importlib.util.spec_from_file_location(name,os.path.join(ROOT,"scripts",name+".py"))
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
POS=["QB","RB","WR","TE"]; STAR={"QB":6,"RB":12,"WR":12,"TE":6}
MS=load("model_season"); DB=MS.DB
con=sqlite3.connect(DB)
df=pd.read_sql("SELECT * FROM season_dataset",con)
ffa=pd.read_sql("SELECT season,player_id,ffa_aav FROM nflv_ffa_proj WHERE player_id IS NOT NULL",con).drop_duplicates(["season","player_id"])
ht=pd.read_sql("SELECT * FROM nflv_half_trend",con)
opp=pd.read_sql("SELECT player_id,season,vac_rb_carries,inc_rb_carries,vac_pc_targets,inc_pc_targets FROM nflv_opportunity",con)
xf=pd.read_sql("""SELECT player_id, season, SUM(total_fantasy_points_exp) xfp, SUM(total_fantasy_points_diff) xgap,
                  COUNT(DISTINCT week) wk FROM nflv_ff_opp GROUP BY player_id, season""",con)
con.close()
xf["xfp_pg"]=xf.xfp/xf.wk; xf["xgap_pg"]=xf.xgap/xf.wk; xf["season"]=xf.season.astype(int)+1
df=df.merge(ffa,on=["season","player_id"],how="left").merge(ht,on=["player_id","season"],how="left") \
     .merge(opp,on=["player_id","season"],how="left").merge(xf[["player_id","season","xfp_pg","xgap_pg"]],on=["player_id","season"],how="left")
df["cheap"]=((df.ffa_aav<=3)|df.ffa_aav.isna()).astype(int)
df["star"]=(df.next_pos_finish<=df.position.map(STAR)).astype(int)
d=df[(df.cheap==1)&df.next_pos_finish.notna()&(df.season>=2014)].copy()
FE=["age","years_exp","draft_pick","draft_round","prior_ppg","prior2_ppg","prior_games","prior_off_pct",
    "prior_target_share","prior_air_yards_share","prior_wopr","prior_targets_pg","prior_carries_pg",
    "prior_cv","prior_receiving_epa","team_change","weight","forty","ffa_aav",
    "ht_d_snap","ht_h2_snap","ht_d_tgtsh","ht_h2_tgtsh","ht_d_ppg","ht_h2_ppg","ht_slope",
    "vac_pc_targets","vac_rb_carries","inc_pc_targets","inc_rb_carries","xfp_pg","xgap_pg"]
FE=[f for f in FE if f in d.columns]
d["pos_id"]=d.position.map({p:i for i,p in enumerate(POS)}); FE+=["pos_id"]
GB=dict(objective="binary",n_estimators=250,learning_rate=0.04,num_leaves=15,min_child_samples=25,
        subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
mf=lgb.LGBMClassifier(**GB).fit(d[FE].astype(float).fillna(-1),d.star)
# ---- 2026 frame (same recipe as projection_overhaul) ----
bsd=load("build_season_dataset")
con=sqlite3.connect(DB); sf=bsd.build_season_frame(con)
r26=pd.read_sql("SELECT player_id, team AS t26 FROM nflv_rosters_2026",con).drop_duplicates("player_id"); con.close()
prior=sf[sf.season==2025][bsd.FEAT_COLS].copy()
prior.columns=["player_id","player_display_name","position","prior_season","prior_team"]+["prior_"+c for c in bsd.FEAT_COLS[5:]]
prior["season"]=2026
p2=sf[sf.season==2024][["player_id","ppg"]].rename(columns={"ppg":"prior2_ppg"})
ctx=sf[sf.season==2025][["player_id","age","years_exp","weight","forty","draft_round","draft_pick"]].copy()
ctx["age"]+=1; ctx["years_exp"]+=1
v26=prior.merge(p2,on="player_id",how="left").merge(ctx,on="player_id",how="left").merge(r26,on="player_id",how="left")
v26["team_change"]=(v26.t26.notna()&v26.prior_team.notna()&(v26.t26!=v26.prior_team)).astype(int)
v26=v26.merge(ht,on=["player_id","season"],how="left").merge(opp,on=["player_id","season"],how="left") \
       .merge(xf[["player_id","season","xfp_pg","xgap_pg"]],on=["player_id","season"],how="left")
v26=v26.merge(ffa[ffa.season==2026],on=["season","player_id"],how="left")   # 2026 FFA AAV when uploaded
v26["pos_id"]=v26.position.map({p:i for i,p in enumerate(POS)})
for f in FE:
    if f not in v26.columns: v26[f]=np.nan
v26=v26[v26.position.isin(POS)&(v26.prior_games.fillna(0)>=1)].copy()
v26["p_star"]=mf.predict_proba(v26[FE].astype(float).fillna(-1))[:,1]
# ---- board pool: exp price from the league bid-curve engine; canonical board names ----
with contextlib.redirect_stdout(io.StringIO()): pk=load("predict_keepers")
nrm=lambda s: re.sub(r"\s+"," ",re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",re.sub(r"[^a-z ]","",str(s).lower()))).strip()
exp_by,canon={},{}
for pl in pk.P:
    k=(nrm(pl["name"]),pl["position"])
    exp_by[k]=pk.exp_price.get(pk.pidof(pl),1); canon[k]=pl["name"]
v26["k"]=[(nrm(n),p) for n,p in zip(v26.player_display_name,v26.position)]
v26["expd"]=v26.k.map(exp_by); v26["board"]=v26.k.map(canon)
cheap=v26[v26.board.notna()&(v26.expd<=3)&(v26.p_star>=0.02)].sort_values("p_star",ascending=False).head(30)
def why(r):
    w=[]
    if (r.prior_target_share or 0)>=0.12: w.append(f"{r.prior_target_share:.0%} tgt share")
    if (r.ht_h2_ppg or 0)>=8: w.append(f"{r.ht_h2_ppg:.0f} PPG 2nd half")
    if (r.prior_off_pct or 0)>=0.5: w.append(f"{r.prior_off_pct:.0%} snaps")
    if (r.draft_pick or 300)<=75: w.append(f"pick {int(r.draft_pick)}")
    if (r.xgap_pg or 0)>=1.0: w.append("xFP over-producer")
    if r.team_change==1: w.append("moved (negative)")
    return "; ".join(w) or "model composite"
out={}
for r in cheap.itertuples():
    out[r.board]={"p":round(float(r.p_star),3),"pos":r.position,"age":int(r.age) if pd.notna(r.age) else None,
                  "young":bool(pd.notna(r.age) and r.age<=26),"why":why(r)}
young=sum(1 for v in out.values() if v["young"])
print(f"Lottery tags: {len(out)} ({young} young <=26, {len(out)-young} vet)")
for n,v in sorted(out.items(),key=lambda x:-x[1]["p"])[:12]:
    print(f"  {'YOUNG' if v['young'] else 'vet  '} {v['pos']:3s} {n:24s} p={v['p']:.1%}  {v['why']}")
js=("// AUTO-GENERATED by scripts/build_lottery_tag.py — $1-3 lottery tickets (validated 6.7x screen,\n"
    "// scripts/lottery_tickets.py / outputs/reports/lottery_tickets.md). Rerun after FFA 2026 uploads.\n"
    "const LOTTERY_2026 = "+json.dumps(out,separators=(",",":"))+";\n")
for dd in (os.path.join(ROOT,"docs"),os.path.join(ROOT,"outputs","draft_tool")):
    try: open(os.path.join(dd,"lottery_2026.js"),"w",encoding="utf-8").write(js)
    except OSError: pass
print("Wrote docs/lottery_2026.js")
