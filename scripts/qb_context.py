"""QB-context-independent WR evaluation.
A) 2025 facts: team QB quality ranks; JJ/Chase role + conversion; MIN passes to-JJ vs others;
   Tyreek/Tua 2024 precedent.
B) HISTORY: do WRs whose down year coincided with a team-QB collapse rebound NEXT year above our
   actual model's projection? (residual test on season_predictions)"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL="https://github.com/nflverse/nflverse-data/releases/download/"
DB=os.path.join(ROOT,"db","nfl_odds.db")
con=sqlite3.connect(DB)
# ---------- team QB quality by season (from QB rows of nflv_season) ----------
qb=pd.read_sql("""SELECT season,recent_team team,SUM(attempts) att,
    SUM(passing_yards) py,SUM(passing_interceptions) ints,SUM(passing_epa) epa,
    AVG(passing_cpoe) cpoe FROM nflv_season WHERE position='QB' AND attempts>=100
    GROUP BY season,recent_team""",con)
qb["ypa"]=qb.py/qb.att; qb["int_rate"]=qb.ints/qb.att
qb["epa_att"]=qb.epa/qb.att
for c in ["cpoe","ypa","epa_att"]:
    qb[c+"_rk"]=qb.groupby("season")[c].rank(ascending=False)
m25=qb[(qb.season==2025)&(qb.team=="MIN")]
c25=qb[(qb.season==2025)&(qb.team=="CIN")]
print("=== A) 2025 TEAM QB QUALITY (rank of 32) ===")
for lbl,r in [("MIN",m25),("CIN",c25)]:
    r=r.iloc[0]
    print(f"  {lbl}: CPOE {r.cpoe:+.1f} (rk {r.cpoe_rk:.0f})  YPA {r.ypa:.1f} (rk {r.ypa_rk:.0f})  EPA/att {r.epa_att:+.3f} (rk {r.epa_att_rk:.0f})  INT% {r.int_rate:.1%}")
# JJ role + conversion (xFP)
w=pd.read_sql("""SELECT player_display_name nm,season,SUM(targets) tg,COUNT(*) g,
    SUM(receiving_yards) ry,SUM(air_yards_share*0)+AVG(air_yards_share) ays, AVG(target_share) tsh
    FROM nflv_weekly WHERE player_display_name IN ('Justin Jefferson','Ja''Marr Chase','Tyreek Hill')
    AND season IN (2023,2024,2025) GROUP BY nm,season""",con)
xf=pd.read_sql("""SELECT f.player_id,f.season,SUM(f.total_fantasy_points) act,SUM(f.total_fantasy_points_exp) exp
    FROM nflv_ff_opp f JOIN (SELECT DISTINCT player_id,player_display_name FROM nflv_season) n USING(player_id)
    WHERE n.player_display_name IN ('Justin Jefferson','Ja''Marr Chase') AND f.season IN ('2024','2025')
    GROUP BY f.player_id,f.season""",con)
print("\n  Role (target share / air-yards share) & conversion:")
for r in w.sort_values(["nm","season"]).itertuples():
    print(f"    {r.nm:18s} {r.season}: tgt/g {r.tg/r.g:4.1f}  tgt-share {r.tsh:.0%}  air-yds share {r.ays:.0%}")
print("  xFP (opportunity) vs actual:")
for r in xf.itertuples(): print(f"    {r.player_id} {r.season}: exp {r.exp:.0f}  act {r.act:.0f}  gap {r.act-r.exp:+.0f}")
con.close()
# ---------- MIN 2025 passes: to JJ vs to everyone else ----------
print("\n  MIN 2025 pbp: passes TO Jefferson vs other MIN targets:")
pbp=pd.read_parquet(REL+"pbp/play_by_play_2025.parquet",
    columns=["posteam","season_type","pass_attempt","sack","interception","complete_pass",
             "yards_gained","epa","air_yards","receiver_player_name"])
mn=pbp[(pbp.posteam=="MIN")&(pbp.season_type=="REG")&(pbp.pass_attempt==1)]
tojj=mn[mn.receiver_player_name.fillna("").str.contains("Jefferson")]
oth=mn[~mn.receiver_player_name.fillna("").str.contains("Jefferson")&mn.receiver_player_name.notna()]
for lbl,dd in [("to Jefferson",tojj),("to others",oth)]:
    print(f"    {lbl:13s} n={len(dd):3d}  comp {dd.complete_pass.mean():.0%}  Y/A {dd.yards_gained.mean():4.1f}  EPA/att {dd.epa.mean():+.2f}  aDOT {dd.air_yards.mean():.1f}")
print(f"    MIN sack rate {mn.sack.mean():.1%} | INT rate {mn.interception.mean():.1%}")

# ---------- B) HISTORY: QB-collapse down-year WRs — rebound vs OUR model ----------
print("\n=== B) HISTORICAL REBOUND TEST (vs actual model projections) ===")
con=sqlite3.connect(DB)
sp=pd.read_sql("SELECT player_id,season,position,pred_ppg,next_ppg,prior_ppg FROM season_predictions WHERE position='WR'",con)
sd=pd.read_sql("SELECT player_id,season,prior_target_share,prior2_ppg,prior_team FROM season_dataset",con)
xfh=pd.read_sql("""SELECT player_id,season,SUM(total_fantasy_points) act,SUM(total_fantasy_points_exp) exp,
     COUNT(DISTINCT week) g FROM nflv_ff_opp GROUP BY player_id,season""",con)
con.close()
xfh["gap_pg"]=(xfh.act-xfh.exp)/xfh.g; xfh["season"]=xfh.season.astype(int)+1   # keyed to target season
d=sp.merge(sd,on=["player_id","season"],how="left").merge(xfh[["player_id","season","gap_pg"]],on=["player_id","season"],how="left")
qbq=qb[["season","team","epa_att_rk","cpoe_rk"]].copy(); qbq["season"]=qbq.season+1   # prior-yr QB quality -> target season
d=d.merge(qbq,left_on=["season","prior_team"],right_on=["season","team"],how="left")
d["resid"]=d.next_ppg-d.pred_ppg
# elite-role WR coming off a DOWN year (had a real role, produced >=10 ppg two years ago, fell >=3)
base=d[(d.prior_target_share>=0.20)&(d.prior2_ppg>=10)&(d.prior_ppg<=d.prior2_ppg-3)&d.next_ppg.notna()].copy()
qbbad=base[base.epa_att_rk>=24]          # prior-yr team QB bottom-9
qbok=base[base.epa_att_rk<=16]
def rep(lbl,g):
    if len(g)<8: print(f"  {lbl}: n={len(g)} too small"); return
    se=g.resid.std()/np.sqrt(len(g))
    print(f"  {lbl:44s} n={len(g):3d}  resid {g.resid.mean():+.2f} (t={g.resid.mean()/se:+.1f})  rebound {(g.next_ppg-g.prior_ppg).mean():+.1f} PPG")
rep("down-yr WR, QB was BAD (EPA rank 24+)",qbbad)
rep("down-yr WR, QB was fine (rank <=16)",qbok)
rep("down-yr WR + role intact + conv gap <=-1/g",base[(base.gap_pg<=-1)&(base.epa_att_rk>=20)])
rep("ALL down-yr elite-role WRs",base)
print("\n  Who the QB-BAD group was (top rebounds):")
con=sqlite3.connect(DB)
nm=pd.read_sql("SELECT DISTINCT player_id,player_display_name FROM nflv_season",con); con.close()
qq=qbbad.merge(nm,on="player_id",how="left"); qq["reb"]=qq.next_ppg-qq.prior_ppg
for r in qq.sort_values("reb",ascending=False).head(8).itertuples():
    print(f"    {r.season} {r.player_display_name[:20]:20s} prior {r.prior_ppg:4.1f} -> next {r.next_ppg:4.1f} (proj {r.pred_ppg:4.1f}, resid {r.resid:+.1f})")

# conditional: QB-BAD down-yr WRs where the team's QB play actually IMPROVED in the target season
qbn=qb[["season","team","epa_att_rk"]].rename(columns={"epa_att_rk":"next_rk"})   # target-season QB rank
cond=qbbad.merge(qbn,left_on=["season","prior_team"],right_on=["season","team"],how="left")
imp=cond[cond.next_rk<=cond.epa_att_rk-8]; non=cond[cond.next_rk>cond.epa_att_rk-8]
for lbl,g in [("QB-BAD & QB context IMPROVED next yr",imp),("QB-BAD & QB stayed bad",non)]:
    if len(g)>=5:
        print(f"  {lbl:44s} n={len(g):3d}  resid {g.resid.mean():+.2f}  rebound {(g.next_ppg-g.prior_ppg).mean():+.1f} PPG")
    else: print(f"  {lbl}: n={len(g)}")

# ---------- C) 2026 watchlist: down-yr elite-role WRs split by 2025 team QB quality ----------
print("\n=== C) 2026 down-year WR watchlist (elite role in 2025, fell >=3 PPG from 2024) ===")
con=sqlite3.connect(DB)
tr=pd.read_sql("""SELECT player_id,player_display_name nm,season,ppg,games FROM nflv_traj
                  WHERE position='WR' AND season IN (2024,2025)""",con)
ts=pd.read_sql("""SELECT player_id,AVG(target_share) tsh,MAX(team) tm FROM nflv_weekly
                  WHERE season=2025 AND position='WR' GROUP BY player_id""",con)
con.close()
p25=tr[tr.season==2025].merge(tr[tr.season==2024][["player_id","ppg"]].rename(columns={"ppg":"ppg24"}),on="player_id")
p25=p25.merge(ts,on="player_id",how="left")
cand=p25[(p25.tsh>=0.20)&(p25.ppg24>=10)&(p25.ppg<=p25.ppg24-3)&(p25.games>=8)]
q25=qb[qb.season==2025][["team","epa_att_rk"]]
cand=cand.merge(q25,left_on="tm",right_on="team",how="left")
for r in cand.sort_values("epa_att_rk").itertuples():
    grp="REBOUND candidate (QB was fine)" if r.epa_att_rk<=16 else ("QB-excuse TRAP (QB bad — no historical rebound)" if r.epa_att_rk>=24 else "middle")
    print(f"  {r.nm[:22]:22s} {r.tm:4s} 2024 {r.ppg24:4.1f} -> 2025 {r.ppg:4.1f}  tgt-sh {r.tsh:.0%}  QB EPA rank {r.epa_att_rk:.0f}  -> {grp}")
