"""
Rookie career comps: match each 2026 rookie to historical rookies (2011-2025) by
PRE-NFL profile — draft capital, combine athleticism, age, size, and landing spot
— then show how those comps actually panned out (their rookie-year PPG).

Merges the rookie entries into docs/comps.js (+ outputs/draft_tool) so the web
Career Comps tab covers rookies too.
"""
import os, sqlite3, json, re, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s=importlib.util.spec_from_file_location("mr",os.path.join(os.path.dirname(__file__),"model_rookie.py"))
MR=importlib.util.module_from_spec(_s); _s.loader.exec_module(MR)

DB=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"db","nfl_odds.db")
DOCS=[os.path.join(os.path.dirname(os.path.dirname(__file__)),"docs"),
      os.path.join(os.path.dirname(os.path.dirname(__file__)),"outputs","draft_tool")]
FEATS=["draft_pick","age","forty","vertical","broad_jump","weight","height",
       "team_off_ppg_prior","team_pos_ppg_prior","team_pass_att_prior","team_rush_att_prior"]
PFR2NFLV={"GNB":"GB","KAN":"KC","LAR":"LA","LVR":"LV","NOR":"NO","NWE":"NE","SFO":"SF","TAM":"TB"}
norm=lambda s:re.sub(r"\s+"," ",re.sub(r"[^a-z ]","",str(s).lower().replace(".","").replace("'",""))).strip()


def rookies_2026(con):
    d=pd.read_sql("""SELECT gsis_id player_id, pfr_player_id, pfr_player_name name, position, team,
                     pick draft_pick, round draft_round, age FROM nflv_draft
                     WHERE season=2026 AND position IN ('QB','RB','WR','TE')""",con)
    d["team"]=d["team"].replace(PFR2NFLV)
    cb=pd.read_sql("SELECT pfr_id, forty, vertical, broad_jump, cone, shuttle, bench, ht, wt FROM nflv_combine WHERE season=2026",con)
    def toin(h):
        try:
            if pd.isna(h):return np.nan
            if "-" in str(h):f,i=str(h).split("-");return int(f)*12+int(i)
            return float(h)
        except Exception:return np.nan
    cb["height"]=cb["ht"].map(toin); cb["weight"]=pd.to_numeric(cb["wt"],errors="coerce")
    d=d.merge(cb.drop(columns=["ht","wt"]),left_on="pfr_player_id",right_on="pfr_id",how="left")
    seas=pd.read_sql("SELECT recent_team team, position, attempts, carries, fantasy_points_ppr FROM nflv_season WHERE season=2025",con)
    ta=seas.groupby("team").agg(team_off_ppg_prior=("fantasy_points_ppr",lambda s:s.sum()/17),
        team_pass_att_prior=("attempts",lambda s:s.sum()/17),team_rush_att_prior=("carries",lambda s:s.sum()/17)).reset_index()
    tp=seas.groupby(["team","position"]).agg(team_pos_ppg_prior=("fantasy_points_ppr",lambda s:s.sum()/17)).reset_index()
    return d.merge(ta,on="team",how="left").merge(tp,on=["team","position"],how="left")


def main():
    con=sqlite3.connect(DB)
    hist=MR.build(con)                              # historical rookies 2011-2025 + rookie ppg
    r26=rookies_2026(con)
    # career outcomes of historical players (peak PPG over their career) + entry year
    tr=pd.read_sql("SELECT player_id, career_year, ppg, rookie_year FROM nflv_traj",con)
    con.close()
    career_prime=tr.groupby("player_id")["ppg"].max().to_dict()
    entry=tr.groupby("player_id")["rookie_year"].min().to_dict()

    # standardize features within position (over the historical pool)
    stats={}
    for pos,g in hist.groupby("position"):
        stats[pos]={f:(g[f].mean(),g[f].std() if g[f].std()>0 else 1.0) for f in FEATS}
    def vec(row):
        st=stats.get(row["position"]);
        return np.array([((row[f]-st[f][0])/st[f][1] if pd.notna(row[f]) else 0.0) for f in FEATS])
    hist["_v"]=hist.apply(vec,axis=1)

    web={}
    for _,q in r26.iterrows():
        if q["position"] not in stats: continue
        qv=vec(q)
        pool=hist[hist.position==q["position"]].copy()
        pool["_d"]=pool["_v"].apply(lambda v:float(np.linalg.norm(qv-v)))
        top12=pool.nsmallest(12,"_d")
        top=top12.head(6)
        comps=[{"n":r.player_display_name,"d":round(r._d,2),"nx":round(r.ppg,1)} for _,r in top.iterrows()]
        proj=round(float(top["ppg"].median()),1)
        # career-outcome distribution over mature comps (entered <=2022) of the 12 nearest
        primes=[career_prime[p] for p in top12["player_id"] if entry.get(p,9999)<=2022 and p in career_prime]
        if len(primes)>=4:
            pr=np.array(primes)
            ceiling=round(float(np.percentile(pr,75)),1); floor=round(float(np.percentile(pr,25)),1)
            bust=round(float(np.mean(pr<8)),2); elite=round(float(np.mean(pr>=18)),2)
        else:
            ceiling=floor=bust=elite=None
        web[norm(q["name"])+"|"+q["position"]]={"p":q["name"],"pos":q["position"],"yr":1,
            "ppg":None,"proj":proj,"rookie":1,"ceiling":ceiling,"floor":floor,"bust":bust,"elite":elite,"comps":comps}

    # merge into existing comps.js
    base={}
    cj=os.path.join(DOCS[0],"comps.js")
    if os.path.exists(cj):
        txt=open(cj,encoding="utf-8").read()
        base=json.loads(txt.split("const COMPS = ",1)[1].rsplit(";",1)[0])
    base.update(web)
    out="const COMPS = "+json.dumps(base)+";\n"
    for d in DOCS:
        os.makedirs(d,exist_ok=True); open(os.path.join(d,"comps.js"),"w",encoding="utf-8").write(out)

    print(f"Rookie comps: {len(web)} rookies merged ({len(base)} total players in comps.js)")
    print("\nSample rookie comps:")
    for k in list(web)[:8]:
        e=web[k]; print(f"  {e['p']} ({e['pos']}, proj {e['proj']}): "+", ".join(f"{c['n']}→{c['nx']}" for c in e['comps'][:4]))


if __name__=="__main__":
    main()
