"""
Career-trajectory comp finder.

For a player at career year N, find the historical players (same position, entered
2006+) whose first-N-year trajectory is most similar, then show how those comps'
careers CONTINUED (years N+1..N+3) as a data-driven projection.

Metrics are era-normalized (z-scored within position-season) so comps match the
relative SHAPE of the arc, not absolute era scoring levels. Distance = mean
per-career-year Euclidean over the position's feature set, with a penalty for
missing/injured years.

Writes comp_results (top comps + projection for current ascending players).
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
FEATS = {
 "QB": ["ppg","attempts_pg","passing_yards_pg","passing_tds_pg","passing_interceptions_pg","rushing_yards_pg","passing_epa","total_tds"],
 "RB": ["ppg","carries_pg","rushing_yards_pg","rushing_tds_pg","targets_pg","receptions_pg","receiving_yards_pg","rushing_epa","total_tds"],
 "WR": ["ppg","targets_pg","receptions_pg","receiving_yards_pg","receiving_tds_pg","target_share","air_yards_share","wopr","receiving_epa","total_tds"],
 "TE": ["ppg","targets_pg","receptions_pg","receiving_yards_pg","receiving_tds_pg","target_share","wopr","receiving_epa","total_tds"],
}
MISS_PEN = 5.0   # per-year penalty when a candidate lacks a career year


def load():
    con = sqlite3.connect(DB)
    t = pd.read_sql("SELECT * FROM nflv_traj", con); con.close()
    t = t[(t.games >= 4)].copy()
    t["career_year"] = t["career_year"].astype(int)
    # era-normalize each feature within (position, season)
    allf = sorted({f for v in FEATS.values() for f in v})
    for f in allf:
        if f not in t.columns: t[f] = np.nan
        g = t.groupby(["position","season"])[f]
        t[f+"_z"] = ((t[f] - g.transform("mean")) / g.transform("std")).fillna(0.0)
    return t


def vecs_for(t, pos):
    """dict[(player_id, career_year)] -> normalized feature vector."""
    zf = [f+"_z" for f in FEATS[pos]]
    sub = t[t.position == pos]
    d = {}
    for _, r in sub.iterrows():
        d[(r.player_id, int(r.career_year))] = r[zf].values.astype(float)
    return d


def comps_for(t, V, pid, name_map, ppg_map, last_career_year, K=8):
    pos = name_map[pid][1]; N = last_career_year[pid]
    qv = {cy: V[(pid, cy)] for cy in range(1, N+1) if (pid, cy) in V}
    if len(qv) < 2: return None
    # candidate pool: same position, entered 2006+, observed at least through year N
    cand = [p for p in last_career_year
            if p != pid and name_map[p][1] == pos and last_career_year[p] >= N and name_map[p][2] >= 2006]
    rows = []
    for c in cand:
        tot = 0.0
        for cy in range(1, N+1):
            a = qv.get(cy); b = V.get((c, cy))
            tot += np.linalg.norm(a-b) if (a is not None and b is not None) else MISS_PEN
        rows.append((c, tot/N))
    rows.sort(key=lambda x: x[1])
    top = rows[:K]
    cont = {h: [] for h in [1,2,3]}
    for c, _ in top:
        for h in [1,2,3]:
            v = ppg_map.get((c, N+h))
            if v is not None: cont[h].append(v)
    proj = {h: (np.median(cont[h]) if cont[h] else None) for h in [1,2,3]}
    detail = [{"id": c, "dist": round(d,2), "nx": ppg_map.get((c, N+1))} for c, d in top]
    return {"N": N, "comps": top, "detail": detail, "proj": proj, "cont": cont}


def cohort_dist(ids, N, ppg_map, season_map, maxseason=2025):
    """Outcome distribution of a comp cohort: prime (best of next 3, attrition=0),
    ceiling (P75), bust% (prime<8), elite% (prime>=18). Right-censoring avoided by
    only evaluating the horizon each comp could actually have completed."""
    primes = []
    for c in ids:
        s = season_map.get((c, N))
        if s is None: continue
        H = min(3, maxseason - s)
        if H < 1: continue
        primes.append(max(ppg_map.get((c, N+h), 0.0) for h in range(1, H+1)))
    if not primes: return {}
    pr = np.array(primes)
    return {"ceiling": round(float(np.percentile(pr, 75)), 1),
            "floor": round(float(np.percentile(pr, 25)), 1),
            "prime": round(float(np.median(pr)), 1),
            "bust": round(float(np.mean(pr < 8)), 2),
            "elite": round(float(np.mean(pr >= 18)), 2)}


def main():
    t = load()
    name_map = {r.player_id: (r.player_display_name, r.position, int(r.rookie_year) if pd.notna(r.rookie_year) else 9999)
                for _, r in t.drop_duplicates("player_id").iterrows()}
    ppg_map = {(r.player_id, int(r.career_year)): r.ppg for _, r in t.iterrows()}
    season_map = {(r.player_id, int(r.career_year)): int(r.season) for _, r in t.iterrows()}
    last_cy = t.groupby("player_id")["career_year"].max().astype(int).to_dict()
    V = {}
    for pos in FEATS: V.update(vecs_for(t, pos))

    # all 2025 players past their rookie year with a real role -> comp coverage
    cur = t[(t.season == 2025) & (t.career_year >= 2) & (t.ppg >= 5)].sort_values("ppg", ascending=False)
    print(f"Finding comps for {len(cur)} current players (2025, yr>=2, ppg>=5)...\n")

    norm = lambda s: __import__("re").sub(r"\s+"," ",__import__("re").sub(r"[^a-z ]","",str(s).lower().replace(".","").replace("'",""))).strip()
    results, web = [], {}
    for _, q in cur.iterrows():
        r = comps_for(t, V, q.player_id, name_map, ppg_map, last_cy, K=8)
        if not r: continue
        comp_names = [(name_map[c][0], round(d,2)) for c, d in r["comps"]]
        comp_med = r["proj"][1]
        blend = round(0.5*comp_med + 0.5*q.ppg, 1) if comp_med is not None else round(q.ppg,1)
        results.append({"player": q.player_display_name, "position": q.position, "career_year": r["N"],
                        "ppg_2025": round(q.ppg,1), "proj_next_ppg": blend,
                        "comp_median": round(comp_med,1) if comp_med is not None else None,
                        "top_comps": ", ".join(f"{n} ({d})" for n,d in comp_names[:5])})
        dd = cohort_dist([c for c, _ in r["comps"]], r["N"], ppg_map, season_map)
        web[norm(q.player_display_name)+"|"+q.position] = {
            "p": q.player_display_name, "pos": q.position, "yr": r["N"], "ppg": round(q.ppg,1),
            "proj": blend, "ceiling": dd.get("ceiling"), "floor": dd.get("floor"),
            "bust": dd.get("bust"), "elite": dd.get("elite"),
            "comps": [{"n": name_map[d["id"]][0], "d": d["dist"],
                       "nx": round(d["nx"],1) if d["nx"] is not None else None} for d in r["detail"][:6]]}

    res = pd.DataFrame(results)
    con = sqlite3.connect(DB); res.to_sql("comp_results", con, if_exists="replace", index=False); con.close()
    import json
    js = "const COMPS = "+json.dumps(web)+";\n"
    for d in [os.path.join(os.path.dirname(os.path.dirname(__file__)),"docs"),
              os.path.join(os.path.dirname(os.path.dirname(__file__)),"outputs","draft_tool")]:
        os.makedirs(d, exist_ok=True); open(os.path.join(d,"comps.js"),"w",encoding="utf-8").write(js)

    print("=== Career comps for notable current young players (proj = comp-blended) ===")
    for _, r in res.head(16).iterrows():
        print(f"\n{r.player} ({r.position}, yr {r.career_year}, 2025 {r.ppg_2025} -> proj {r.proj_next_ppg} PPG)")
        print(f"   comps: {r.top_comps}")
    print(f"\nSaved comp_results + comps.js ({len(web)} players).")


if __name__ == "__main__":
    main()
