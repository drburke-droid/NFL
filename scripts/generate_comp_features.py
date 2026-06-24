"""
Leakage-free comp features for the projection/risk model.

For each (player, career year N, season S), find same-position historical players
whose first-N-year trajectory is most similar AND whose forward outcomes (years
N+1..N+3) were already known by season S, then summarize that cohort:
  comp_prime  = median of comps' prime (best of next 3, attrition=0)
  comp_bust   = share of comps whose prime stayed < 8 PPG
  comp_elite  = share of comps who reached 18+ PPG
  comp_dist   = mean trajectory distance of the cohort (how "unprecedented" he is)
Keyed by target season (S+1) so it merges onto season_dataset.

Writes nflv_comp_features.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("cf", os.path.join(os.path.dirname(__file__), "comp_finder.py"))
CF = importlib.util.module_from_spec(_s); _s.loader.exec_module(CF)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
K = 10; MISS = CF.MISS_PEN


def main():
    t = CF.load()
    pos_of = t.drop_duplicates("player_id").set_index("player_id")["position"].to_dict()
    rook = t.drop_duplicates("player_id").set_index("player_id")["rookie_year"].to_dict()
    ppg = {(r.player_id, int(r.career_year)): r.ppg for _, r in t.iterrows()}
    season = {(r.player_id, int(r.career_year)): int(r.season) for _, r in t.iterrows()}
    V = {}
    for p in CF.FEATS: V.update(CF.vecs_for(t, p))
    by_pos = {}
    for p in pos_of: by_pos.setdefault(pos_of[p], []).append(p)
    prime = lambda c, N, H: max(ppg.get((c, N+h), 0.0) for h in range(1, H+1))

    rows = []
    q = t[t.career_year >= 2]
    for _, r in q.iterrows():
        pid, N = r.player_id, int(r.career_year); S = season.get((pid, N))
        if S is None: continue
        qv = {cy: V[(pid, cy)] for cy in range(1, N+1) if (pid, cy) in V}
        if len(qv) < 2: continue
        pos = pos_of[pid]; cands = []
        for c in by_pos[pos]:
            if c == pid: continue
            rc = rook.get(c)
            if pd.isna(rc) or rc < 2006 or rc + N + 2 > S: continue   # forward (yr N+3) known by S
            d = sum(np.linalg.norm(qv.get(cy)-V[(c,cy)]) if (cy in qv and (c,cy) in V) else MISS
                    for cy in range(1, N+1)) / N
            cands.append((c, d))
        if len(cands) < 4: continue
        cands.sort(key=lambda x: x[1]); top = cands[:K]
        primes = np.array([prime(c, N, 3) for c, _ in top])
        rows.append({"player_id": pid, "season": S+1,           # predicting next year
                     "comp_prime": float(np.median(primes)),
                     "comp_bust": float(np.mean(primes < 8)),
                     "comp_elite": float(np.mean(primes >= 18)),
                     "comp_dist": float(np.mean([d for _, d in top])),
                     "comp_n": len(top)})
    out = pd.DataFrame(rows)
    con = sqlite3.connect(DB); out.to_sql("nflv_comp_features", con, if_exists="replace", index=False); con.close()
    print(f"nflv_comp_features: {len(out):,} player-seasons ({int(out.season.min())}-{int(out.season.max())})")
    print(out[["comp_prime","comp_bust","comp_elite","comp_dist"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
