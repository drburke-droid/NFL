"""
Leakage-free backtest of the comp-based next-year projection.

For each historical (player, career year N) with an observed year N+1: find the
nearest-trajectory comps whose OWN year N+1 already happened on or before this
player's year-N season (so their continuation was knowable at the time), and
project = median of those comps' year-(N+1) PPG. Compare to actual, vs a
repeat-last-year baseline.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import importlib.util
_s = importlib.util.spec_from_file_location("cf", os.path.join(os.path.dirname(__file__), "comp_finder.py"))
CF = importlib.util.module_from_spec(_s); _s.loader.exec_module(CF)
K = 8; MISS = CF.MISS_PEN


def main():
    t = CF.load()
    pos_of = t.drop_duplicates("player_id").set_index("player_id")["position"].to_dict()
    rook = t.drop_duplicates("player_id").set_index("player_id")["rookie_year"].to_dict()
    ppg = {(r.player_id, int(r.career_year)): r.ppg for _, r in t.iterrows()}
    season = {(r.player_id, int(r.career_year)): int(r.season) for _, r in t.iterrows()}
    V = {}
    for p in CF.FEATS: V.update(CF.vecs_for(t, p))
    by_pos = {}
    for p in pos_of:
        by_pos.setdefault(pos_of[p], []).append(p)

    rows = []
    obs = t[t.career_year >= 2]  # need year>=2 so there's a year-1.. to compare
    queries = [(r.player_id, int(r.career_year)) for _, r in
               t[(t.career_year >= 2) & (t.ppg >= 5)].iterrows()]
    for pid, N in queries:
        if (pid, N+1) not in ppg: continue
        S = season.get((pid, N))
        if S is None: continue
        qv = {cy: V[(pid, cy)] for cy in range(1, N+1) if (pid, cy) in V}
        if len(qv) < 2: continue
        pos = pos_of[pid]
        dists = []
        for c in by_pos[pos]:
            if c == pid: continue
            rc = rook.get(c)
            if pd.isna(rc) or rc < 2006: continue
            if (c, N+1) not in ppg: continue
            if rc + N > S: continue                # comp's year N+1 must be known by season S
            tot = 0.0
            for cy in range(1, N+1):
                a = qv.get(cy); b = V.get((c, cy))
                tot += np.linalg.norm(a-b) if (a is not None and b is not None) else MISS
            dists.append((c, tot/N))
        if len(dists) < 4: continue
        dists.sort(key=lambda x: x[1])
        comp_next = [ppg[(c, N+1)] for c, _ in dists[:K]]
        rows.append({"pid": pid, "N": N, "proj": float(np.median(comp_next)),
                     "baseline": ppg[(pid, N)], "actual": ppg[(pid, N+1)]})

    d = pd.DataFrame(rows)
    print(f"Backtested {len(d):,} (player, year) projections (leakage-free)\n")
    for lab, col in [("Comp projection", "proj"), ("Repeat last year", "baseline")]:
        mae = mean_absolute_error(d.actual, d[col]); rho = spearmanr(d[col], d.actual)[0]
        print(f"  {lab:18s} MAE {mae:.2f}  Spearman {rho:.3f}")
    # blend
    d["blend"] = 0.5*d.proj + 0.5*d.baseline
    print(f"  {'50/50 blend':18s} MAE {mean_absolute_error(d.actual,d.blend):.2f}  Spearman {spearmanr(d.blend,d.actual)[0]:.3f}")
    imp = 100*(mean_absolute_error(d.actual,d.baseline)-mean_absolute_error(d.actual,d.proj))/mean_absolute_error(d.actual,d.baseline)
    print(f"\n  Comp proj vs repeat-last-year: {imp:+.1f}% MAE")
    print("  By career year (comp MAE | baseline MAE):")
    for N in range(2,8):
        s = d[d.N==N]
        if len(s) >= 30:
            print(f"    yr {N}->{N+1} (n={len(s):4d}): {mean_absolute_error(s.actual,s.proj):.2f} | {mean_absolute_error(s.actual,s.baseline):.2f}")


if __name__ == "__main__":
    main()
