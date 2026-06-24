"""
Make career comps more powerful: instead of a fading far-future average (which
decays to ~0 as careers end), summarize each comp cohort's OUTCOME DISTRIBUTION —
prime/peak production, bust rate, survival — and test whether it carries more
signal than the current next-year point projection.

1. Demonstrates the attrition problem (mean forward PPG by horizon -> 0) and that
   PRIME (best of the next 3 years, attrition counted as 0) does NOT collapse.
2. Leakage-free backtest: does the comp cohort's prime predict a player's actual
   prime (best of next 3 yrs) better than a naive baseline? And does cohort bust
   rate / survival predict actual bust / survival?
"""
import os, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("cf", os.path.join(os.path.dirname(__file__), "comp_finder.py"))
CF = importlib.util.module_from_spec(_s); _s.loader.exec_module(CF)

K = 10; MISS = CF.MISS_PEN
BUST = 8.0     # forward prime < 8 PPG = never a useful starter again
ELITE = 18.0


def main():
    t = CF.load()
    pos_of = t.drop_duplicates("player_id").set_index("player_id")["position"].to_dict()
    rook = t.drop_duplicates("player_id").set_index("player_id")["rookie_year"].to_dict()
    ppg = {(r.player_id, int(r.career_year)): r.ppg for _, r in t.iterrows()}  # only games>=4 seasons
    season = {(r.player_id, int(r.career_year)): int(r.season) for _, r in t.iterrows()}
    V = {}
    for p in CF.FEATS: V.update(CF.vecs_for(t, p))
    by_pos = {}
    for p in pos_of: by_pos.setdefault(pos_of[p], []).append(p)

    def fwd(pid, N, h): return ppg.get((pid, N+h), 0.0)              # attrition -> 0
    def prime(pid, N, H=3): return max(fwd(pid, N, h) for h in range(1, H+1))
    def played(pid, N, h): return (pid, N+h) in ppg

    # ---------- 1. attrition demonstration ----------
    print("=== Career comps decay to ~0 if you average the far future (attrition) ===")
    base = t[(t.career_year >= 2) & (t.ppg >= 8)]
    hz = {h: [] for h in range(1, 6)}; surv = {h: [] for h in range(1, 6)}
    for _, q in base.iterrows():
        pid, N = q.player_id, int(q.career_year)
        for h in range(1, 6):
            hz[h].append(fwd(pid, N, h)); surv[h].append(1.0 if played(pid, N, h) else 0.0)
    print("  horizon (yrs out):   " + "  ".join(f"+{h}" for h in range(1,6)))
    print("  mean PPG (attrition):" + "".join(f"{np.mean(hz[h]):6.1f}" for h in range(1,6)) + "   <- fades toward 0")
    print("  still playing %:     " + "".join(f"{np.mean(surv[h])*100:5.0f}%" for h in range(1,6)))
    print(f"  mean PRIME (best of next 3, attrition=0): {np.mean([prime(q.player_id,int(q.career_year)) for _,q in base.iterrows()]):.1f}  <- does NOT collapse")

    # ---------- 2. leakage-free backtest ----------
    rows = []
    q = t[(t.career_year >= 2) & (t.career_year <= 7) & (t.ppg >= 5)]
    for _, r in q.iterrows():
        pid, N = r.player_id, int(r.career_year); S = season.get((pid, N))
        if S is None or (pid, N+1) not in ppg: continue       # need at least next yr observed for player
        qv = {cy: V[(pid, cy)] for cy in range(1, N+1) if (pid, cy) in V}
        if len(qv) < 2: continue
        pos = pos_of[pid]; cands = []
        for c in by_pos[pos]:
            if c == pid: continue
            rc = rook.get(c)
            if pd.isna(rc) or rc < 2006: continue
            if rc + N + 2 > S: continue                        # comp's yr N+3 must be known by season S
            d = sum(np.linalg.norm(qv.get(cy)-V[(c,cy)]) if (cy in qv and (c,cy) in V) else MISS
                    for cy in range(1, N+1)) / N
            cands.append((c, d))
        if len(cands) < 5: continue
        cands.sort(key=lambda x: x[1]); top = [c for c, _ in cands[:K]]
        cohort_prime = [prime(c, N) for c in top]
        cohort_next = [fwd(c, N, 1) for c in top]
        rows.append({
            "pid": pid, "N": N, "base": r.ppg,
            "proj_prime": float(np.median(cohort_prime)),
            "proj_next": float(np.median(cohort_next)),
            "bust_rate": float(np.mean([p < BUST for p in cohort_prime])),
            "elite_rate": float(np.mean([p >= ELITE for p in cohort_prime])),
            "act_prime": prime(pid, N), "act_next": fwd(pid, N, 1),
            "act_bust": 1.0 if prime(pid, N) < BUST else 0.0,
            "act_elite": 1.0 if prime(pid, N) >= ELITE else 0.0,
        })
    d = pd.DataFrame(rows)
    print(f"\n=== Leakage-free backtest (n={len(d):,}) ===")

    print("\n  Predicting PLAYER'S PRIME (best of next 3 yrs):")
    for lab, col in [("comp prime median", "proj_prime"), ("repeat last year", "base")]:
        print(f"    {lab:20s} MAE {mean_absolute_error(d.act_prime,d[col]):.2f}  Spearman {spearmanr(d[col],d.act_prime)[0]:.3f}")
    d["blend_prime"] = 0.5*d.proj_prime + 0.5*d.base
    print(f"    {'50/50 blend':20s} MAE {mean_absolute_error(d.act_prime,d.blend_prime):.2f}  Spearman {spearmanr(d.blend_prime,d.act_prime)[0]:.3f}")

    print("\n  Predicting NEXT-YEAR (current method, for reference):")
    for lab, col in [("comp next median", "proj_next"), ("repeat last year", "base")]:
        print(f"    {lab:20s} MAE {mean_absolute_error(d.act_next,d[col]):.2f}  Spearman {spearmanr(d[col],d.act_next)[0]:.3f}")

    print("\n  Risk signals (do cohort rates predict the actual outcome?):")
    print(f"    cohort BUST rate  -> actual bust  : AUC {roc_auc_score(d.act_bust,d.bust_rate):.3f}  (base rate {d.act_bust.mean():.0%})")
    print(f"    cohort ELITE rate -> actual elite : AUC {roc_auc_score(d.act_elite,d.elite_rate):.3f}  (base rate {d.act_elite.mean():.0%})")
    # decile lift on elite
    d["eb"] = pd.qcut(d.elite_rate.rank(method="first"), 5, labels=False)
    print("    actual-elite rate by cohort elite-rate quintile:",
          [f"{d[d.eb==i].act_elite.mean():.0%}" for i in range(5)])


if __name__ == "__main__":
    main()
