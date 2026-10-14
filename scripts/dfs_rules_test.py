"""
Do the structural traits of near-oracle rosters (dfs_near_oracle_report) help when imposed as rules?

For every 2016-25 main slate, build 150 lineups with the jitter-15 method (max projection on FFA x
(1 + N(0, 0.15)) per lineup) under each rule set, using the SAME 150 jitter draws for every rule set so the
comparison is paired. Score each portfolio on actual points vs the full-slate oracle:
    best      best lineup's share of oracle
    hits95    lineups within 5% of oracle (the near-oracle set)
    any95     at least one such lineup
    any90     at least one within 10%
    mean      average lineup share of oracle

Rule sets:
    base        none
    stack1      QB + >= 1 own WR/TE
    nodst       no offensive player facing own DST
    noTEflex    at most 1 TE
    s1_nodst    stack1 + nodst
    s1_nd_nte   stack1 + nodst + noTEflex
    s1_bb       stack1 + >= 1 opponent RB/WR/TE (bring-back)
    s2_bb_nd    QB + 2 own WR/TE + bring-back + nodst

The rules came from looking at the same slates, so the report also splits 2016-20 vs 2021-25.

    python scripts/dfs_rules_test.py --procs 3
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfs_optimizer import Slate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dfs")
CK = os.path.join(OUT, "rules_test")
N = 150
RULES = {"base": {}, "stack1": dict(stack=1), "nodst": dict(no_dst_vs=True), "noTEflex": dict(te_max=1),
         "s1_nodst": dict(stack=1, no_dst_vs=True), "s1_nd_nte": dict(stack=1, no_dst_vs=True, te_max=1),
         "s1_bb": dict(stack=1, bringback=1), "s2_bb_nd": dict(stack=2, bringback=1, no_dst_vs=True)}


def run(args):
    sl, path, seed = args
    try:
        rng = np.random.default_rng(seed)
        full = sl.reset_index(drop=True)
        if not (full.pos == "DST").any(): return None
        oracle = full.actual.values[Slate(full).solve(full.actual.values)].sum()
        s = full[(full.proj > 0.5) | (full.pos == "DST")].reset_index(drop=True)
        proj, act = s.proj.values, s.actual.values
        draws = [proj * np.clip(1 + rng.normal(0, .15, len(proj)), 0, None) for _ in range(N)]
        rows = []
        for name, kw in RULES.items():
            S = Slate(s, **kw)
            sc = np.array([act[i].sum() for i in (S.solve(v) for v in draws) if i is not None]) / oracle
            if not len(sc): continue
            rows.append(dict(season=s.season.iat[0], week=s.week.iat[0], rule=name, n=len(sc), best=sc.max(),
                             hits95=int((sc >= .95).sum()), any95=float((sc >= .95).any()),
                             any90=float((sc >= .90).any()), any80=float((sc >= .80).any()), mean=sc.mean()))
        pd.DataFrame(rows).to_parquet(path, index=False)
        return s.season.iat[0], s.week.iat[0]
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}", flush=True); return None


def report():
    import glob
    r = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(CK, "*.parquet"))])
    print(f"{r.groupby(['season', 'week']).ngroups} slates, {N} lineups per rule set\n")
    t = r.groupby("rule")[["best", "any80", "any90", "any95", "hits95", "mean"]].mean()
    w = r.pivot_table(index=["season", "week"], columns="rule", values=["best", "hits95"])
    rng = np.random.default_rng(0)
    for m in ("best", "hits95"):
        lo, hi, win = [], [], []
        for k in t.index:
            dl = (w[m][k] - w[m]["base"]).dropna().values
            bs = rng.choice(dl, (4000, len(dl))).mean(1)
            lo.append(np.quantile(bs, .025)); hi.append(np.quantile(bs, .975)); win.append((dl > 0).mean() - (dl < 0).mean())
        t[f"{m}_vs_base"] = t[m] - t.loc["base", m]; t[f"{m}_ci"] = [f"[{a:+.3f},{b:+.3f}]" for a, b in zip(lo, hi)]
    print(t.sort_values("best", ascending=False).round(3).to_string())
    r["era"] = np.where(r.season <= 2020, "2016-20", "2021-25")
    print("\nbest share and hits95 by era:")
    print(r.pivot_table(index="rule", columns="era", values=["best", "hits95"]).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    A = ap.parse_args()
    if A.report: return report()
    os.makedirs(CK, exist_ok=True)
    d = pd.read_parquet(os.path.join(OUT, "slates.parquet"))
    jobs = [(sl, os.path.join(CK, f"{y}_{w:02d}.parquet"), y * 100 + w) for (y, w), sl in d.groupby(["season", "week"])
            if not os.path.exists(os.path.join(CK, f"{y}_{w:02d}.parquet"))]
    print(f"{len(jobs)} slates to run", flush=True); t0 = time.time()
    with Pool(A.procs) as p:
        for i, r in enumerate(p.imap_unordered(run, jobs)):
            if (i + 1) % 20 == 0: print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
