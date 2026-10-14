"""
150-lineup portfolios (DK max entry): how close does our BEST lineup get to the hindsight oracle?

Per slate, metric = max over the portfolio of (lineup actual / oracle actual), the oracle solved on the full
slate. Reported: mean best %, and P(best >= 70 / 80 / 90% of oracle). Walk-forward 2018-25: the
projection is FFA recalibrated per position on earlier seasons (dfs_oracle_gap: the best single-lineup
input), and the outcome model (dfs_backtest.fit: residual quantiles + role copula) is fitted on
earlier seasons.

Portfolios (N lineups each):
  single      the one max-projection lineup (reference; N = 1)
  proj_div    N best by projection, each pair differing by >= 3 players
  jitter15    max projection on projections x (1 + N(0, 0.15)), N draws (the common MME method)
  jitter30    same, sd 0.30
  sim_l05     optimum of proj + 0.5 x (simulated world - proj), N worlds
  sim_l1      optimum of a full simulated world, N worlds
  sim_l1_st   sim_l1 under stack rules (QB + 2 own WR/TE + 1 bring-back, nothing vs own DST)
  greedy      from the pool of everything above (+ extra sims), greedily pick N maximising the mean, over
              fresh simulated worlds, of the best lineup's share of that world's oracle
  greedy_p80  same, maximising P(best >= 80% of the world oracle)

    python scripts/dfs_mme_oracle.py --procs 4
Each finished slate is saved to outputs/dfs/mme_n<N>/<season>_<week>.parquet and skipped on a re-run, so an
interrupted run resumes where it stopped.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfs_optimizer import Slate
from dfs_backtest import fit, roles, game_corr, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dfs")
EVAL_W = 300


def recalibrate(d, y):
    """Per-position linear fit of actual on FFA proj, earlier seasons -> applied to everything <= y."""
    tr = d[(d.season < y) & d.has_proj]
    out = d.proj.copy()
    for pos, x in tr.groupby("pos"):
        b = np.polyfit(x.proj, x.actual, 1)
        m = (d.pos == pos) & d.has_proj
        out[m] = np.clip(b[1] + b[0] * d.proj[m], 0, None)
    return out


def greedy(Fr, n, crit):
    best = np.zeros(Fr.shape[1]); chosen = []
    for _ in range(min(n, len(Fr))):
        if crit == "mean":
            gain = np.maximum(Fr, best).mean(1) - best.mean()
        else:
            gain = ((np.maximum(Fr, best) >= 0.8).mean(1) - (best >= 0.8).mean()) + 1e-3 * Fr.mean(1)
        gain[chosen] = -np.inf
        i = int(np.argmax(gain)); chosen.append(i); best = np.maximum(best, Fr[i])
    return chosen


def run(args):
    path = args[-1]
    try:
        rows = _run(*args[:-1])
        if rows: pd.DataFrame(rows).to_parquet(path, index=False)
        return rows
    except Exception as e:
        print(f"  slate failed: {type(e).__name__}: {e}", flush=True); return []


def _run(full, model, n, seed):
    rng = np.random.default_rng(seed)
    if not (full.pos == "DST").any(): return []
    oracle = full.actual.values[Slate(full.reset_index(drop=True)).solve(full.actual.values)].sum()
    sl = roles(full).reset_index(drop=True)
    sl = sl[(sl.proj > 0.5) | (sl.pos == "DST")].reset_index(drop=True)
    base = Slate(sl); stk = Slate(sl, stack=2, bringback=1, no_dst_vs=True)
    proj, act = sl.proj.values, sl.actual.values
    C = game_corr(sl, model)
    P = {"single": [base.solve(proj)]}
    got = []
    for _ in range(n):
        idx = base.solve(proj, exclude=got, min_diff=3)
        if idx is None: break
        got.append(idx)
    P["proj_div"] = got
    for sd, k in ((0.15, "jitter15"), (0.30, "jitter30")):
        P[k] = [base.solve(proj * np.clip(1 + rng.normal(0, sd, len(proj)), 0, None)) for _ in range(n)]
    W = simulate(sl, model, C, 2 * n, rng)
    P["sim_l05"] = [base.solve(proj + 0.5 * (w - proj)) for w in W[:n]]
    P["sim_l1"] = [base.solve(w) for w in W[:n]]
    P["sim_l1_st"] = [x for x in (stk.solve(w) for w in W[:n]) if x is not None]
    extra = [base.solve(proj + lam * (w - proj)) for w, lam in zip(W[n:], np.tile([0.3, 0.7, 1.0], n))]
    pool = {}
    for k, L in P.items():
        for a in L:
            if a is not None: pool[tuple(a)] = a
    for a in extra:
        if a is not None: pool[tuple(a)] = a
    cands = list(pool.values())
    E = simulate(sl, model, C, EVAL_W, rng)
    world_or = np.array([e[base.solve(e)].sum() for e in E])
    Fr = np.stack([E[:, a].sum(1) for a in cands]) / world_or            # candidate x world share of oracle
    P["greedy"] = [cands[i] for i in greedy(Fr, n, "mean")]
    P["greedy_p80"] = [cands[i] for i in greedy(Fr, n, "p80")]
    rows = []
    for k, L in P.items():
        L = [a for a in L if a is not None]
        s = np.array([act[a].sum() for a in L])
        frac = s / oracle
        qbs = {sl.player.values[a][sl.pos.values[a] == "QB"][0] for a in L}
        uniq = len({tuple(sorted(a)) for a in L})
        rows.append(dict(season=sl.season.iat[0], week=sl.week.iat[0], portfolio=k, n=len(L), unique=uniq,
                         oracle=oracle, best=frac.max(), mean=frac.mean(), qbs=len(qbs),
                         ge70=float(frac.max() >= .7), ge80=float(frac.max() >= .8), ge90=float(frac.max() >= .9)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2018-2025"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--procs", type=int, default=4)
    A = ap.parse_args()
    a, b = (A.seasons.split("-") + [A.seasons])[:2]
    d0 = pd.read_parquet(os.path.join(OUT, "slates.parquet"))
    ck = os.path.join(OUT, f"mme_n{A.n}"); os.makedirs(ck, exist_ok=True)
    jobs, done = [], []
    for y in range(int(a), int(b) + 1):
        d = d0[d0.season <= y].copy(); d["proj"] = recalibrate(d, y)
        m = fit(d[d.season < y])
        for w, sl in d[d.season == y].groupby("week"):
            path = os.path.join(ck, f"{y}_{w:02d}.parquet")
            if os.path.exists(path): done.append(pd.read_parquet(path)); continue
            jobs.append((sl, m, A.n, y * 100 + w, path))
    print(f"{len(done)} slates already saved, {len(jobs)} to run", flush=True)
    t0 = time.time(); res = [x for f in done for x in f.to_dict("records")]
    with Pool(A.procs) as p:
        for i, r in enumerate(p.imap_unordered(run, jobs)):
            res += r
            if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(res); r.to_parquet(os.path.join(OUT, f"mme_oracle_n{A.n}.parquet"), index=False)
    print(f"\n{r.groupby(['season', 'week']).ngroups} slates, N = {A.n}\n")
    t = r.groupby("portfolio")[["best", "ge70", "ge80", "ge90", "mean", "unique", "qbs"]].mean()
    print(t.sort_values("best", ascending=False).round(3).to_string())
    w = r.pivot_table(index=["season", "week"], columns="portfolio", values="best")
    rng = np.random.default_rng(0)
    print("\nbest-lineup share vs jitter15, paired (95% bootstrap):")
    for k in w.columns:
        if k == "jitter15": continue
        dl = (w[k] - w["jitter15"]).dropna().values
        bs = rng.choice(dl, (4000, len(dl))).mean(1)
        print(f"  {k:11s} {dl.mean():+.3f} [{np.quantile(bs, .025):+.3f},{np.quantile(bs, .975):+.3f}]")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
