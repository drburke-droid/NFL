"""
GPP test with 20-lineup portfolios per slate (single-lineup top-1% rates are too rare to separate
strategies on ~120 slates). Same data, walk-forward outcome model and simulated field as
dfs_backtest.py.

Portfolios (N lineups each):
  proj       N best lineups by projection, each differing from the others by >= 3 players
  proj_st    the same under stack2bb rules (QB + 2 own WR/TE + 1 bring-back, no player vs the DST)
  sim_ev     candidate pool scored in simulated worlds; the N with the highest expected GPP payout
  sim_cover  greedy: each next lineup maximises P(the portfolio has >= one top-1% lineup)

Payout proxy (per 1-unit entry, by finishing percentile in the field; roughly a DK large-field GPP,
~20% paid, top-heavy): top 0.1% 100, top 1% 12, top 5% 4, top 10% 2.5, top 20% 1.6, else 0.
Ties with the field and duplicated lineups are ignored (the proxy flatters chalk slightly).

    python scripts/dfs_portfolio_backtest.py              # 2019-25
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfs_optimizer import Slate
from dfs_backtest import fit, roles, game_corr, simulate, LAMBDAS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dfs")
PAY = [(0.999, 100.0), (0.99, 12.0), (0.95, 4.0), (0.90, 2.5), (0.80, 1.6)]


def payout(pct):
    pct = np.asarray(pct, float); out = np.zeros_like(pct)
    for q, v in reversed(PAY): out = np.where(pct >= q, v, out)
    return out


def diverse(S, score, n, min_diff=3):
    got = []
    for _ in range(n):
        idx = S.solve(score, exclude=got, min_diff=min_diff)
        if idx is None: break
        got.append(idx)
    return got


def run(args):
    try:
        return _run(*args)
    except Exception as e:
        print(f"  slate failed: {type(e).__name__}: {e}", flush=True); return []


def _run(sl, model, field_n, sims, n, seed):
    rng = np.random.default_rng(seed)
    if not (sl.pos == "DST").any(): return []
    sl = roles(sl).reset_index(drop=True)
    sl = sl[(sl.proj > 0.5) | (sl.pos == "DST")].reset_index(drop=True)
    base = Slate(sl); stk1 = Slate(sl, stack=1); stk2 = Slate(sl, stack=2, bringback=1, no_dst_vs=True)
    proj, act = sl.proj.values, sl.actual.values
    field = []
    for f in range(field_n):
        idx = (stk1 if f % 2 else base).solve(proj * np.exp(rng.normal(0, 0.30, len(proj))))
        if idx is not None: field.append(idx)
    F = np.zeros((len(field), len(sl)))
    for i, idx in enumerate(field): F[i, idx] = 1
    ports = {"proj": diverse(base, proj, n), "proj_st": diverse(stk2, proj, n)}
    C = game_corr(sl, model)
    worlds = simulate(sl, model, C, sims, rng)
    pool = {}
    for k, w in enumerate(worlds):
        v = proj + LAMBDAS[k % len(LAMBDAS)] * (w - proj)
        for S in (base, stk2):
            a = S.solve(v)
            if a is not None: pool[tuple(a)] = a
    for p in ports.values():
        for a in p: pool[tuple(a)] = a
    cands = list(pool.values())
    pos_of = {tuple(a): i for i, a in enumerate(cands)}
    ev = simulate(sl, model, C, 2000, rng)
    fs = np.sort(ev @ F.T, 1)                                            # field scores per world, sorted
    cs = np.stack([ev[:, a].sum(1) for a in cands])                      # candidates x worlds
    pct = np.stack([np.searchsorted(fs[w], cs[:, w], side="left") for w in range(len(ev))], 1) / fs.shape[1]
    exp_pay = payout(pct).mean(1)
    ports["sim_ev"] = [cands[i] for i in np.argsort(-exp_pay)[:n]]
    hit = pct >= 0.99
    covered = np.zeros(hit.shape[1], bool); chosen = []
    for _ in range(n):
        gain = (hit & ~covered).sum(1).astype(float); gain[chosen] = -1
        i = int(np.argmax(gain + 1e-6 * exp_pay)); chosen.append(i); covered |= hit[i]
    ports["sim_cover"] = [cands[i] for i in chosen]
    # grade on actual points
    fa = np.sort(F @ act)
    rows = []
    for k, p in ports.items():
        s = np.array([act[a].sum() for a in p])
        pc = np.searchsorted(fa, s, side="left") / len(fa)
        own_qb = [sl.player.values[a][sl.pos.values[a] == "QB"][0] for a in p]
        rows.append(dict(season=sl.season.iat[0], week=sl.week.iat[0], portfolio=k, n=len(p),
                         mean_pts=s.mean(), best_pts=s.max(), best_pct=pc.max(),
                         top1_hits=int((pc >= 0.99).sum()), any_top1=float((pc >= 0.99).any()),
                         top01_hits=int((pc >= 0.999).sum()), cash_rate=float((pc >= 0.8).mean()),
                         roi=payout(pc).mean() - 1, distinct_qbs=len(set(own_qb)),
                         model_exp_roi=float(np.mean([exp_pay[pos_of[tuple(a)]] for a in p])) - 1))
    return rows


def boot(x, n=4000):
    x = np.asarray(x, float); rng = np.random.default_rng(0); m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2019-2025"); ap.add_argument("--field", type=int, default=1000)
    ap.add_argument("--sims", type=int, default=200); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=12)
    A = ap.parse_args()
    a, b = (A.seasons.split("-") + [A.seasons])[:2]
    d = pd.read_parquet(os.path.join(OUT, "slates.parquet"))
    jobs = []
    for y in range(int(a), int(b) + 1):
        m = fit(d[d.season < y])
        for w, sl in d[d.season == y].groupby("week"): jobs.append((sl, m, A.field, A.sims, A.n, y * 100 + w))
    t0 = time.time(); res = []
    with Pool(A.procs) as p:
        for i, r in enumerate(p.imap_unordered(run, jobs)):
            res += r
            if (i + 1) % 20 == 0: print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(res); r.to_parquet(os.path.join(OUT, "portfolio_backtest.parquet"), index=False)
    print(f"\n{r.groupby(['season', 'week']).ngroups} slates, {A.n} lineups each\n")
    print(r.groupby("portfolio")[["mean_pts", "best_pts", "best_pct", "top1_hits", "any_top1", "top01_hits",
                                  "cash_rate", "roi", "model_exp_roi", "distinct_qbs"]].mean().round(3).to_string())
    w = r.pivot_table(index=["season", "week"], columns="portfolio", values=["roi", "top1_hits", "mean_pts"])
    print("\npaired vs proj (95% bootstrap over slates):")
    for k in ("proj_st", "sim_ev", "sim_cover"):
        print(f"  {k:9s} " + "   ".join(f"{m} {x:+.3f} [{lo:+.3f},{hi:+.3f}]"
                                         for m in ("roi", "top1_hits", "mean_pts")
                                         for x, lo, hi in [boot(w[m][k] - w[m]["proj"])]))
    print("\nROI by season:"); print(r.pivot_table(index="portfolio", columns="season", values="roi").round(2).to_string())
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
