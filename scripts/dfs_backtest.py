"""
Walk-forward backtest of DraftKings NFL Classic lineup strategies on 2016-25 Sunday main slates.

Everything a strategy sees is pregame: salary, the FFA consensus projection (outputs/dfs/slates.parquet,
built by dfs_slates.py) and distributions fitted on EARLIER seasons only. Test seasons 2019-25.

Outcome model (fitted per test season on all prior seasons):
  * marginal: each player's DK points = projection + a residual drawn from the empirical residuals of
    players at the same position and projection decile (keeps the right skew: booms, zeroes).
  * dependence: Gaussian copula over same-game players. Each player gets a role by projection rank in
    his team-week (QB1, RB1, RB2, WR1..WR3, TE1, DST, else OTH_<pos>); the correlation of every
    role pair, own team and opponent, is measured on normal scores of the residuals.

Field: FIELD lineups per slate, each the max-projection lineup on that entrant's own noisy view of the
projections (x lognormal sd 0.30 per player), half of them forced to stack QB + 1 receiver. Same
public projection, different opinions: concentrated ownership on chalk, like a real field. No real
contest results are free, so every "beat the field" number is against this proxy.

Strategies (one lineup per slate):
  mean         max projection, no rules
  stack1       max projection, QB + >=1 own WR/TE
  stack2bb     max projection, QB + >=2 own WR/TE + >=1 opponent RB/WR/TE, no player facing the DST
  sim_mean     candidates = optimal lineups of SIMS simulated worlds; pick max simulated mean
  sim_cash     same pool; pick max P(score >= field median), field scored in the same worlds
  sim_gpp      same pool; pick max P(score >= field 99th percentile)
  sim_gpp_st   sim_gpp with stack2bb rules on every candidate
  oracle       hindsight optimum (actual points), the ceiling

Metrics per strategy over test slates: actual points, field percentile, cash rate (beat field median),
top-10% and top-1% rates, and "wins" (beat every field lineup).

    python scripts/dfs_backtest.py                # all test seasons, 12 processes
    python scripts/dfs_backtest.py --seasons 2024 --field 300 --sims 60    # quick
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import norm as N

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfs_optimizer import Slate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dfs")
NBIN = 10
EVAL_SIMS = 2000
LAMBDAS = (0.2, 0.35, 0.5)


# ---------------------------------------------------------------- outcome model
def roles(d):
    """Role by projection rank within team-week-position."""
    d = d.copy()
    d["rk"] = d.groupby(["season", "week", "team", "pos"]).proj.rank(ascending=False, method="first").astype(int)
    cap = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    d["role"] = [f"{p}{r}" if r <= cap[p] else f"OTH_{p}" for p, r in zip(d.pos, d.rk)]
    return d


def fit(train):
    """Residual quantile tables per position x projection decile, and role-pair copula correlations."""
    t = train[train.has_proj].copy()
    t["res"] = t.actual - t.proj
    edges, resid = {}, {}
    for pos, x in t.groupby("pos"):
        e = np.unique(np.quantile(x.proj, np.linspace(0, 1, NBIN + 1)))
        e[0], e[-1] = -np.inf, np.inf
        edges[pos] = e
        b = np.clip(np.searchsorted(e, x.proj, side="right") - 1, 0, len(e) - 2)
        resid[pos] = [np.sort(x.res.values[b == k]) for k in range(len(e) - 1)]
        t.loc[x.index, "bin"] = b
    # normal scores of residuals within (pos, bin)
    t["z"] = t.groupby(["pos", "bin"]).res.rank(pct=True)
    t["z"] = N.ppf(t.z.clip(0.001, 0.999) * 0.998 + 0.001)
    t = roles(t)
    key = ["season", "week", "game_id"]
    a = t[key + ["team", "role", "z"]]
    m = a.merge(a, on=key, suffixes=("_a", "_b"))
    m = m[(m.role_a != m.role_b) | (m.team_a != m.team_b)]
    m["side"] = np.where(m.team_a == m.team_b, "own", "opp")
    # OTH roles can hold several players: their own-team pairs among themselves are still pairs
    corr = {}
    for (ra, rb, side), x in m.groupby(["role_a", "role_b", "side"]):
        if len(x) < 150: continue
        r = np.corrcoef(x.z_a, x.z_b)[0, 1]
        corr[(ra, rb, side)] = r * len(x) / (len(x) + 300)          # shrink thin pairs toward 0
    return dict(edges=edges, resid=resid, corr=corr)


def game_corr(sl, model):
    """Correlation matrix for one slate's players (block-diagonal by game), projected to PSD."""
    n = len(sl); C = np.eye(n)
    g = sl.game_id.values; team = sl.team.values; role = sl.role.values
    for gid in np.unique(g):
        ix = np.where(g == gid)[0]
        for i in ix:
            for j in ix:
                if i >= j: continue
                side = "own" if team[i] == team[j] else "opp"
                r = model["corr"].get((role[i], role[j], side), 0.0)
                C[i, j] = C[j, i] = r
    w, v = np.linalg.eigh(C)
    if w.min() < 1e-6:
        C = (v * np.clip(w, 1e-6, None)) @ v.T
        s = np.sqrt(np.diag(C)); C = C / np.outer(s, s)
    return C


def simulate(sl, model, C, k, rng):
    """k x n simulated DK points."""
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((k, len(sl))) @ L.T
    u = N.cdf(z)
    out = np.empty_like(u)
    for j, (pos, p) in enumerate(zip(sl.pos.values, sl.proj.values)):
        e = model["edges"][pos]
        b = int(np.clip(np.searchsorted(e, p, side="right") - 1, 0, len(e) - 2))
        r = model["resid"][pos][b]
        out[:, j] = p + r[np.minimum((u[:, j] * len(r)).astype(int), len(r) - 1)]
    # no projection -> not expected to play: mostly 0
    out[:, ~sl.has_proj.values] = np.where(rng.random((k, (~sl.has_proj.values).sum())) < 0.85, 0.0,
                                           out[:, ~sl.has_proj.values])
    return out


# ---------------------------------------------------------------- one slate
def run_slate(args):
    try:
        return _run_slate(args)
    except Exception as e:                                             # one bad slate must not sink the run
        sl = args[0]
        print(f"  slate {sl.season.iat[0]} wk{sl.week.iat[0]} failed: {type(e).__name__}: {e}", flush=True)
        return []


def _run_slate(args):
    sl, model, field_n, sims, seed = args
    rng = np.random.default_rng(seed)
    sl = roles(sl).reset_index(drop=True)
    # drop players nobody would consider (keeps the integer programs small)
    sl = sl[(sl.proj > 0.5) | (sl.pos == "DST")].reset_index(drop=True)
    base = Slate(sl); stk1 = Slate(sl, stack=1); stk2 = Slate(sl, stack=2, bringback=1, no_dst_vs=True)
    proj, act = sl.proj.values, sl.actual.values
    lineups = {}
    lineups["mean"] = base.solve(proj)
    lineups["stack1"] = stk1.solve(proj)
    lineups["stack2bb"] = stk2.solve(proj)
    lineups["oracle"] = base.solve(act)
    # field
    field = []
    for f in range(field_n):
        view = proj * np.exp(rng.normal(0, 0.30, len(proj)))
        idx = (stk1 if f % 2 else base).solve(view)
        if idx is not None: field.append(idx)
    F = np.zeros((len(field), len(sl))); [F.__setitem__((i, idx), 1) for i, idx in enumerate(field)]
    # simulation candidates
    C = game_corr(sl, model)
    worlds = simulate(sl, model, C, sims, rng)
    # candidates: optima of worlds pulled only partway from the projection (a full world's optimum is
    # mostly luck and averages ~20 pts below the mean lineup), plus a stack around each top QB
    pool, pool_st = {}, {}
    for k, w in enumerate(worlds):
        lam = LAMBDAS[k % len(LAMBDAS)]
        v = proj + lam * (w - proj)
        a = base.solve(v); b = stk2.solve(v)
        if a is not None: pool[tuple(a)] = a
        if b is not None: pool_st[tuple(b)] = b
    qbs = sl.index[sl.pos == "QB"][np.argsort(-proj[sl.pos.values == "QB"])][:12]
    for q in qbs:
        bump = proj.copy(); bump[q] += 1000                                # force this QB
        b = stk2.solve(bump)                                               # None: team cannot be stacked
        if b is not None: pool_st[tuple(b)] = b; pool[tuple(b)] = b
    for k in ("mean", "stack2bb"):
        if lineups[k] is not None: pool[tuple(lineups[k])] = lineups[k]
    if lineups["stack2bb"] is not None: pool_st[tuple(lineups["stack2bb"])] = lineups["stack2bb"]
    ev = simulate(sl, model, C, EVAL_SIMS, rng)                       # fresh worlds for selection
    fs = ev @ F.T                                                      # field scores per world
    med = np.median(fs, 1); p99 = np.quantile(fs, 0.99, 1)

    def pick(cands, crit):
        best, bv = None, -np.inf
        for idx in cands.values():
            s = ev[:, idx].sum(1)
            v = s.mean() if crit == "mean" else np.mean(s >= (med if crit == "cash" else p99))
            if v > bv: best, bv = idx, v
        return best
    lineups["sim_mean"] = pick(pool, "mean")
    lineups["sim_cash"] = pick(pool, "cash")
    lineups["sim_gpp"] = pick(pool, "gpp")
    lineups["sim_gpp_st"] = pick(pool_st, "gpp")
    # grade on actual points
    field_act = F @ act
    rows = []
    for k, idx in lineups.items():
        if idx is None: continue
        s = act[idx].sum()
        rows.append(dict(season=sl.season.iat[0], week=sl.week.iat[0], strategy=k, actual=s, proj=proj[idx].sum(),
                         salary=sl.salary.values[idx].sum(), pct=np.mean(field_act < s),
                         cash=s >= np.median(field_act), top10=s >= np.quantile(field_act, 0.9),
                         top1=s >= np.quantile(field_act, 0.99), win=s > field_act.max(),
                         n_teams=len(set(sl.team.values[idx])),
                         stacked=int(any((sl.team.values[idx] == sl.team.values[i]).sum() >= 2
                                         for i in idx if sl.pos.values[i] == "QB")),
                         lineup="|".join(sl.player.values[idx])))
    rows.append(dict(season=sl.season.iat[0], week=sl.week.iat[0], strategy="field_median",
                     actual=np.median(field_act), pct=0.5))
    rows.append(dict(season=sl.season.iat[0], week=sl.week.iat[0], strategy="field_p99",
                     actual=np.quantile(field_act, 0.99), pct=0.99))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2019-2025")
    ap.add_argument("--field", type=int, default=1000)
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--tag", default="")
    A = ap.parse_args()
    a, b = (A.seasons.split("-") + [A.seasons])[:2]
    seasons = range(int(a), int(b) + 1)
    d = pd.read_parquet(os.path.join(OUT, "slates.parquet"))
    jobs = []
    for y in seasons:
        model = fit(d[d.season < y])
        for w, sl in d[d.season == y].groupby("week"):
            jobs.append((sl, model, A.field, A.sims, y * 100 + w))
    t0 = time.time()
    with Pool(A.procs) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(run_slate, jobs)):
            res += r
            if (i + 1) % 20 == 0: print(f"  {i + 1}/{len(jobs)} slates, {time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(res)
    for c in ("cash", "top10", "top1", "win", "stacked"): r[c] = r[c].astype(float)
    path = os.path.join(OUT, f"backtest{A.tag}.parquet"); r.to_parquet(path, index=False)
    summ = r.groupby("strategy").agg(slates=("actual", "size"), actual=("actual", "mean"), pct=("pct", "mean"),
                                     cash=("cash", "mean"), top10=("top10", "mean"), top1=("top1", "mean"),
                                     wins=("win", "sum"), stacked=("stacked", "mean"))
    print(summ.sort_values("actual", ascending=False).round(3).to_string())
    print(f"-> {os.path.relpath(path, ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
