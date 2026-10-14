"""
How close can a pregame lineup get to the hindsight-optimal (oracle) DraftKings lineup?

Target = lineup actual points / oracle actual points on the same slate. The oracle is fixed per slate,
so maximising expected % of oracle is maximising expected points: the lever is the projection the
optimizer is fed. Each projection below goes through the same DK Classic integer program
(dfs_optimizer.Slate, no stacking rules); the oracle is solved on the full slate.

Pregame projections (walk-forward: fitted on earlier seasons only, test 2018-25):
  ffa        FFA consensus, DK-scored (dfs_slates.py)
  ffa_cal    per-position linear recalibration of ffa
  salary     per-position linear fit on DK salary alone (DK's price is its own projection)
  ffa_sal    per-position linear: ffa + salary
  ffa_sal_v  per-position linear: ffa + salary + implied team total + spread
  gbm        LightGBM on ffa, salary, implied, spread, position, has-projection, salary rank in team-pos
Ceilings / floors (not pregame):
  loo_mean   player's mean DK points over his OTHER main-slate weeks that season (knowing true level)
  random     a random legal lineup (average of 20 per slate)

    python scripts/dfs_oracle_gap.py      # -> outputs/dfs/oracle_gap.parquet + table
"""
import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfs_optimizer import Slate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dfs")
TEST = range(2018, 2026)


def features(d):
    d = d.copy()
    d["sal_k"] = d.salary / 1000
    d["implied"] = d.implied.fillna(22.0); d["spread"] = d.spread.fillna(0.0)
    d["sal_rank"] = d.groupby(["season", "week", "team", "pos"]).salary.rank(ascending=False)
    d["pos_code"] = d.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 4})
    d["hp"] = d.has_proj.astype(float)
    return d


def linear(train, test, cols):
    out = np.zeros(len(test))
    for pos in test.pos.unique():
        tr = train[train.pos == pos]; te = test.pos.values == pos
        X = np.c_[np.ones(len(tr)), tr[cols].values]
        beta = np.linalg.lstsq(X, tr.actual.values, rcond=None)[0]
        out[te] = np.c_[np.ones(te.sum()), test.loc[te, cols].values] @ beta
    return out


def gbm(train, test):
    import lightgbm as lgb
    cols = ["proj", "sal_k", "implied", "spread", "pos_code", "hp", "sal_rank"]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(train[cols], train.actual, categorical_feature=["pos_code"])
    return m.predict(test[cols])


def main():
    t0 = time.time()
    d = features(pd.read_parquet(os.path.join(OUT, "slates.parquet")))
    d = d.groupby(["season", "week"]).filter(lambda s: (s.pos == "DST").any())
    # leave-one-out season mean (ceiling)
    g = d.groupby(["season", "player", "team"]).actual
    s, n = g.transform("sum"), g.transform("count")
    d["loo_mean"] = np.where(n > 1, (s - d.actual) / (n - 1).clip(lower=1), 0.0)
    preds = []
    for y in TEST:
        tr, te = d[d.season < y], d[d.season == y].copy()
        te["ffa"] = te.proj
        te["ffa_cal"] = linear(tr, te, ["proj"])
        te["salary_p"] = linear(tr, te, ["sal_k"])
        te["ffa_sal"] = linear(tr, te, ["proj", "sal_k"])
        te["ffa_sal_v"] = linear(tr, te, ["proj", "sal_k", "implied", "spread"])
        te["gbm"] = gbm(tr, te)
        preds.append(te)
    p = pd.concat(preds)
    P = ["ffa", "ffa_cal", "salary_p", "ffa_sal", "ffa_sal_v", "gbm", "loo_mean"]
    print("projection accuracy on test rows (all slate players):")
    for k in P:
        print(f"  {k:10s} MAE {np.mean(np.abs(p[k] - p.actual)):.3f}   r {np.corrcoef(p[k], p.actual)[0, 1]:.3f}")
    rng = np.random.default_rng(0)
    rows = []
    for (y, w), sl in p.groupby(["season", "week"]):
        sl = sl.reset_index(drop=True); S = Slate(sl); act = sl.actual.values
        orc = act[S.solve(act)].sum()
        r = dict(season=y, week=w, oracle=orc)
        for k in P:
            idx = S.solve(sl[k].values); r[k] = act[idx].sum()
            r[k + "_proj"] = sl[k].values[idx].sum()
        r["random"] = np.mean([act[S.solve(rng.random(len(sl)) * sl.salary.values / 1000)].sum() for _ in range(20)])
        rows.append(r)
    r = pd.DataFrame(rows); r.to_parquet(os.path.join(OUT, "oracle_gap.parquet"), index=False)
    print(f"\n{len(r)} slates {min(TEST)}-{max(TEST)}; oracle mean {r.oracle.mean():.1f} pts\n")
    print(f"{'lineup from':12s} {'pts':>6s} {'%oracle':>8s} {'median':>7s} {'p10':>6s} {'p90':>6s}   vs ffa (95% CI)")
    rng = np.random.default_rng(1)
    for k in P + ["random"]:
        f = r[k] / r.oracle
        dlt = f - r.ffa / r.oracle
        bs = rng.choice(dlt.values, (4000, len(dlt))).mean(1)
        print(f"{k:12s} {r[k].mean():6.1f} {f.mean():8.1%} {f.median():7.1%} {f.quantile(.1):6.1%} {f.quantile(.9):6.1%}"
              f"   {dlt.mean():+.1%} [{np.quantile(bs, .025):+.1%},{np.quantile(bs, .975):+.1%}]")
    print("\n% of oracle by season:")
    print(pd.DataFrame({k: (r[k] / r.oracle).groupby(r.season).mean() for k in P + ["random"]}).T.round(3).to_string())
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
