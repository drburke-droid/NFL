"""
Paired comparison of backtest strategies (outputs/dfs/backtest.parquet from dfs_backtest.py).

Every strategy played the same slates, so differences are taken slate by slate and bootstrapped
over slates (95% interval). Prints the summary table, per-season actual points, and each strategy
against `mean` (the max-projection lineup).

    python scripts/dfs_backtest_report.py [--tag _x]
"""
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--vs", default="mean")
    A = ap.parse_args()
    r = pd.read_parquet(os.path.join(ROOT, "outputs", "dfs", f"backtest{A.tag}.parquet"))
    s = r[~r.strategy.str.startswith("field")]
    print(f"{s.groupby(['season', 'week']).ngroups} slates, seasons {r.season.min()}-{r.season.max()}\n")
    summ = s.groupby("strategy").agg(actual=("actual", "mean"), proj=("proj", "mean"), field_pct=("pct", "mean"),
                                     cash=("cash", "mean"), top10=("top10", "mean"), top1=("top1", "mean"),
                                     wins=("win", "sum"), stacked=("stacked", "mean"), teams=("n_teams", "mean"))
    f = r[r.strategy.str.startswith("field")].groupby("strategy").actual.mean()
    print(summ.sort_values("actual", ascending=False).round(3).to_string())
    print(f"\nfield median {f.get('field_median', np.nan):.1f}, field 99th pct {f.get('field_p99', np.nan):.1f}"
          "  (random baseline: cash 0.50, top10 0.10, top1 0.01)\n")
    print("actual points by season:")
    print(s.pivot_table(index="strategy", columns="season", values="actual").round(1).to_string(), "\n")
    w = s.pivot_table(index=["season", "week"], columns="strategy", values=["actual", "cash", "top10", "top1"])
    print(f"paired vs '{A.vs}' (mean difference per slate, 95% bootstrap interval):")
    for k in sorted(s.strategy.unique()):
        if k in (A.vs, "oracle"): continue
        out = []
        for m in ("actual", "cash", "top10", "top1"):
            d, lo, hi = boot(w[m][k] - w[m][A.vs])
            out.append(f"{m} {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"  {k:11s} " + "   ".join(out))


if __name__ == "__main__":
    main()
