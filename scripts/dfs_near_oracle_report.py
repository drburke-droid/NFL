"""
Report on dfs_near_oracle.py output: what pregame traits separate rosters within 5% of the oracle from
(a) random legal rosters and (b) the 150 rosters our jitter-15 method builds. Each slate is weighted
equally (trait means are taken within slate first). Also a player-level view: which pregame player
traits predict being in near-oracle rosters.

    python scripts/dfs_near_oracle_report.py
"""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "outputs", "dfs", "near_oracle")
NUM = ["proj_frac", "salary", "stack", "bringback", "max_game", "games", "dst_vs", "punts", "qb_imp",
       "top_game", "proj_rank", "value_rank", "chalk_n"]


def main():
    f = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(CK, "*.parquet")) if "_players" not in p])
    pl = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(CK, "*_players.parquet"))])
    f["stack2"] = (f["stack"] >= 2).astype(float); f["stack1"] = (f["stack"] >= 1).astype(float)
    f["bb1"] = (f.bringback >= 1).astype(float); f["dst_conflict"] = (f.dst_vs > 0).astype(float)
    for p in ("RB", "WR", "TE"): f[f"flex_{p}"] = (f.flex == p).astype(float)
    f["sal_49_5"] = (f.salary >= 49500).astype(float)
    cols = NUM + ["stack1", "stack2", "bb1", "dst_conflict", "flex_RB", "flex_WR", "flex_TE", "sal_49_5"]
    slate = f.groupby(["season", "week", "set"])[cols].mean()
    m = slate.groupby("set").mean().T[["near", "ours", "random"]]
    # paired near - ours across slates, and how many slates agree in sign
    w = slate.unstack("set")
    rows = []
    for c in cols:
        dl = (w[c]["near"] - w[c]["ours"]).dropna()
        rng = np.random.default_rng(0); bs = rng.choice(dl.values, (3000, len(dl))).mean(1)
        rows.append((c, dl.mean(), np.quantile(bs, .025), np.quantile(bs, .975), (dl > 0).mean()))
    m["near-ours"] = [r[1] for r in rows]; m["ci_lo"] = [r[2] for r in rows]; m["ci_hi"] = [r[3] for r in rows]
    m["slates near>ours"] = [r[4] for r in rows]
    n = f[f.set == "near"].groupby(["season", "week"]).n_near.first()
    print(f"{n.size} slates; near-oracle rosters per slate: median {n.median():.0f}, range {n.min()}-{n.max()}, total {n.sum():,}")
    sc = f.groupby("set").score.mean()
    print(f"mean % of oracle: near {sc['near']:.3f}, ours {sc['ours']:.3f}, random {sc['random']:.3f}\n")
    print(m.round(3).to_string())
    # what share of OUR rosters / random rosters are near-oracle
    print("\nshare of rosters that are within 5% of oracle:  ours "
          f"{(f[f.set == 'ours'].score >= .95).mean():.4f}   random {(f[f.set == 'random'].score >= .95).mean():.6f}")
    # player level
    pl["value"] = pl.proj / pl.salary * 1000
    pl["proj_rk"] = pl.groupby(["season", "week", "pos"]).proj.rank(ascending=False)
    pl["val_rk"] = pl.groupby(["season", "week", "pos"]).value.rank(ascending=False)
    pl["sal_band"] = pd.cut(pl.salary, [0, 3500, 4500, 5500, 6500, 7500, 20000],
                            labels=["<3.5k", "3.5-4.5k", "4.5-5.5k", "5.5-6.5k", "6.5-7.5k", "7.5k+"])
    print("\nplayer level: share of near-oracle rosters containing a player, by position x salary band (mean over players)")
    print(pl.pivot_table(index="pos", columns="sal_band", values="near_share", aggfunc="mean", observed=False).round(3).to_string())
    print("\nprobability a player is in the ORACLE lineup, by projection rank within position on the slate")
    pl["rk_band"] = pd.cut(pl.proj_rk, [0, 1, 3, 6, 10, 20, 999], labels=["1", "2-3", "4-6", "7-10", "11-20", "21+"])
    print(pl.pivot_table(index="pos", columns="rk_band", values="oracle_pick", aggfunc="mean", observed=False).round(3).to_string())
    pl["vrk_band"] = pd.cut(pl.val_rk, [0, 1, 3, 6, 10, 20, 999], labels=["1", "2-3", "4-6", "7-10", "11-20", "21+"])
    print("\n... by projected VALUE (pts per $1k) rank within position")
    print(pl.pivot_table(index="pos", columns="vrk_band", values="oracle_pick", aggfunc="mean", observed=False).round(3).to_string())
    pl["imp_band"] = pd.qcut(pl.implied, 4, labels=["low", "mid-", "mid+", "high"])
    print("\n... by team implied total quartile")
    print(pl.pivot_table(index="pos", columns="imp_band", values="oracle_pick", aggfunc="mean", observed=False).round(3).to_string())
    out = os.path.join(ROOT, "outputs", "dfs", "near_oracle_summary.csv"); m.to_csv(out)
    print(f"\n-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
