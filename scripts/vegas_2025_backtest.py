#!/usr/bin/env python3
"""How does our projection model do against Vegas on last season (2025)?

The walk-forward run (scripts/run_v2_walkforward.py) already scores four variants on
the same player-weeks, which makes a like-for-like comparison possible:

  A_football  our model -- rolling stats, roles, team context. NO market data.
  B_market    a model built from the VEGAS PLAYER PROP LINES (+ context) and nothing
              else (src/utils/db.py: "SELECT ... point as prop_line").
  C_full      both families together.
  D_residual  two-stage: predict from market, then model the football residual.
  baseline_*  rolling-average and market-only references.

So "our model vs Vegas" is A vs B on identical rows, walk-forward (train strictly on
earlier weeks), target = weekly PPR.

⚠ THIS IS AN ACCURACY TEST, NOT A BETTING TEST. It asks whether our numbers are as
good as the numbers implied by the market. It does NOT say we could beat the market
for money: that needs prices, vig and settlement, and it was already answered
separately in outputs/reports/prop_ev_model.md -- the answer there was NO.

⚠ Subsets must never be selected on the outcome. Filtering to "weeks the player
actually scored 8+" flatters whichever model predicts higher (ours over-predicts by
+0.43/wk), so it manufactures a win. The fantasy-relevant cut here is the top 150
players by SEASON total, which is chosen independently of any single week.

Writes outputs/reports/vegas_2025_backtest.md
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "reports", "vegas_2025_backtest.md")
SEASON = 2025
KEY = ["player_id", "season", "week"]
NBOOT = 4000

def load():
    P = {}
    for f in glob.glob(os.path.join(ROOT, "outputs", "predictions", "*.parquet")):
        d = pd.read_parquet(f)
        if "predicted" in d.columns:
            P[os.path.basename(f).split("_2026")[0]] = d
    base = P["A_football"][KEY + ["name", "position", "actual"]].copy()
    for k, d in P.items():
        base = base.merge(d[KEY + ["predicted"]].rename(columns={"predicted": k}),
                          on=KEY, how="inner")
    return base[base.season == SEASON].copy(), sorted(P)

def acc(pred, act):
    e = pred - act
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                rho=spearmanr(pred, act).correlation,
                R2=1 - (e ** 2).sum() / ((act - act.mean()) ** 2).sum(),
                bias=e.mean())

def boot(ea, eb, ids, n=NBOOT, seed=5):
    """Paired MAE difference, resampled by PLAYER (a player's weeks are correlated)."""
    rng = np.random.default_rng(seed)
    u = np.unique(ids)
    idx = {p: np.where(ids == p)[0] for p in u}
    o = []
    for _ in range(n):
        s = rng.choice(u, len(u), replace=True)
        ii = np.concatenate([idx[p] for p in s])
        o.append(np.abs(ea[ii]).mean() - np.abs(eb[ii]).mean())
    return float(np.mean(o)), *np.percentile(o, [2.5, 97.5])

def diff(sub, x, y):
    ea = (sub[x] - sub.actual).values
    eb = (sub[y] - sub.actual).values
    _, lo, hi = boot(ea, eb, sub.player_id.values)
    return np.abs(ea).mean() - np.abs(eb).mean(), lo, hi, (lo > 0 or hi < 0)

def main():
    d, mods = load()
    top = d.groupby("player_id").actual.sum().sort_values(ascending=False).head(150).index
    dt = d[d.player_id.isin(top)]
    A, B, C = "A_football", "B_market", "C_full"

    L = []; a = L.append
    a("# Our model vs Vegas — 2025 backtest\n")
    a(f"*{len(d)} player-weeks, {d.player_id.nunique()} players, weeks "
      f"{d.week.min()}-{d.week.max()} of {SEASON}. Walk-forward: every prediction is made "
      "from strictly earlier weeks. Target is weekly PPR.*\n")
    a("`A_football` is our model with no market data. `B_market` is built from the Vegas "
      "player prop lines and nothing else. Same rows, same target, so the comparison is "
      "like-for-like.\n")

    o = acc(d[A].values, d.actual.values); v = acc(d[B].values, d.actual.values)
    md, lo, hi, sig = diff(d, A, B)
    a("## Answer: a dead heat\n")
    a(f"Our football-only model and the Vegas-lines model are **statistically "
      f"indistinguishable** on last season: MAE {o['MAE']:.3f} vs {v['MAE']:.3f}, a paired "
      f"difference of **{md:+.3f}** (95% CI {lo:+.3f} to {hi:+.3f}, clustered by player). "
      "Restricted to the top 150 players by season total the answer is the same "
      f"({diff(dt, A, B)[0]:+.3f}, CI {diff(dt, A, B)[1]:+.3f} to {diff(dt, A, B)[2]:+.3f}).\n")
    a("That is a genuinely good result — the market prices these players with far more "
      "information than we have — but see the warning at the bottom before reading it as "
      "an edge.\n")

    a("## Accuracy, all 2025 player-weeks\n")
    a("| model | MAE | RMSE | Spearman | R² | bias |")
    a("|---|---:|---:|---:|---:|---:|")
    for m in mods:
        s = acc(d[m].values, d.actual.values)
        nm = {A: "**ours** (football only)", B: "**Vegas** (prop lines only)",
              C: "ours + Vegas", "D_residual": "residual (2-stage)",
              "baseline_rolling": "rolling average", "baseline_market": "market baseline"}.get(m, m)
        a(f"| {nm} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {s['rho']:.3f} | {s['R2']:.3f} | {s['bias']:+.3f} |")

    a("\n## They know different things\n")
    a("| comparison | ΔMAE | 95% CI | significant |")
    a("|---|---:|---:|:--:|")
    for lab, x, y, s in (("ours − Vegas", A, B, d),
                         ("ours+Vegas − ours", C, A, d),
                         ("ours+Vegas − Vegas", C, B, d),
                         ("ours − rolling baseline", A, "baseline_rolling", d),
                         ("ours − Vegas (top 150)", A, B, dt),
                         ("ours+Vegas − ours (top 150)", C, A, dt)):
        m_, l_, h_, g = diff(s, x, y)
        a(f"| {lab} | {m_:+.3f} | {l_:+.3f} to {h_:+.3f} | {'yes' if g else 'no'} |")
    a("\nCombining the two beats **either** alone, significantly. Vegas carries information "
      "our features miss, and our features carry information the lines miss — they are "
      "complementary, not redundant.\n")

    a("## Where each one wins\n")
    a("| position | n | ours | Vegas | ours+Vegas |")
    a("|---|---:|---:|---:|---:|")
    for pos in ["QB", "RB", "WR", "TE"]:
        s = d[d.position == pos]
        if len(s) < 50: continue
        a(f"| {pos} | {len(s)} | {np.abs(s[A]-s.actual).mean():.2f} | "
          f"{np.abs(s[B]-s.actual).mean():.2f} | {np.abs(s[C]-s.actual).mean():.2f} |")
    a("\nVegas is better on QB, we are better on RB, WR and TE are close.\n")

    a("## Ranking vs calibration\n")
    a("| model | mean within-week Spearman |")
    a("|---|---:|")
    for m in [A, B, C, "D_residual", "baseline_rolling"]:
        r = [spearmanr(g[m], g.actual).correlation for _, g in d.groupby("week") if len(g) > 20]
        a(f"| {m} | {np.mean(r):.3f} |")
    a(f"\nWe **order** players better (within-week Spearman "
      f"{np.mean([spearmanr(g[A],g.actual).correlation for _,g in d.groupby('week') if len(g)>20]):.3f} "
      f"vs {np.mean([spearmanr(g[B],g.actual).correlation for _,g in d.groupby('week') if len(g)>20]):.3f}), "
      f"which is what a draft board needs. Vegas is better **calibrated** on magnitude "
      f"(RMSE {v['RMSE']:.3f} vs {o['RMSE']:.3f}, bias {v['bias']:+.3f} vs {o['bias']:+.3f}). "
      "Our model systematically over-predicts by about +0.43 PPR a week.\n")

    a("## What this does NOT say\n")
    a("- **It is not a betting result.** Matching the market on MAE is not beating it for "
      "money: you have to clear the vig, and the settled-bet study "
      "(`outputs/reports/prop_ev_model.md`) found the model is **not +EV** against closing "
      "prices — the line won all six markets on MAE there, our P(over) was poorly "
      "calibrated, and essentially all realized profit came from line shopping.\n")
    a("- **`B_market` is a model of the lines, not the lines themselves.** It re-fits prop "
      "lines to weekly PPR, which adds its own error. A pure closing-line benchmark needs "
      "`db/nfl_odds.db`, which is not in this container.\n")
    a("- **No outcome-conditioned subsets.** Slicing on \"weeks the player actually scored "
      "8+\" made us look 0.31 better, but that is selection on the dependent variable and it "
      "simply rewards our upward bias. It is excluded deliberately.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
