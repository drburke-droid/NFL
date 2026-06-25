"""
Is there a better DRAFT STRATEGY than our current neutral projections+VORP?

We already know the edge is the projections. Here we keep OUR projections (proj+VORP+
tilt values) and layer each positional strategy on top — QB-heavy, TE-premium, WR-heavy,
RB-heavy, zero-RB, hero-RB, stars-and-scrubs, balanced(current) — then Monte Carlo each
and compare our team's avg finish. Bots + in-season logic held fixed.

Usage: python scripts/test_draft_strategy.py [N]   (default N=60)
"""
import os, sqlite3, sys
import numpy as np
import importlib.util
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(os.path.dirname(__file__), n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V2 = _load("league_sim_v2"); AE = _load("attribute_edge"); L1 = V2.L1
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "DRAFT_STRATEGY.md")

STRATS = ["current(neutral)", "qb_heavy", "elite_te", "te_premium", "wr_heavy", "rb_heavy",
          "robust_rb", "zero_rb", "hero_rb", "mild_stars_scrubs", "extreme_balanced"]
STYLE_OF = {"current(neutral)": "balanced"}      # balanced mult == 1.0 == our current neutral


def styled_val(y, strat):
    base = AE.our_val_for("tilt", y)             # our projections + VORP + leap/fade tilt
    style = STYLE_OF.get(strat, strat)
    order = sorted(base, key=lambda p: -base[p]); orank = {p: i + 1 for i, p in enumerate(order)}
    return {p: base[p] * L1.style_mult(style, y["pos_of"][p], orank[p], p in y["rookies"]) for p in y["pos_of"]}


def run(D, wproj, yd, strat, N):
    run_avg = []
    for run in range(N):
        rng = np.random.default_rng(4000 + run)
        teams = [V2.Team(i, V2.STYLES[i]) for i in range(12)]
        us = next(i for i in range(12) if teams[i].style == "ours")
        fs = []
        for Y in V2.YEARS:
            y = yd[Y]; pv = (y["pos_of"], y["rookies"], y["bot_val"], styled_val(y, strat))
            rank, _ = V2.simulate_year(Y, teams, D, pv, wproj, rng=rng, bid_sigma=0.15)
            fs.append(rank[us])
        run_avg.append(np.mean(fs))
    return np.array(run_avg)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    con = sqlite3.connect(V2.DB); D = V2.load(con); con.close()
    wproj = V2.make_wproj(D)
    print("Precomputing per-year models/values once..."); yd = AE.precompute(D)
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    res = {}
    for s in STRATS:
        res[s] = run(D, wproj, yd, s, N); print(f"  {s} done -> {res[s].mean():.2f}")
    board = sorted(res.items(), key=lambda kv: kv[1].mean())
    L = [f"# Draft-strategy backtest — our projections + each positional tilt (N={N} MC each)\n",
         "Our projection-based values held constant; only the positional strategy layered on top changes. Bots + in-season fixed. Lower avg finish = better.\n",
         "| Strategy (on top of our projections) | Avg finish | 95% CI |", "|---|---|---|"]
    for s, a in board:
        lo, hi = ci(a); L.append(f"| {'**'+s+'**' if s=='current(neutral)' else s} | {a.mean():.2f} | [{lo:.2f}, {hi:.2f}] |")
    cur = res["current(neutral)"].mean(); best = board[0]
    L += ["\n## Read",
          f"- Current neutral (projections + VORP, no positional tilt): **{cur:.2f}**.",
          f"- Best tilt: **{best[0]} = {best[1].mean():.2f}** ({cur-best[1].mean():+.2f} vs current).",
          "- If no tilt beats neutral by more than the CIs overlap, the lesson is: with accurate projections, **positional draft strategies don't add a reliable edge** — take the best value by VORP regardless of position. Forced tilts mostly hurt by making you overpay a position and underpay elsewhere."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("\n" + "\n".join(L)); print("Saved", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
