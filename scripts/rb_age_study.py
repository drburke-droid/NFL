#!/usr/bin/env python3
"""RB age: where FFA and our model are wrong, and what is actually exploitable.

Records BOTH results, because the negative one matters:

  REJECTED - an age multiplier on RB projections. Fitted walk-forward it made MAE
             4.8% WORSE, and no shrinkage dose beat the current board.
  SHIPPED  - an age TIEBREAKER. Between two RBs projected within 10% and 4+ years
             apart, the younger outscored the older 59.0% of the time.

The bias is real in aggregate but swamped by RB variance at the player level: it
survives as an ordering rule and dies as a number. Do not re-add the multiplier.

    python scripts/rb_age_study.py
"""

import os
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
OUT = os.path.join(ROOT, "outputs", "reports", "rb_age.md")
CLIP = (0.85, 1.20)


def load(con):
    p = pd.read_sql("""SELECT season, player_id, position, pred_ppg, next_ppg, next_pos_finish
                       FROM season_ffa_predictions""", con).dropna(subset=["pred_ppg", "next_ppg"])
    ag = pd.read_sql("SELECT season, player_id, age FROM season_dataset WHERE age IS NOT NULL",
                     con).drop_duplicates(["season", "player_id"])
    d = p.merge(ag, on=["season", "player_id"], how="inner")
    return d[(d.pred_ppg > 0) & (d.next_ppg >= 0)].copy()


def headtohead(d, pos, minproj=5.0, pct=0.10, gap=4):
    g = d[(d.position == pos) & (d.pred_ppg > minproj)]
    wins = n = 0
    for _, gs in g.groupby("season"):
        a = gs.reset_index(drop=True)
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                x, y = a.iloc[i], a.iloc[j]
                if abs(x.pred_ppg - y.pred_ppg) / max(x.pred_ppg, y.pred_ppg) > pct:
                    continue
                if abs(x.age - y.age) < gap or x.next_ppg == y.next_ppg:
                    continue
                yng, old = (x, y) if x.age < y.age else (y, x)
                n += 1
                wins += yng.next_ppg > old.next_ppg
    if n < 30:
        return n, wins, np.nan
    return n, wins, stats.binomtest(wins, n, 0.5).pvalue


def main():
    con = sqlite3.connect(DB)
    d = load(con)
    rb = d[d.position == "RB"].copy()
    rb["rel"] = (rb.next_ppg - rb.pred_ppg) / rb.pred_ppg
    L = []
    add = L.append
    add("# RB age — one rejected fix, one shipped\n")
    add(f"Walk-forward on `season_ffa_predictions` joined to `season_dataset.age` "
        f"({len(rb)} RB player-seasons, {rb.season.min()}-{rb.season.max()}).\n")

    full = stats.linregress(rb.age.values, rb.rel.values)
    add("## The bias is real\n")
    add(f"Our model's RB relative error drifts **{100*full.slope:+.2f}% per year of age** "
        f"(p={full.pvalue:.4f}) — young RBs under-projected, 28+ over-projected. The model "
        f"inherits this from FFA almost exactly (FFA -1.93%/yr), so Blend does not fix it.\n")

    add("## REJECTED: an age multiplier on RB projections\n")
    seasons = sorted(rb.season.unique())
    TEST = [s for s in seasons if s >= seasons[0] + 3]
    add("| lambda | mean MAE | vs board | MAE wins | mean rank | rank wins |")
    add("|---|---|---|---|---|---|")
    base = None
    for lam in (0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0):
        maes, rks, w, n, rw, rn = [], [], 0, 0, 0, 0
        for S in TEST:
            tr, te = rb[rb.season < S], rb[rb.season == S]
            if len(tr) < 60 or len(te) < 10:
                continue
            r = stats.linregress(tr.age.values, tr.rel.values)
            mult = np.clip(1 + lam * ((1 + (r.intercept + r.slope * te.age.values)) - 1), *CLIP)
            m0 = (te.next_ppg - te.pred_ppg).abs().mean()
            m1 = (te.next_ppg - te.pred_ppg.values * mult).abs().mean()
            maes.append(m1); n += 1; w += m1 < m0
            fin = te.dropna(subset=["next_pos_finish"])
            if len(fin) >= 12:
                mm = np.clip(1 + lam * ((1 + (r.intercept + r.slope * fin.age.values)) - 1), *CLIP)
                k0 = spearmanr(fin.pred_ppg, -fin.next_pos_finish)[0]
                k1 = spearmanr(fin.pred_ppg.values * mm, -fin.next_pos_finish)[0]
                rks.append(k1); rn += 1; rw += k1 > k0
        mm_, rr_ = float(np.mean(maes)), float(np.mean(rks))
        if lam == 0:
            base = mm_
        add(f"| {lam:.2f} | {mm_:.4f} | {100*(base-mm_)/base:+.2f}% | {w}/{n} | {rr_:.4f} | {rw}/{rn} |")
    add("")
    add("Every dose either does nothing or hurts. **Not shipped.**\n")

    add("## SHIPPED: an age tiebreaker\n")
    add("Two players, same season, projections within 10%, age gap >= 4 years — how often "
        "does the younger outscore the older?\n")
    add("| pos | younger wins | rate | p |")
    add("|---|---|---|---|")
    for pos in ("RB", "QB", "WR", "TE"):
        n, w, p = headtohead(d, pos)
        add(f"| {pos} | {w}/{n} | {100*w/n:.1f}% | {p:.4f} |")
    n, w, p = headtohead(d, "RB", minproj=10.0)
    add("")
    add(f"Among draftable RBs (proj >= 10 PPG) it holds: **{w}/{n} = {100*w/n:.1f}%**, p={p:.4f}.\n")
    add("`docs/index.html` `rbAgeTag()` surfaces this as **⬆ YOUTH** (RB <= 24) and "
        "**⚠ RB AGE** (RB >= 28). Context only — projections and prices are unchanged.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    # console may be cp1252; the file itself is utf-8 and keeps the real glyphs
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
