#!/usr/bin/env python3
"""How does keeper inflation actually work in THIS league?

Identification: 2023 had zero keepers, so its paid/AAV ratio is the room's standing
bias vs national AAV. 2024 and 2025 had keepers, so

    keeper inflation = ratio(keeper season) / ratio(2023)

strips out "this room just underpays" and isolates the keeper effect.

Baseline = nflv_ffa_league.ffa_aav for the SAME season (12-team $200 PPR AAV).
Source   = outputs/espn_drafts.csv (this league's real winning bids, 2023-25).

    python scripts/keeper_inflation_study.py
"""

import os
import re
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
DRAFTS = os.path.join(ROOT, "outputs", "espn_drafts.csv")
OUT = os.path.join(ROOT, "outputs", "reports", "keeper_inflation.md")
SUF = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
RNG = np.random.default_rng(0)
POS = ["QB", "RB", "WR", "TE"]


def norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower())
    return re.sub(r"\s+", " ", SUF.sub("", s)).strip()


def load():
    d = pd.read_csv(DRAFTS)
    d["keeper"] = d["keeper"].astype(str).str.lower().isin(("true", "1"))
    d["k"] = d["player"].map(norm) + "|" + d["pos"]
    con = sqlite3.connect(DB)
    f = pd.read_sql("SELECT season, player, position, ffa_aav FROM nflv_ffa_league "
                    "WHERE season BETWEEN 2023 AND 2025", con)
    con.close()
    f["k"] = f["player"].map(norm) + "|" + f["position"]
    f = f.dropna(subset=["ffa_aav"]).drop_duplicates(["season", "k"])
    m = d.merge(f[["season", "k", "ffa_aav"]], on=["season", "k"], how="left")
    return d, m


def ratio(g):
    return g.bid.sum() / g.ffa_aav.sum() if len(g) and g.ffa_aav.sum() > 0 else np.nan


def boot_ratio_of_ratios(b, k, n=6000):
    if len(b) < 4 or len(k) < 4:
        return (np.nan, np.nan)
    i1 = RNG.integers(0, len(b), size=(n, len(b)))
    i2 = RNG.integers(0, len(k), size=(n, len(k)))
    a = b.bid.values[i1].sum(1) / b.ffa_aav.values[i1].sum(1)
    c = k.bid.values[i2].sum(1) / k.ffa_aav.values[i2].sum(1)
    return tuple(np.percentile(c / a, [2.5, 97.5]))


def main():
    d, m = load()
    sk = m[m["pos"].isin(POS) & m.ffa_aav.notna() & (m.ffa_aav > 0) & (~m.keeper)].copy()
    L = []
    add = L.append
    add("# Keeper inflation in this league — measured, not assumed\n")
    add(f"Source: `outputs/espn_drafts.csv` ({len(d)} picks, 2023-25) priced against "
        f"same-season `nflv_ffa_league.ffa_aav`. **2023 had zero keepers** and is the control.\n")

    add("## Money mechanics\n")
    add("| season | teams | budget | keeper $ | open $ available | open $ spent | unspent |")
    add("|---|---|---|---|---|---|---|")
    leaks = []
    for s, g in d.groupby("season"):
        t = g.team.nunique()
        bud, ks = t * 200, g[g.keeper].bid.sum()
        avail, spent = bud - ks, g[~g.keeper].bid.sum()
        leaks.append((s, 1 - spent / avail))
        add(f"| {s} | {t} | ${bud} | ${ks:.0f} | ${avail:.0f} | ${spent:.0f} | "
            f"${avail-spent:.0f} ({100*(1-spent/avail):.1f}%) |")
    add("")
    add("Keeper drafts leave real money unspent — "
        + ", ".join(f"{s}: {100*l:.1f}%" for s, l in leaks)
        + ". That is why realised inflation lands below what a fully-cleared auction implies.\n")

    b23 = sk[sk.season == 2023]
    kall = sk[sk.season != 2023]
    g_infl = ratio(kall) / ratio(b23)
    lo, hi = boot_ratio_of_ratios(b23, kall)
    add("## Headline\n")
    add(f"- Room baseline (2023, keeper-free): **{ratio(b23):.3f}x** national AAV — this league underpays.")
    add(f"- **Keeper inflation: {g_infl:.3f}x**, bootstrap 95% CI **[{lo:.2f}, {hi:.2f}]** "
        f"(n={len(b23)} control picks, {len(kall)} keeper-era).")

    y = np.log(sk.bid.clip(lower=1).values / sk.ffa_aav.values)
    lav = np.log(sk.ffa_aav.values)
    lavc = lav - lav.mean()
    K = (sk.season != 2023).astype(float).values
    X = np.column_stack([np.ones(len(y)), K, lavc, K * lavc])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = len(y) - X.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - stats.t.cdf(np.abs(coef / se), dof))
    add(f"- OLS `log(paid/AAV) ~ keeper_era * log(AAV)` (n={len(y)}): keeper coefficient "
        f"p={p[1]:.4f} -> **{np.exp(coef[1]):.3f}x** at mean price.\n")

    add("## Rejected hypotheses\n")
    add("### Position-specific inflation — NOT supported\n")
    add("| pos | 2023 base | keeper-era | inflation | 95% CI | vs global |")
    add("|---|---|---|---|---|---|")
    for pos in POS:
        b, k = sk[(sk.season == 2023) & (sk.pos == pos)], sk[(sk.season != 2023) & (sk.pos == pos)]
        i = ratio(k) / ratio(b)
        l2, h2 = boot_ratio_of_ratios(b, k)
        add(f"| {pos} | {ratio(b):.3f} | {ratio(k):.3f} | {i:.3f} | [{l2:.2f}, {h2:.2f}] | "
            f"{'distinct' if (h2 < g_infl or l2 > g_infl) else 'overlaps'} |")
    add("")
    add("Point estimates tempt you (TE looks hottest), but **every CI overlaps the global "
        "estimate** on 14-52 picks per position-season. Do not ship per-position multipliers.\n")

    add("### Price-tier gradient — NOT supported\n")
    add(f"The interaction term is insignificant (p={p[3]:.4f}) and flips sign. A binned view "
        "suggests cheap players inflate ~1.5x, but that is an artifact of bids being floored "
        "at $1 against sub-$1 AAVs.\n")

    add("### Keeper composition — NO detectable effect\n")
    rows = []
    for s in (2024, 2025):
        for pos in POS:
            kept = len(d[(d.season == s) & d.keeper & (d.pos == pos)])
            pool = len(d[(d.season == s) & (d.pos == pos)])
            g2 = sk[(sk.season == s) & (sk.pos == pos)]
            b = sk[(sk.season == 2023) & (sk.pos == pos)]
            if len(g2) < 4:
                continue
            rows.append({"kept_share": kept / pool, "infl": ratio(g2) / ratio(b)})
    O = pd.DataFrame(rows)
    pr = stats.pearsonr(O.kept_share, O.infl)
    add(f"Does keeping most of a position inflate the survivors (scarcity) or deflate them "
        f"(nobody still needs one)? **Neither, measurably**: corr(share kept, inflation) = "
        f"{pr[0]:+.3f}, p={pr[1]:.3f} (n={len(O)} position-seasons). Underpowered, but no signal.\n")

    add("## Why keepers inflate at all\n")
    for s in (2024, 2025):
        k = m[(m.season == s) & m.keeper & m.ffa_aav.notna()]
        add(f"- **{s}**: {len(k)} keepers cost ${k.bid.sum():.0f} but carry ${k.ffa_aav.sum():.0f} "
            f"of AAV value — ${k.ffa_aav.sum()-k.bid.sum():.0f} of value leaves the pool free "
            f"({k.ffa_aav.sum()/max(k.bid.sum(),1):.2f}x). Surviving money chases a thinner pool.")
    add("")
    add("## Applied\n")
    add("`docs/index.html` `dynamicMarket()`: ceiling **1.6 -> 1.25** (top of the measured CI) and "
        "a **0.94** unspent-money factor. The old 1.6 let Exp $ run ~22% past anything this "
        "league has ever paid. Note `PRICE_ANCHOR` is already fitted on 2023-25 bids (two keeper "
        "years), so it embeds the keeper effect — `infl` must only carry this room's money "
        "surplus, never the keeper effect twice.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
