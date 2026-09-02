#!/usr/bin/env python3
"""Stars who came cheap: is there a class of below-average-priced auction buys
that finishes in the top 10% of its position?

The dart studies (`late_breakout_*.py`, `dart_price_band_study.py`) asked whether
a $1-10 flier clears replacement. This asks the bigger question: which buys at
BELOW the average price for their position end up a top-3 QB, top-6 RB, top-9 WR
or top-3 TE -- roughly the top 10% of startable players at each spot.

Universe        skill players in the league-scored FFA preseason file, 2016-2025,
                priced (AAV >= $1) and inside the top 168 by AAV -- the ~14 skill
                players per team a 12-team/16-slot room actually buys.
Price           FFA league AAV, expressed as a RATIO to that season's positional
                mean inside the pool. "Cheap" = ratio < 1.
Star            season total in this league's scoring (full PPR, 6-pt pass TD)
                ranked QB<=3, RB<=6, WR<=9, TE<=3.
Validation      2016-21 vs 2022-25 split, and an out-of-sample replication on
                THIS league's own winning bids (outputs/espn_drafts.csv, 2023-25).

    python scripts/cheap_stars_build.py && python scripts/cheap_stars_study.py
"""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import statsmodels.api as sm
from scipy.stats import binomtest, chi2, fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "outputs", "models", "cheap_stars_frame.pkl")
DRAFTS = os.path.join(ROOT, "outputs", "espn_drafts.csv")
FFA26 = os.path.join(ROOT, "data", "ffanalytics", "FFAn_league", "projections_2026_wk0.csv")
OUT = os.path.join(ROOT, "outputs", "reports", "cheap_stars.md")
CSV26 = os.path.join(ROOT, "outputs", "cheap_stars_2026.csv")

FIRST, LAST = 2016, 2025
POOL = 168                                    # 12 teams x 14 skill slots
STAR = {"QB": 3, "RB": 6, "WR": 9, "TE": 3}   # top ~10% of startable players
START = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}   # projected-starter tier
REPL = {"QB": 13, "RB": 27, "WR": 35, "TE": 13}    # VORP baseline (as in LATE_BREAKOUTS)
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(s):
    s = str(s).lower().strip().replace(".", "").replace("'", "").replace("-", " ").replace(",", "")
    return re.sub(r"\s+", " ", SUFFIX.sub("", s)).strip()


def md(f, *lines):
    f.write("\n".join(lines) + "\n")


def tbl(f, df, floats=None):
    d = df.copy()
    for c in (floats or []):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    md(f, "| " + " | ".join(str(c) for c in d.columns) + " |",
       "|" + "|".join("---" for _ in d.columns) + "|")
    for r in d.itertuples(index=False):
        md(f, "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    md(f, "")


# --------------------------------------------------------------------------- #
def prep():
    f = pd.read_pickle(FRAME)
    f = f[f.season.between(FIRST, LAST)].copy()
    f["aav"] = f.aav.fillna(0)
    f["ov"] = f.groupby("season").aav.rank(ascending=False, method="first")

    rep = {(y, p): float(g[g.pos_rank <= REPL[p]].lpts.min())
           for (y, p), g in f.groupby(["season", "position"])}
    f["vorp"] = f.lpts - [rep[(r.season, r.position)] for r in f.itertuples()]

    # prior-season per-game rank: "how good was he per game last year", which the
    # season-total rank hides for anyone who missed time.
    pr = f[["player_id", "season", "position", "ppg", "games"]].copy()
    pr["season"] += 1
    pr = pr[pr.games >= 4]
    pr["prior_ppg_rank"] = pr.groupby(["season", "position"]).ppg.rank(ascending=False, method="min")
    f = f.merge(pr[["player_id", "season", "prior_ppg_rank"]], on=["player_id", "season"], how="left")

    p = f[(f.ov <= POOL) & (f.aav >= 1)].copy()
    p["pos_avg"] = p.groupby(["season", "position"]).aav.transform("mean")
    p["ratio"] = p.aav / p.pos_avg
    p["star"] = [1 if r.pos_rank <= STAR[r.position] else 0 for r in p.itertuples()]
    p["proj_rank"] = p.groupby(["season", "position"]).ffa_points.rank(ascending=False, method="first")
    p["price_rank"] = p.groupby(["season", "position"]).aav.rank(ascending=False, method="first")
    p["gap"] = p.price_rank - p.proj_rank
    p["starter_proj"] = [1 if r.proj_rank <= START[r.position] else 0 for r in p.itertuples()]
    p["opt"] = p.vorp.clip(lower=0)      # a bust gets cut for a waiver body
    p["band"] = np.where(p.ratio >= 1, "at/above avg",
                         np.where(p.ratio >= 0.5, "50-100% of avg", "under 50% of avg"))
    return f, p


def price_bands(p, bands, by=None):
    rows = []
    for lo, hi, lbl in bands:
        b = p[(p.ratio >= lo) & (p.ratio < hi)]
        if not len(b):
            continue
        rows.append({"price vs pos. avg": lbl, "n": len(b), "avg $": round(b.aav.mean(), 1),
                     "star %": round(100 * b.star.mean(), 1), "stars": int(b.star.sum()),
                     "$ per star": round(b.aav.mean() / max(b.star.mean(), 1e-9)),
                     "P(above repl.)": round(100 * (b.vorp > 0).mean(), 1),
                     "E[val if kept]": round(b.opt.mean())})
    d = pd.DataFrame(rows)
    d["share of stars %"] = (100 * d.stars / max(d.stars.sum(), 1)).round(0).astype(int)
    return d


def feature_scan(c):
    """Every draft-day feature, tested against price + position controls."""
    c = c.copy()
    for x in ("RB", "WR", "TE"):
        c[f"pos_{x}"] = (c.position == x).astype(int)
    c["lr"] = np.log(c.ratio)
    base = ["lr", "pos_RB", "pos_WR", "pos_TE"]
    m0 = sm.Logit(c.star, sm.add_constant(c[base])).fit(disp=0)

    prior_g = np.where(c.season <= 2020, 16, 17) - c.prior_games
    feats = {
        "projected inside starter tier": c.starter_proj,
        "price ranks worse than projection (gap>=5)": (c.gap >= 5).astype(int),
        "age <= 25 and NFL round 1-2": ((c.age.fillna(26) <= 25) & (c.draft_round.fillna(9) <= 2)).astype(int),
        "NFL round 1-2 (any age)": (c.draft_round.fillna(9) <= 2).astype(int),
        "rookie with round 1-2 capital": ((c.rookie == 1) & (c.draft_round.fillna(9) <= 2)).astype(int),
        "years 1-3 of career": c.years_exp.fillna(9).between(1, 3).astype(int),
        "prior year: starter-tier PPG": [1 if (pd.notna(r.prior_ppg_rank) and r.prior_ppg_rank <= START[r.position]) else 0 for r in c.itertuples()],
        "prior year: missed 4+ games": ((c.prior_games.notna()) & (prior_g >= 4)).astype(int),
        "prior year: snap share >= 60%": (c.prior_snap_pct.fillna(0) >= 0.6).astype(int),
        "prior year: target share >= 18%": (c.prior_target_share.fillna(0) >= 0.18).astype(int),
        "changed teams": c.team_change.fillna(0).astype(int),
        "new team vacated 100+ targets": (c.vac_targets.fillna(0) >= 100).astype(int),
        "new team vacated 120+ carries": (c.vac_carries.fillna(0) >= 120).astype(int),
        "age >= 30": (c.age.fillna(26) >= 30).astype(int),
        "FFA ceiling premium (continuous)": ((c.ffa_ceiling - c.ffa_points) / c.ffa_points.replace(0, np.nan)).fillna(0),
        "FFA uncertainty below median": (c.ffa_unc <= c.ffa_unc.median()).astype(int),
    }
    rows = []
    for lbl, ser in feats.items():
        c["_f"] = pd.Series(np.asarray(ser, dtype=float), index=c.index)
        m = sm.Logit(c.star, sm.add_constant(c[base + ["_f"]])).fit(disp=0)
        binary = set(pd.unique(c._f.dropna())) <= {0.0, 1.0}
        rows.append({"draft-day feature": lbl,
                     "n with it": int(c._f.sum()) if binary else len(c),
                     "raw P(star)": round(100 * c[c._f > 0].star.mean(), 1) if binary else np.nan,
                     "odds ratio": round(float(np.exp(m.params["_f"])), 2),
                     "_p": float(chi2.sf(2 * (m.llf - m0.llf), 1))})
    d = pd.DataFrame(rows).sort_values("_p")
    d["p (vs price + position)"] = d._p.map(lambda v: "<0.0001" if v < 1e-4 else f"{v:.4f}")
    return d.drop(columns="_p")


def sensitivity(f):
    """Does the class survive different pools, centres and star cutoffs?"""
    rows = []
    variants = [
        ("as reported: top 168 pool, mean, top 3/6/9/3", 168, "mean", STAR),
        ("pool = top 192 (every drafted body)", 192, "mean", STAR),
        ("pool = top 140 (tighter)", 140, "mean", STAR),
        ("centre = positional MEDIAN, not mean", 168, "median", STAR),
        ("star = top 2/4/6/2 (tighter)", 168, "mean", {"QB": 2, "RB": 4, "WR": 6, "TE": 2}),
        ("star = top 4/8/12/4 (looser)", 168, "mean", {"QB": 4, "RB": 8, "WR": 12, "TE": 4}),
        ("star = top 20% (6/12/18/6)", 168, "mean", {"QB": 6, "RB": 12, "WR": 18, "TE": 6}),
    ]
    for lbl, pool, centre, star in variants:
        q = f[(f.groupby("season").aav.rank(ascending=False, method="first") <= pool) & (f.aav >= 1)].copy()
        q["ctr"] = q.groupby(["season", "position"]).aav.transform(centre)
        q["ratio"] = q.aav / q.ctr
        q["st"] = [1 if r.pos_rank <= star[r.position] else 0 for r in q.itertuples()]
        q["pj"] = q.groupby(["season", "position"]).ffa_points.rank(ascending=False, method="first")
        q["sp"] = [1 if r.pj <= START[r.position] else 0 for r in q.itertuples()]
        c = q[(q.ratio >= .5) & (q.ratio < 1) & (q.sp == 1)]
        o = q[(q.ratio >= .5) & (q.ratio < 1) & (q.sp == 0)]
        pv = fisher_exact([[c.st.sum(), len(c) - c.st.sum()],
                           [o.st.sum(), len(o) - o.st.sum()]])[1]
        rows.append({"variant": lbl, "class n": len(c), "class stars": int(c.st.sum()),
                     "class star %": round(100 * c.st.mean(), 1),
                     "same price, weaker proj. %": round(100 * o.st.mean(), 1),
                     "p": "<0.0001" if pv < 1e-4 else f"{pv:.4f}"})
    return pd.DataFrame(rows)


def league_bids(f):
    """Replicate the finding on this league's own winning bids (2023-25)."""
    if not os.path.exists(DRAFTS):
        return None, None
    d = pd.read_csv(DRAFTS)
    d = d[d.pos.isin(STAR) & ~d.keeper].copy()
    d["nm"] = d.player.map(norm)
    g = f.copy()
    g["nm"] = g.player_display_name.fillna(g.player).map(norm)
    m = d.merge(g[["nm", "position", "season", "pos_rank", "ffa_points"]],
                left_on=["nm", "pos", "season"], right_on=["nm", "position", "season"], how="left")
    m["star"] = [1 if (pd.notna(r.pos_rank) and r.pos_rank <= STAR[r.pos]) else 0 for r in m.itertuples()]
    m["pos_avg"] = m.groupby(["season", "pos"]).bid.transform("mean")
    m["ratio"] = m.bid / m.pos_avg
    m["proj_rank"] = m.groupby(["season", "pos"]).ffa_points.rank(ascending=False, method="first")
    m["starter_proj"] = [1 if (pd.notna(r.proj_rank) and r.proj_rank <= START[r.pos]) else 0
                         for r in m.itertuples()]
    m["band"] = np.where(m.ratio >= 1, "at/above avg",
                         np.where(m.ratio >= 0.5, "50-100% of avg", "under 50% of avg"))
    t = (m.groupby("band").agg(**{"n": ("star", "size"), "stars": ("star", "sum"), "avg bid": ("bid", "mean")})
         .reindex(["under 50% of avg", "50-100% of avg", "at/above avg"]).reset_index())
    t["star %"] = (100 * t.stars / t.n).round(1)
    t["avg bid"] = t["avg bid"].round(1)
    cls = m[(m.ratio >= 0.5) & (m.ratio < 1) & (m.starter_proj == 1)]
    return t, cls


def slate_2026():
    d = pd.read_csv(FFA26)
    d = d[d.position.isin(STAR)].copy()
    d["aav"] = d.aav.fillna(0)
    d["ov"] = d.aav.rank(ascending=False, method="first")
    p = d[(d.ov <= POOL) & (d.aav >= 1)].copy()
    p["pos_avg"] = p.groupby("position").aav.transform("mean")
    p["ratio"] = p.aav / p.pos_avg
    p["proj_rank"] = p.groupby("position").points.rank(ascending=False, method="first")
    p["starter_proj"] = [1 if r.proj_rank <= START[r.position] else 0 for r in p.itertuples()]
    p["gap"] = p.groupby("position").aav.rank(ascending=False, method="first") - p.proj_rank
    cls = p[(p.ratio >= 0.5) & (p.ratio < 1) & (p.starter_proj == 1)].copy()
    cls = cls.sort_values(["position", "aav"], ascending=[True, False])
    return p, cls


# --------------------------------------------------------------------------- #
def main():
    f, p = prep()
    cheap = p[p.ratio < 1].copy()
    base = cheap.star.mean()

    # --- per-position accounting -------------------------------------------
    pos_rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        q = p[p.position == pos]
        ch, ex = q[q.ratio < 1], q[q.ratio >= 1]
        pos_rows.append({"pos": pos, "star = top": STAR[pos],
                         "pos. avg $": round(q.pos_avg.mean(), 1),
                         "cheap buys": len(ch), "cheap stars": int(ch.star.sum()),
                         "P(star) if cheap": round(100 * ch.star.mean(), 1),
                         "P(star) if at/above": round(100 * ex.star.mean(), 1),
                         "% of elite seasons bought cheap":
                             round(100 * ch.star.sum() / max(q.star.sum(), 1))})
    pos_tab = pd.DataFrame(pos_rows)

    bands = [(0, .25, "under 25%"), (.25, .5, "25-50%"), (.5, .75, "50-75%"),
             (.75, 1.0, "75-100%"), (1.0, 1.5, "100-150%"), (1.5, 2.5, "150-250%"),
             (2.5, 99, "250%+")]
    band_tab = price_bands(p, bands)

    scan = feature_scan(cheap)

    # --- the class ----------------------------------------------------------
    grid = (p.groupby(["band", np.where(p.starter_proj == 1, "starter-tier projection",
                                        "outside starter tier")])
            .agg(n=("star", "size"), stars=("star", "sum"), avg=("aav", "mean")))
    grid["star %"] = (100 * grid.stars / grid.n).round(1)
    grid["$ per star"] = [f"{v:.0f}" if k else "—" for v, k in
                          zip(grid.avg / (grid.stars / grid.n).replace(0, np.nan), grid.stars > 0)]
    grid["avg $"] = grid.avg.round(1)
    grid = (grid.drop(columns="avg").reset_index()
            .rename(columns={"level_1": "projection"})
            .sort_values(["band", "projection"]))
    order = {"under 50% of avg": 0, "50-100% of avg": 1, "at/above avg": 2}
    grid = grid.sort_values(["band", "projection"], key=lambda s: s.map(order).fillna(9))

    cls = p[(p.ratio >= 0.5) & (p.ratio < 1) & (p.starter_proj == 1)]
    k, n = int(cls.star.sum()), len(cls)
    ci = binomtest(k, n).proportion_ci()
    peer = p[(p.ratio >= 0.5) & (p.ratio < 1) & (p.starter_proj == 0)]
    pfish = fisher_exact([[k, n - k], [int(peer.star.sum()), len(peer) - int(peer.star.sum())]])[1]

    split = []
    for lbl, sub in (("2016-2021", p[p.season <= 2021]), ("2022-2025", p[p.season >= 2022])):
        for bl, m in (("the class", (sub.ratio >= .5) & (sub.ratio < 1) & (sub.starter_proj == 1)),
                      ("same price, weaker projection", (sub.ratio >= .5) & (sub.ratio < 1) & (sub.starter_proj == 0)),
                      ("under 50% of avg + starter-tier proj.", (sub.ratio < .5) & (sub.starter_proj == 1)),
                      ("under 50% of avg (all)", sub.ratio < .5),
                      ("at/above avg (all)", sub.ratio >= 1)):
            b = sub[m]
            split.append({"era": lbl, "cell": bl, "n": len(b), "stars": int(b.star.sum()),
                          "star %": round(100 * b.star.mean(), 1)})
    split = pd.DataFrame(split)

    per_season = (cls.groupby("season").agg(n=("star", "size"), stars=("star", "sum"))
                  .reset_index())
    per_pos = (cls.groupby("position")
               .agg(**{"n": ("star", "size"), "stars": ("star", "sum"),
                       "avg $": ("aav", "mean")}).round(1).reset_index())
    per_pos["star %"] = (100 * per_pos.stars / per_pos.n).round(1)

    cells = {"the class (50-100% avg + starter-tier proj.)": (p.ratio >= .5) & (p.ratio < 1) & (p.starter_proj == 1),
             "bargain bin (under 50% of avg)": p.ratio < .5,
             "mid market (100-200% of avg)": (p.ratio >= 1) & (p.ratio < 2),
             "studs (200%+ of avg)": p.ratio >= 2}
    econ = []
    for lbl, m in cells.items():
        b = p[m]
        econ.append({"cell": lbl, "per season": round(len(b) / (LAST - FIRST + 1), 1),
                     "avg $": round(b.aav.mean(), 1), "star %": round(100 * b.star.mean(), 1),
                     "$ per star": round(b.aav.mean() / b.star.mean()),
                     "P(above repl.)": round(100 * (b.vorp > 0).mean(), 1),
                     "E[val if kept]": round(b.opt.mean()),
                     "value per $": round(b.opt.mean() / b.aav.mean(), 1)})
    econ = pd.DataFrame(econ)

    sens = sensitivity(f[(f.ov <= 999)].copy())
    bids_tab, bids_cls = league_bids(f)
    pool26, cls26 = slate_2026()

    # --- write --------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as o:
        md(o, "# Stars who came cheap — do below-average buys ever finish top-10%?", "",
           "`scripts/cheap_stars_build.py` (frame) → `scripts/cheap_stars_study.py` (this report).",
           "", "**The question.** Not \"does a $2 dart clear replacement\" (that is",
           "`LATE_BREAKOUTS.md`), but: is there a *class* of player who goes for **less than the",
           "average price at his position** and finishes **top-3 QB / top-6 RB / top-9 WR /",
           "top-3 TE** — the top ~10% of startable players at the spot?", "",
           "## Setup", "",
           f"- **Universe** — {FIRST}-{LAST}, the league-scored FFA preseason (wk0) file, skill",
           f"  positions, AAV ≥ $1, inside the top {POOL} by AAV (the ~14 skill players per team a",
           "  12-team/16-slot room actually buys). 10 seasons, "
           f"{len(p):,} buys, {int(p.star.sum())} elite finishes.",
           "- **Price** — FFA league AAV as a **ratio to that season's positional mean** inside the",
           "  pool, so 2016 dollars and 2025 dollars are comparable. Cheap = ratio < 1.",
           "  (FFA AAV vs this league's real clearing bids: Spearman 0.90, `LATE_BREAKOUTS.md`;",
           "  §6 re-runs everything on the real bids.)",
           "- **Star** — season total in league scoring (full PPR, 6-pt pass TD, −1 INT), ranked",
           "  QB≤3, RB≤6, WR≤9, TE≤3. Season totals, not per-game: missed games count against you.",
           "- **Value** — VORP against QB13/RB27/WR35/TE13. `E[val if kept]` is VORP floored at 0,",
           "  because a busted $8 buy gets cut in week 4 for a waiver body: you lose the dollars",
           "  and the slot, not 60 points of VORP.", "")

        md(o, "## 1. Yes — and at QB and TE it is most of the market", "",
           "Below-average buys star far less often than expensive ones, but they are **not** a",
           "rounding error: they supplied a quarter of all elite seasons, and at QB nearly half.", "")
        tbl(o, pos_tab)
        md(o, f"Pooled: **{int(cheap.star.sum())} of {int(p.star.sum())} elite seasons "
              f"({100*cheap.star.sum()/p.star.sum():.0f}%) were bought below the positional average**, "
              f"at {100*base:.1f}% per buy.", "",
           "The asymmetry is structural. QB and TE are one-starter positions where the whole",
           "market is compressed into single digits, so \"below average\" means $4-6 and the gap",
           "between the average QB buy and an elite one is a few dollars. RB is the opposite: the",
           "elite tier is bought, and 81% of top-6 RB seasons came from at-or-above-average prices.", "")

        md(o, "## 2. The bargain bin is not where cheap stars come from", "",
           "Sorting the whole pool by price relative to the positional mean:", "")
        tbl(o, band_tab)
        md(o, "Two things fall out of this table:", "",
           "1. **P(star) is monotone in price.** The market is not fooled: there is no price band",
           "   where stardom is more *likely* than the band above it.",
           "2. **`$ per star` is not monotone, and it is flattest where you would not expect.**",
           "   Across the whole board a top-10% season costs $80-140 in expectation no matter",
           "   where you shop — the cheapest cell is 50-75% of the positional average ($81),",
           "   the most expensive is 25-50% ($142). Dollars buy stardom at roughly one rate;",
           "   what changes down the price curve is the **slot cost and the floor**: 76% of",
           "   sub-50% buys finish below replacement, against 49% at 50-75% and 19% at the top.",
           "",
           "So \"cheap star\" does not mean the $1-3 flier. Below 50% of the positional average",
           f"the star rate is {100*p[p.ratio<.5].star.mean():.1f}% and three-quarters of the buys are dead roster spots by",
           "October. The cheap stars live in the **discount rack right below the average price** —",
           "the 50-100% band, roughly $6-16 depending on position.", "")

        md(o, "## 3. Every draft-day feature, run against the price", "",
           "Every draft-day feature we have, tested inside the cheap pool against controls for",
           "log(price ratio) and position — so this is *residual* signal, over and above what the",
           "price already says:", "")
        tbl(o, scan)
        md(o, "Two rows clear the bar (they are the same finding, and §4 takes them apart);",
           "with 16 tests and 53 events nothing else survives a multiplicity correction.",
           "Note what died: **age + draft capital, second/third-year breakout window,",
           "vacated targets/carries, prior snap and target share, ceiling premium**. Those are all",
           "real breakout signals — `LATE_BREAKOUTS.md` finds them at the $1 end — but at this",
           "price level the market has *already* paid for them. Changing teams is, if anything, a",
           "negative, which matches the dart study.", "")

        md(o, "## 4. The one thing that survives: the projection already knows", "",
           "The class is defined by two draft-day facts, no model required:", "",
           "> **Priced at 50-100% of the positional average, and projected inside the positional",
           "> starter tier (QB1-12 / RB1-24 / WR1-30 / TE1-12 by consensus points).**", "")
        tbl(o, grid[["band", "projection", "n", "stars", "star %", "avg $", "$ per star"]])
        md(o, f"**{k}/{n} = {100*k/n:.1f}%** (95% CI {100*ci.low:.0f}-{100*ci.high:.0f}%) versus",
           f"**{100*peer.star.mean():.1f}%** for buys at the *same price* whose projection is weaker",
           f"(Fisher p = {pfish:.4f}). At **${cls.aav.mean():.1f} a head and ~{n/10:.0f} of them a season**,",
           f"this is the cheapest stardom on the board: **${cls.aav.mean()/cls.star.mean():.0f} per elite",
           f"season** against ${p[p.ratio>=2].aav.mean()/p[p.ratio>=2].star.mean():.0f} at the top of the market.", "",
           "The mechanism is boring and that is the point. These are not sleepers — they are",
           "*consensus starters the room does not want to pay for*: the quarterback who finished",
           "QB15 on a bad team, the tight end coming off an injury year, the WR3 on a good",
           "offense. The projections have them in the starter tier; the auction prices them as",
           "depth. **No breakout is required for the class to pay** — one healthy season at the",
           "projected role is often enough, because the elite tier at QB/TE/WR is thin.", "",
           "One caution, stated plainly: an earlier cut of this analysis used *price rank minus",
           "projection rank ≥ 5* (\"the market disagrees with the projection\") and got 37% on",
           "n=27. Put both variables in one model and the disagreement term goes to zero",
           "(p = 0.77) while the projection rank stays (p = 0.005). The disagreement version was",
           "the same signal with a smaller sample; the projection-tier version above is the one",
           "to trade.", "")

        md(o, "### Per season and per position", "")
        tbl(o, per_season)
        tbl(o, per_pos)
        md(o, "Stars in **all 10 seasons**, 1-5 a year. WR is the deepest source (12 of 26), which",
           "is simply where the pool is biggest; TE and QB have the best rate per dollar.", "")

        md(o, "## 5. Does it hold out?", "")
        tbl(o, split)
        md(o, "The class rate is flat across eras (14.1% → 14.3%) while the *cheaper* cell that",
           "looked good early — a starter-tier projection under 50% of the positional average —",
           "went 6/61 in 2016-21 and 0/12 in 2022-25. Same conclusion as the dart study: the",
           "market closed the deep end, not the discount rack.", "")

        md(o, "### Sensitivity to the cutoffs", "",
           "The two cutoffs — a 50% price floor and the starter-tier projection — were chosen",
           "after looking at the data, so here is the same class under every reasonable",
           "alternative definition:", "")
        tbl(o, sens)
        md(o, "It survives every pool size and every star definition. It does **not** survive",
           "swapping the positional mean for the median, which is informative rather than fatal:",
           "the median sits at $4-8, so \"50-100% of median\" lands squarely in the bargain bin,",
           "where §2 already showed there is nothing to find. *Below average* is the frame that",
           "works, which is also the frame the question was asked in.", "")

        if bids_tab is not None:
            md(o, "## 6. Replication on this league's own auction (2023-25)", "",
               "Everything above rides on FFA AAV. Re-run it on the actual winning bids in this",
               f"room ({int(bids_tab.n.sum())} non-keeper buys, `outputs/espn_drafts.csv`), with the positional",
               "average computed from what the room actually paid:", "")
            tbl(o, bids_tab)
            md(o, f"And the class itself: **{int(bids_cls.star.sum())}/{len(bids_cls)} = "
                  f"{100*bids_cls.star.mean():.0f}%** at an average winning bid of "
                  f"**${bids_cls.bid.mean():.1f}** — the study said 14.2% at $9.9. Independent data,",
               "same number. The hits: "
               + ", ".join(f"{r.player} (${r.bid:.0f}, {r.pos}{int(r.pos_rank)})"
                           for r in bids_cls[bids_cls.star == 1].itertuples()) + ".", "")

        md(o, "## 7. What it is worth at the table", "")
        tbl(o, econ)
        md(o, "Note what the `value per $` column does *not* say: the class and the bargain bin",
           f"return the same **{econ.loc[0,'value per $']}× per dollar**. Dollars are not the scarce thing at this end of the",
           "auction — **roster spots are**, and that is the whole difference. A class buy clears",
           f"replacement **{econ.loc[0,'P(above repl.)']:.0f}%** of the time, so the slot is live even when the elite season does",
           f"not come; a bargain-bin buy is a dead slot **{100-econ.loc[1,'P(above repl.)']:.0f}%** of the time. Stacking them:", "")
        for kk in (1, 2, 3, 4):
            md(o, f"- **{kk} class buy{'s' if kk > 1 else ''} (~${kk*cls.aav.mean():.0f})** → "
                  f"{100*(1-(1-cls.star.mean())**kk):.0f}% chance of at least one elite finisher")
        st = p[p.ratio >= 2]
        md(o, f"- **one stud (~${st.aav.mean():.0f})** → {100*st.star.mean():.0f}% chance, and he is a top-6 asset when he misses", "",
           "Four class buys cost the same as one stud and land an elite season more often — but",
           "they burn four roster spots and their misses are droppable, not tradeable. This is",
           "not an argument for punting the top of the draft; it is an argument for where the",
           f"**middle** of your budget goes. The room's own history says the same thing: ~{n/10:.0f} of these",
           "exist per auction, so budgeting **$30-40 for three or four of them** is the whole play.", "")

        md(o, "## 8. The 2026 slate", "",
           "Same rule applied to `projections_2026_wk0.csv` (league-scored FFA). Positional",
           "averages inside the 2026 pool: "
           + ", ".join(f"{r.position} ${r.pos_avg:.0f}"
                       for r in pool26.groupby("position", as_index=False).pos_avg.mean().itertuples())
           + ".", "")
        show = cls26[["player", "position", "team", "aav", "pos_avg", "points", "proj_rank", "adp"]].copy()
        show.columns = ["player", "pos", "team", "FFA $", "pos. avg $", "proj pts", "proj pos. rank", "ADP"]
        for c in ("FFA $", "pos. avg $", "proj pts", "ADP"):
            show[c] = show[c].round(1)
        tbl(o, show)
        md(o, f"{len(cls26)} names, ~${cls26.aav.sum():.0f} to buy the lot — obviously you take three or four,",
           "not twenty. Two health warnings on this list:", "",
           "- **FFA AAV is the national market, not this room.** Keeper inflation runs this",
           "  league's prices above national AAV. Apply the rule to the draft tool's live Exp $:",
           "  the test is *cheaper than the positional average of what this room is paying*, and",
           "  it has to be re-checked live, not from this table.",
           "- **The TE list is long this year** because 2026 TE pricing is unusually flat ($10",
           "  average with eight bodies inside the starter tier). That is the market saying the",
           "  position is a coin flip, and the class rule cannot break the tie.", "",
           "## 9. Limits", "",
           "- 26 events over 10 seasons. The 95% CI on the class rate is 9-20%; this is a real",
           "  effect but not a precise one.",
           "- The 50% price floor and the starter-tier cutoff were chosen after looking at the",
           "  data. The sensitivity table above says they are not knife-edge, and the era split",
           "  holds — but the honest forward test is the next two drafts, not this report.",
           "- Everything here is priced pre-season. It says nothing about in-season acquisition,",
           "  and it deliberately ignores keeper value, which is the *other* reason these buys are",
           "  good (`LATE_BREAKOUTS.md` §5).", "")

    out26 = cls26[["player", "position", "team", "aav", "pos_avg", "ratio", "points",
                   "proj_rank", "adp", "floor", "ceiling", "uncertainty"]].copy()
    out26["ratio"] = out26.ratio.round(3)
    out26["pos_avg"] = out26.pos_avg.round(1)
    out26.to_csv(CSV26, index=False)
    print(f"wrote {OUT}")
    print(f"wrote {CSV26}  ({len(cls26)} players)")
    print(f"\nclass: {k}/{n} = {100*k/n:.1f}%  vs same-price weaker projection "
          f"{100*peer.star.mean():.1f}%  (fisher p={pfish:.4f})")


if __name__ == "__main__":
    main()
