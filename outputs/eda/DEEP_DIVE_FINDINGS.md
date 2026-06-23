# NFL Fantasy Deep Dive — Findings Report

**Two avenues explored**
1. **Full-season** — what preseason/historical data explains a player's full-season fantasy output (PPG + positional finish).
2. **Weekly DFS** — what historical data explains weekly DFS performance, beyond the existing walk-forward model.

**Data assembled for this dive** (all free, pulled into `nfl_odds.db` as `nflv_*` tables):
season aggregates 2011–2025 (28.6K), weekly stats (269K), rosters/age/draft/experience, draft picks, combine, snap counts (324K), depth charts, **expected fantasy points / xFP** (`ff_opportunity`, 80K weekly rows), and a current preseason ADP/ECR snapshot. The season-projection dataset (`season_dataset`) has **5,909 player-seasons across 14 year-over-year transitions (2012→2025)** — vs. the 2 transitions the odds-only DB allowed.

> Scope note: per your choices, inputs are **prior-season production + player context (age/draft/experience/size/combine) + role (snaps)**. Actual preseason *game* stats were intentionally excluded (weak signal). Historical preseason ADP isn't cleanly archived in nflverse, so prior-year production serves as the market-baseline proxy; ADP is a noted gap.

---

## AVENUE 1 — Full-season fantasy

### 1. How repeatable is full-season production?
"Repeat last year's PPG" is a surprisingly strong baseline. Year-over-year rank correlation of PPG (players with ≥4 prior games):

| Pos | n | Spearman (prior→next PPG) | "Repeat" R² | "Repeat" MAE (PPG) |
|---|---|---|---|---|
| WR | 2,089 | **0.754** | 0.544 | 2.90 |
| TE | 1,153 | 0.719 | 0.515 | 2.13 |
| RB | 1,380 | 0.697 | 0.419 | 3.35 |
| QB | 625 | 0.659 | 0.333 | 4.15 |

**Takeaway:** WR/TE are the most predictable year to year; **RB and especially QB carry the most surprise** (where projection edges and busts live).

### 2. Strongest correlates of next-season PPG (overall)
| Signal | Spearman |
|---|---|
| Prior PPG | **0.744** |
| PPG two years ago | 0.663 |
| Prior total TDs | 0.663 |
| **Prior snap %** | **0.631** |
| Prior weekly std (volume of scoring) | 0.622 |
| Prior coeff. of variation | **−0.610** |
| Draft pick / round | −0.466 / −0.455 |
| Prior games played (durability) | 0.422 |

**Takeaways:** (a) two years of history beats one — a single down/up year is partly noise; (b) **snap share is nearly as predictive as points** — role is destiny; (c) **low coefficient of variation (a steady, high-floor player) strongly predicts a good next year** — consistency is a feature, not just a description; (d) draft capital still matters years into a career.

### 3. The actionable edge — what predicts BEATING "repeat last year" (residual)
Correlations with `next_ppg − prior_ppg`:

| Signal | Spearman | Interpretation |
|---|---|---|
| Prior PPG | −0.320 | **Mean reversion** — high scorers regress, low scorers bounce |
| Prior weekly std | −0.283 | Boom seasons regress |
| **Prior total TDs** | **−0.274** | **TD regression** (see below) |
| Prior coeff. of variation | **+0.246** | **Volatile/under-utilized players have the most upside** |
| Prior snap % | −0.221 | Already-maxed roles can't grow |
| Years exp / Age | −0.176 / −0.162 | **Aging decline** is real after controlling for prior level |

### 4. Mechanisms (with evidence)

**TD regression is large and monotonic.** Grouping skill players by prior total TDs, the next-year change in PPG:

| Prior TDs | Mean next-year PPG change |
|---|---|
| Low | **+0.67** |
| Med | −0.28 |
| High | −1.19 |
| Elite | **−1.94** |

→ Touchdown-driven seasons are the #1 fade signal for next year. Yardage/target volume is far stickier than TDs.

**Draft capital persists** (next-season PPG by draft slot): e.g. WR R1-top15 **11.8** vs UDFA **4.9**; RB R1-top15 **13.2** vs UDFA **4.9**; TE first-rounders nearly double Day-3 picks. Capital is a real prior even mid-career.

**Aging:** decline shows clearly in the residual (older players underperform their prior level). Raw age-curve charts (`a1_age_curves.png`) are confounded by survivorship at the young end, so the residual is the cleaner read: **expect decline, not growth, once experience accrues.**

### 5. Can we beat the naive baseline? (temporal validation, train ≤2022 / test 2023–25)
| Model | Test MAE (next PPG) |
|---|---|
| Repeat last year | 2.876 |
| GB, prior PPG only | 2.852 |
| **GB, full feature set** | **2.660** (R² = 0.644) |

→ A full model cuts season-projection error **~7.5%** over "repeat last year." `prior_ppg` carries ~72% of importance, but **prior2_ppg, draft pick, age, team change, prior games (durability), CV, and rushing involvement** all add real signal. This is a buildable season model.

---

## AVENUE 2 — Weekly DFS

### A. Volume is the signal; efficiency is noise (week-to-week lag-1 correlation)
| Metric | Stickiness |
|---|---|
| Opportunities (targets+carries) | **0.712** |
| Targets | 0.634 |
| Fantasy points (PPR) | 0.520 |
| **Yards per opportunity (efficiency)** | **0.145** |

→ Opportunity is ~5× more repeatable than efficiency. **Project volume; treat efficiency as regression-prone.** This is the cleanest lever for weekly projections.

### B. Expected points (xFP) are as predictive as actual — with less luck
Split-half (first 9 weeks → rest of season, per-game):

| Predictor of H2 actual | Spearman |
|---|---|
| H1 **actual** points | 0.766 |
| H1 **expected** points (xFP) | **0.768** |
| H1 **luck** (actual − expected) | **0.126** |

→ xFP matches actual points as a forward signal while stripping out variance, and **over/under-performance vs. expectation barely persists (0.126).** Practical rule: **players who beat their xFP are sell/fade candidates; those who underperformed xFP are buy-low.** `total_fantasy_points_exp` is now in the DB (`nflv_ff_opp`) and should be a core weekly feature.

### C. The prop market is sharp — but unevenly (2023–25 lines vs actual)
| Market | n | Over rate | Line MAE | Corr(line, actual) |
|---|---|---|---|---|
| Rush yds | 4,609 | 0.455 | 17.9 | **0.732** |
| Reception yds | 9,419 | 0.477 | 19.6 | 0.585 |
| Receptions | 8,903 | 0.481 | 1.5 | 0.566 |
| **Pass yds** | 1,641 | 0.500 | 57.1 | **0.341** |

→ Lines are well-calibrated (over rates 45–50%) and hard to beat on yards/receptions — confirming the existing model's reliance on props is rational. **The QB passing-yards market is the softest (corr 0.34)** — the most likely place to find independent edge.

### D. Where the existing model systematically misses (residual diagnostics on `dfs_predictions`)
- **QB is the weakest position** (MAE 6.19, corr 0.47) — the biggest improvement target, and it lines up with the soft pass-yds market in (C).
- **Persistent under-prediction in high-scoring environments:** bias grows from −0.20 (O/U ≤42) to **−0.32 (O/U 50+)**, and MAE rises to 5.0 in shootouts. The model under-rates ceiling games — exactly the GPP/tournament spots that matter.
- Moderate favorites (spread 7–10) are under-predicted (bias −0.45).

→ Add stronger **game-environment interactions** (O/U × role, implied team total) and treat QB separately.

---

## What this sets up (proposed models, pending your go-ahead)

**Avenue 1 — season model (new):** GB/LightGBM regression on `season_dataset` predicting next-season PPG with proper year-blocked validation, then derive positional finish/VBD. Core features confirmed above. Add historical-ADP only if we source it. Lances the two biggest fantasy questions: who regresses (TDs, mean reversion) and who's durable.

**Avenue 2 — weekly extensions (build on existing):** (1) add **xFP** and **opportunity-share** features; (2) a **luck-correction** feature (rolling actual − expected); (3) **game-environment interactions** for the shootout under-prediction; (4) a **QB-specific** treatment given the soft pass market.

**Known gaps / caveats:** no historical preseason ADP (used prior production as proxy); season dataset conditions on playing in year Y (a separate attrition/durability analysis would capture players who vanish); injuries/weather not yet integrated; 2023–25 is the only window with odds for prop-market work.

*All numbers reproducible via `scripts/eda_avenue1_season.py` and `scripts/eda_avenue2_weekly.py`; tables/charts in `outputs/eda/`.*
