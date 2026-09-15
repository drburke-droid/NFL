# Week-2-only tweak study (2026-09-15)

**Verdict: no week-2-specific tweak beats the incumbent.** On 2,667 played week-2 rows across nine walk-forward
test seasons (2017-2025) the incumbent (Model_Burke: FFA + calibrated residual correction + median offset) is
the best week-2 projection tested, MAE 4.639 vs FFA 4.685 (2/9 seasons FFA better, t p = 0.02). Every
week-2-specific correction built from week-1 usage (snap share, route share, target share, carries, targets,
WOPR), the week-1 points surprise, FFA's own week-1-to-week-2 move, the Vegas implied total, per-position
week-2 offsets, or DraftKings-implied points is worse than or indistinguishable from the incumbent:

- Per-position week-2 offsets flip sign season to season (RB median residual ranges -1.88 to +0.43); there is no
  stable week-2 position bias to exploit.
- A shrunk ridge on week-1 surprise features hurts (+0.05 to +0.15 MAE), whether trained on week-2 rows only,
  weeks 2-4, or weeks 2-18, and whether stacked on FFA or on the incumbent (0/8 seasons better at k = 0.5).
- Decile diagnostics of the incumbent's residual show no monotone under-reaction to any week-1 usage signal;
  the only slopes with p < 0.05 are negative (RB snap share / carries: the incumbent, via FFA, if anything
  OVER-reacts to a big week-1 workload) and the preseason gap.
- DraftKings-implied points (props to PPR) blended on top of the incumbent at week 2: -0.02 MAE in 2024,
  +0.01 to +0.05 in 2025, n = 442. Production already blends DK per stat; nothing extra at week 2.
- The one sliver: nudging the projection 10-20% of the way from the FFA week-2 line back toward the preseason
  per-game line improves MAE by 0.2% (-0.010 to -0.017 on 5.31, 8/9 seasons better for lam = 0.1, t p = 0.02,
  row-level sign test p = 0.28, Spearman unchanged, and the walk-forward-fitted version of the same idea is
  WORSE). Same size as the Round 1 shading effects; not a ranking improvement and not a home run.

Ranking (Spearman) is untouched by every variant (0.646-0.649), which is the metric a week-2 "home run" would
need to move. Week 2 is not special: it is the same ~1% competitive range as every other week.

Scripts: scripts/week2_study.py (parts 1), scripts/week2_study_part2.py (on top of the incumbent), inline (d).
Frame: league visible frame + 2025 with actuals rebuilt from weekly_skill; incumbent refit on played rows from
2016 (inc) and from 2023 (inc23, production window: identical at week 2).

---

# Detail

Played week-2 rows, walk-forward by season (test seasons 2017-2025, n = 2,667). Train = earlier seasons' week-2 rows unless the label says `_all` (weeks 2-18). Incumbent = the Model_Burke package refit on this frame (played rows, FFA baseline, no market), `inc` = trained from 2016, `inc23` = trained from 2023 (production window; 2023 rows fall back to `inc`).

## All seasons pooled

| model | MAE | RMSE | Spearman | bias |
|---|---|---|---|---|
| ffa | 4.685 | 6.320 | 0.648 | -0.33 |
| ffa_off | 4.682 | 6.350 | 0.648 | -0.59 |
| inc | 4.639 | 6.297 | 0.649 | -0.62 |
| inc23 | 4.637 | 6.303 | 0.649 | -0.63 |
| pos_off | 4.688 | 6.351 | 0.647 | -0.55 |
| pos_off_all | 4.673 | 6.365 | 0.647 | -0.76 |
| surp_k0.5 | 4.691 | 6.297 | 0.646 | -0.18 |
| surp_k1.0 | 4.792 | 6.342 | 0.638 | +0.19 |
| surp_all_k0.5 | 4.658 | 6.283 | 0.649 | -0.37 |
| surp_all_k1.0 | 4.727 | 6.265 | 0.649 | +0.02 |
| surp_usage | 4.755 | 6.311 | 0.644 | +0.09 |
| surp_pts | 4.778 | 6.315 | 0.643 | +0.16 |
| surp_vegas | 4.762 | 6.296 | 0.645 | +0.07 |

## Per-season MAE (week 2)

| season | n | ffa | ffa_off | inc | inc23 | pos_off | pos_off_all | surp_k0.5 | surp_k1.0 | surp_all_k0.5 | surp_all_k1.0 | surp_usage | surp_pts | surp_vegas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2017 | 304 | 4.352 | 4.363 | 4.238 | 4.238 | 4.396 | 4.313 | 4.369 | 4.455 | 4.296 | 4.379 | 4.463 | 4.408 | 4.438 |
| 2018 | 296 | 4.717 | 4.753 | 4.696 | 4.696 | 4.761 | 4.770 | 4.709 | 4.776 | 4.730 | 4.764 | 4.738 | 4.774 | 4.731 |
| 2019 | 299 | 4.627 | 4.608 | 4.611 | 4.611 | 4.606 | 4.584 | 4.645 | 4.793 | 4.561 | 4.654 | 4.744 | 4.727 | 4.735 |
| 2020 | 292 | 5.459 | 5.469 | 5.404 | 5.404 | 5.482 | 5.482 | 5.451 | 5.494 | 5.415 | 5.417 | 5.423 | 5.455 | 5.456 |
| 2021 | 294 | 4.750 | 4.717 | 4.667 | 4.667 | 4.709 | 4.696 | 4.756 | 4.895 | 4.703 | 4.797 | 4.823 | 4.889 | 4.865 |
| 2022 | 296 | 4.875 | 4.842 | 4.760 | 4.760 | 4.822 | 4.832 | 4.781 | 4.858 | 4.842 | 4.946 | 4.887 | 4.886 | 4.909 |
| 2023 | 298 | 4.563 | 4.585 | 4.565 | 4.565 | 4.565 | 4.586 | 4.552 | 4.617 | 4.543 | 4.575 | 4.610 | 4.692 | 4.645 |
| 2024 | 287 | 4.164 | 4.119 | 4.144 | 4.119 | 4.176 | 4.103 | 4.275 | 4.469 | 4.169 | 4.322 | 4.412 | 4.463 | 4.405 |
| 2025 | 301 | 4.660 | 4.684 | 4.673 | 4.675 | 4.680 | 4.697 | 4.691 | 4.784 | 4.667 | 4.700 | 4.708 | 4.718 | 4.682 |

## Paired tests vs the incumbent (per-season MAE differences, negative = better)

| model | mean dMAE | seasons better | t-test p (9 seasons) | row-level sign p |
|---|---|---|---|---|
| ffa | +0.045 | 2/9 | 0.023 | 0.000 (1229/2667) |
| ffa_off | +0.043 | 2/9 | 0.025 | 0.009 (1266/2667) |
| inc23 | -0.003 | 1/9 | 0.386 | 0.967 (293/588) |
| pos_off | +0.049 | 2/9 | 0.020 | 0.016 (1271/2667) |
| pos_off_all | +0.034 | 2/9 | 0.053 | 0.230 (1302/2667) |
| surp_k0.5 | +0.052 | 1/9 | 0.018 | 0.005 (1260/2667) |
| surp_k1.0 | +0.154 | 0/9 | 0.001 | 0.000 (1195/2667) |
| surp_all_k0.5 | +0.019 | 3/9 | 0.201 | 0.063 (1285/2667) |
| surp_all_k1.0 | +0.088 | 0/9 | 0.006 | 0.000 (1232/2667) |
| surp_usage | +0.117 | 0/9 | 0.004 | 0.000 (1227/2667) |
| surp_pts | +0.139 | 0/9 | 0.001 | 0.000 (1198/2667) |
| surp_vegas | +0.123 | 0/9 | 0.003 | 0.000 (1218/2667) |

## What the week-2 surprise ridge learned (fit on every week-2 row 2016-2025, standardised coefficients, PPR per 1 sd)

| feature | QB | RB | WR | TE |
|---|---|---|---|---|
| pts_surp | +0.01 | +0.23 | -0.33 | -0.15 |
| tshare1 | +0.22 | -0.12 | +0.19 | +0.11 |
| rshare1 | +0.00 | +0.81 | +0.33 | +0.13 |
| snap1 | -0.46 | -0.25 | +0.65 | -0.02 |
| car1 | +0.53 | -0.75 | -0.04 | -0.00 |
| tgt1 | +0.89 | -0.76 | -0.12 | +0.38 |
| wopr1 | -0.68 | +0.22 | +0.13 | +0.07 |
| ffa_move | +0.13 | -0.02 | +0.61 | +0.10 |
| pre_gap | +0.56 | -0.19 | -0.75 | -0.19 |
| impl_c | +0.09 | +0.17 | +0.15 | +0.15 |
| played1 | +0.01 | +0.05 | -0.51 | -0.38 |
| ffa_lvl | -0.39 | -0.22 | -1.32 | -0.94 |

## Week-2 position offsets by season (median residual actual - FFA, played rows)

| season | QB | RB | TE | WR |
|---|---|---|---|---|
| 2016 | +0.70 | -0.30 | -0.18 | +0.17 |
| 2017 | -1.58 | -0.46 | +0.37 | -1.43 |
| 2018 | +0.29 | -0.52 | +0.57 | +1.73 |
| 2019 | +0.15 | -1.33 | -0.48 | -0.63 |
| 2020 | -0.25 | +0.43 | -0.09 | +0.69 |
| 2021 | +0.33 | -1.75 | -0.92 | -0.65 |
| 2022 | -0.54 | -1.88 | -0.32 | -0.11 |
| 2023 | +1.19 | -0.72 | +0.70 | -0.02 |
| 2024 | -1.83 | +0.00 | -0.77 | -1.03 |
| 2025 | +1.18 | +0.07 | -0.18 | -0.05 |

Weeks 5+ position medians for reference: QB -0.33, RB -0.58, TE -0.08, WR -0.71

## DraftKings-implied points at week 2 (2024-25 test, rows with a DK line)

n = 442; fitted DK weights: all weeks 1-9 -> {2024: 0.30000000000000004, 2025: 0.4}, week 2 only -> {2024: 0.35000000000000003, 2025: 0.8500000000000001}

| model | MAE | RMSE | Spearman | bias |
|---|---|---|---|---|
| ffa | 4.898 | 6.441 | 0.622 | -0.04 |
| inc | 4.897 | 6.442 | 0.617 | -0.38 |
| pos_off | 4.900 | 6.478 | 0.617 | -0.36 |
| surp_k0.5 | 4.969 | 6.470 | 0.618 | +0.07 |
| mkt_w_all19 | 4.891 | 6.487 | 0.618 | -0.55 |
| mkt_w_wk2 | 4.927 | 6.547 | 0.618 | -0.70 |
| mkt_w0.5 | 4.887 | 6.489 | 0.619 | -0.62 |
| mkt_w0.5_surp | 4.940 | 6.470 | 0.621 | -0.20 |

Per season:

| season | n | ffa | inc | pos_off | surp_k0.5 | mkt_w_all19 | mkt_w_wk2 | mkt_w0.5 | mkt_w0.5_surp |
|---|---|---|---|---|---|---|---|---|---|
| 2024 | 201 | 4.908 | 4.878 | 4.899 | 5.028 | 4.852 | 4.846 | 4.832 | 4.938 |
| 2025 | 241 | 4.890 | 4.913 | 4.901 | 4.919 | 4.923 | 4.995 | 4.934 | 4.943 |

DK line coverage among played week-2 rows 2023-25 by position: {'QB': 0.86, 'RB': 0.71, 'TE': 0.7, 'WR': 0.73}


# Part 2: on top of the incumbent

## (a) Incumbent residual by within-position decile of each week-1 signal (week 2, 2017-2025, played rows)

Slope = OLS of residual on decile (PPR per decile), with p. A real under-reaction shows as a monotone positive slope.

| signal | QB slope (p) | RB slope (p) | WR slope (p) | TE slope (p) |
|---|---|---|---|---|
| pts_surp | +0.03 (0.83) | -0.11 (0.21) | -0.04 (0.63) | +0.01 (0.93) |
| snap1 | +0.01 (0.96) | -0.21 (0.02) | +0.01 (0.86) | -0.08 (0.36) |
| rshare1 | n/a | -0.14 (0.13) | -0.00 (0.97) | -0.08 (0.39) |
| tshare1 | n/a | -0.17 (0.06) | -0.05 (0.55) | -0.02 (0.85) |
| car1 | +0.18 (0.24) | -0.24 (0.01) | n/a | n/a |
| tgt1 | n/a | -0.16 (0.07) | -0.03 (0.70) | -0.00 (0.97) |
| pre_gap | +0.32 (0.03) | -0.14 (0.07) | -0.19 (0.01) | -0.00 (0.97) |
| ffa_move | +0.07 (0.62) | -0.07 (0.40) | +0.02 (0.77) | -0.04 (0.62) |
| impl_c | -0.08 (0.56) | +0.05 (0.51) | -0.05 (0.46) | +0.02 (0.79) |
| ffa_lvl | -0.16 (0.27) | -0.20 (0.01) | -0.18 (0.01) | -0.14 (0.06) |

Top vs bottom decile of the incumbent residual (mean actual - inc), played-week-1 rows:

| signal | pos | bottom decile | top decile | n/decile |
|---|---|---|---|---|
| pts_surp | QB | +0.06 | +2.46 | 29 |
| pts_surp | RB | +1.68 | -0.58 | 62 |
| pts_surp | WR | +1.62 | +0.29 | 92 |
| pts_surp | TE | -0.92 | +0.03 | 43 |
| snap1 | QB | +2.39 | +0.68 | 29 |
| snap1 | RB | +0.92 | -1.57 | 62 |
| snap1 | WR | +0.49 | +1.42 | 92 |
| snap1 | TE | +0.76 | -0.40 | 43 |
| pre_gap | QB | -1.42 | +0.97 | 29 |
| pre_gap | RB | +1.51 | -1.06 | 62 |
| pre_gap | WR | +2.22 | -0.93 | 92 |
| pre_gap | TE | -0.88 | -1.23 | 43 |

## (b) Ridge on the incumbent residual, k = 0.5 (week 2, test seasons 2018-2025, n = 2363)

| model | MAE | dMAE vs inc | seasons better | t p | sign p |
|---|---|---|---|---|---|
| inc+ridge_wk2_a30 | 4.750 | +0.060 | 0/8 | 0.00 | 0.00 |
| inc+ridge_wk2_a100 | 4.741 | +0.051 | 0/8 | 0.01 | 0.00 |
| inc+ridge_wk2-4_a30 | 4.740 | +0.050 | 2/8 | 0.04 | 0.00 |
| inc+ridge_wk2-4_a100 | 4.738 | +0.047 | 2/8 | 0.04 | 0.00 |

incumbent MAE on the same rows: 4.691

## (c) DK-implied points on top of the incumbent, inc + w * (mkt - ffa) (2024-25 week 2, rows with a DK line, n = 442)

| w | MAE | 2024 | 2025 | Spearman |
|---|---|---|---|---|
| 0 | 4.897 | 4.878 | 4.913 | 0.617 |
| 0.2 | 4.881 | 4.843 | 4.913 | 0.619 |
| 0.3 | 4.877 | 4.828 | 4.917 | 0.621 |
| 0.4 | 4.874 | 4.814 | 4.924 | 0.622 |
| 0.5 | 4.874 | 4.804 | 4.933 | 0.622 |
| 0.7 | 4.883 | 4.789 | 4.961 | 0.622 |

## (d) Regress toward the preseason line: pred = inc + lam * (preseason per-game - FFA wk2), week 2, rows with a preseason line (n = 1815)

Fixed lambda (no fitting) and a walk-forward OLS slope per position shrunk by 0.5. Coverage of a preseason line among played week-2 rows: 68%.

| variant | MAE | dMAE vs inc | seasons better | t p | sign p | Spearman |
|---|---|---|---|---|---|---|
| inc | 5.308 | +0.000 | 0/9 | nan | nan | 0.574 |
| lam=0.1 RB+WR | 5.298 | -0.010 | 6/9 | 0.08 | 0.16 | 0.575 |
| lam=0.1 all | 5.297 | -0.011 | 8/9 | 0.02 | 0.28 | 0.575 |
| lam=0.2 RB+WR | 5.291 | -0.017 | 5/9 | 0.12 | 0.24 | 0.574 |
| lam=0.2 all | 5.293 | -0.015 | 8/9 | 0.10 | 0.54 | 0.573 |
| lam=0.3 RB+WR | 5.287 | -0.020 | 5/9 | 0.20 | 0.40 | 0.573 |
| lam=0.3 all | 5.297 | -0.010 | 8/9 | 0.45 | 0.93 | 0.571 |
| walk-forward OLS x0.5 (2018+ only) | 5.316 | +0.008 | 1/9 | 0.20 | 0.24 | 0.574 |

Per-position mean incumbent residual by preseason-gap tercile (in-sample, all week-2 rows):

| pos | FFA far above preseason | middle | FFA far below preseason |
|---|---|---|---|
| QB | +1.32 | +0.18 | -0.40 |
| RB | +0.27 | -0.75 | +0.70 |
| WR | -0.33 | +0.73 | +0.97 |
| TE | -0.14 | +0.72 | -0.28 |
