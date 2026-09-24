# Game scripts as a fantasy signal -- 2607 games, 5214 team-games, seasons 2016-2025

Player rows: every QB/RB/WR/TE with an FFA weekly line (league frame 2016-24 + 2025 holdout), PPR. Baseline = the FFA consensus line scored PPR (the league's reference; Model_Burke sits ~1% under it). MAE is on played rows (any touch/attempt), the judge's convention. Scores and the 5 quarter-trajectory script labels come from db/nfl_odds.db game_scripts; spread/total are the closing lines in the frame.

## The three taxonomies and how often each script happens

**traj** (side = wl)

| script | n | total | abs_margin | fav_margin | total_line | abs_spread | share |
|---|---|---|---|---|---|---|---|
| Steady Build | 688.0 | 37.8 | 9.3 | 3.6 | 44.7 | 5.1 | 0.264 |
| 2nd Half Blowout | 590.0 | 35.2 | 9.7 | 4.2 | 43.9 | 5.0 | 0.226 |
| Shootout | 538.0 | 59.3 | 6.7 | 2.5 | 46.9 | 5.1 | 0.206 |
| Tight Throughout | 396.0 | 49.1 | 5.6 | 2.0 | 45.2 | 5.0 | 0.152 |
| Wire-to-Wire Blowout | 395.0 | 50.5 | 26.7 | 15.9 | 45.5 | 6.5 | 0.152 |

**abs** (side = fd)

| script | n | total | abs_margin | fav_margin | total_line | abs_spread | share |
|---|---|---|---|---|---|---|---|
| Fav blowout | 685.0 | 45.7 | 22.0 | 22.0 | 45.2 | 6.8 | 0.263 |
| Upset | 661.0 | 45.1 | 9.0 | -9.0 | 45.1 | 4.8 | 0.254 |
| Standard | 508.0 | 44.7 | 6.1 | 3.3 | 45.2 | 5.0 | 0.195 |
| Slog | 380.0 | 29.5 | 6.1 | 3.3 | 43.4 | 4.4 | 0.146 |
| Shootout | 373.0 | 61.6 | 5.4 | 3.6 | 47.0 | 4.7 | 0.143 |

**rel** (side = fd)

| script | n | total | abs_margin | fav_margin | total_line | abs_spread | share |
|---|---|---|---|---|---|---|---|
| Fav blowout | 685.0 | 45.7 | 22.0 | 22.0 | 45.2 | 6.8 | 0.263 |
| Upset | 661.0 | 45.1 | 9.0 | -9.0 | 45.1 | 4.8 | 0.254 |
| Standard | 512.0 | 45.2 | 6.1 | 3.2 | 45.2 | 4.9 | 0.196 |
| Slog | 400.0 | 30.7 | 6.1 | 3.4 | 44.9 | 4.4 | 0.153 |
| Shootout | 349.0 | 61.5 | 5.3 | 3.6 | 45.6 | 4.7 | 0.134 |

## a) Are there discrete scripts in the fantasy data?

Per game, a 10-vector: each side's actual-minus-FFA residual by position and in total (favourite, then dog). If scripts were real categories these would clump.

Residual shape (what an oracle would need to exploit):

| k | silhouette | gmm_bic | sizes |
|---|---|---|---|
| 2 | 0.189 | -3650 | 1420, 1183 |
| 3 | 0.135 | -3417 | 991, 812, 800 |
| 4 | 0.125 | -3032 | 725, 720, 674, 484 |
| 5 | 0.123 | -2705 | 626, 597, 496, 447, 437 |
| 6 | 0.114 | -2350 | 529, 528, 469, 406, 360, 311 |
| 7 | 0.114 | -1887 | 493, 467, 418, 360, 355, 256, 254 |
| 8 | 0.118 | -1493 | 458, 450, 359, 316, 295, 264, 232, 229 |

Raw shape (actual PPR by position and side):

| k | silhouette | gmm_bic | sizes |
|---|---|---|---|
| 2 | 0.198 | -3615 | 1536, 1067 |
| 3 | 0.138 | -3441 | 1005, 844, 754 |
| 4 | 0.126 | -3240 | 749, 691, 685, 478 |
| 5 | 0.129 | -2918 | 695, 618, 546, 391, 353 |
| 6 | 0.133 | -2565 | 675, 511, 424, 335, 333, 325 |
| 7 | 0.128 | -2262 | 643, 415, 378, 322, 302, 296, 247 |
| 8 | 0.133 | -1856 | 585, 358, 331, 281, 279, 265, 254, 250 |

Adjusted mutual information between a k=5 clustering of the residual shape and each taxonomy: {'traj': 0.083, 'abs': 0.121, 'rel': 0.121} (0 = independent, 1 = identical).

### Where the miss lives: variance decomposition of the player residual (played rows)

| component | share of residual variance |
|---|---|
| team-game mean residual (ceiling for any game+side info) | 0.1388 |
| game mean residual (both sides pooled) | 0.0847 |
| team-game x position mean residual | 0.4265 |
| traj: (script, side, position) cell mean | 0.0483 |
| abs: (script, side, position) cell mean | 0.0506 |
| rel: (script, side, position) cell mean | 0.0512 |
| ridge on realized margin/total x position x proj (continuous, in-sample) | 0.1295 |

Reading: the first row is the ceiling for any information that is constant across a team's players in a game -- an oracle who knew the team's exact fantasy total residual. Everything below it is what a label or the final score recovers of that ceiling. The (script, side, position) cell means are in-sample (~40 cells) so they are slightly generous.

### Effect sizes: mean residual by script, side, position (all seasons, played rows, FFA line > 0.5)

**traj** -- actual/FFA ratio (left) and mean residual in PPR points (right)

| script, side | QB | RB | WR | TE | QB pts | RB pts | WR pts | TE pts |
|---|---|---|---|---|---|---|---|---|
| ('2nd Half Blowout', 'loser') | 0.78 | 0.88 | 0.89 | 1.02 | -2.97 | -1.03 | -0.88 | 0.13 |
| ('2nd Half Blowout', 'winner') | 0.92 | 1.05 | 0.94 | 0.97 | -1.27 | 0.44 | -0.52 | -0.16 |
| ('Shootout', 'loser') | 1.27 | 1.09 | 1.21 | 1.28 | 4.18 | 0.84 | 1.85 | 1.7 |
| ('Shootout', 'winner') | 1.27 | 1.22 | 1.15 | 1.22 | 4.4 | 2.07 | 1.44 | 1.31 |
| ('Steady Build', 'loser') | 0.8 | 0.88 | 0.93 | 1.02 | -2.9 | -1.06 | -0.58 | 0.14 |
| ('Steady Build', 'winner') | 0.97 | 1.04 | 0.95 | 1.01 | -0.4 | 0.34 | -0.47 | 0.04 |
| ('Tight Throughout', 'loser') | 1.03 | 1.02 | 1.03 | 1.08 | 0.41 | 0.21 | 0.27 | 0.47 |
| ('Tight Throughout', 'winner') | 1.14 | 1.1 | 1.08 | 1.16 | 2.24 | 0.88 | 0.73 | 0.94 |
| ('Wire-to-Wire Blowout', 'loser') | 0.67 | 0.82 | 0.88 | 0.91 | -4.04 | -1.54 | -0.97 | -0.52 |
| ('Wire-to-Wire Blowout', 'winner') | 1.25 | 1.35 | 1.11 | 1.22 | 3.51 | 3.13 | 1.04 | 1.34 |

**abs** -- actual/FFA ratio (left) and mean residual in PPR points (right)

| script, side | QB | RB | WR | TE | QB pts | RB pts | WR pts | TE pts |
|---|---|---|---|---|---|---|---|---|
| ('Fav blowout', 'dog') | 0.72 | 0.83 | 0.88 | 1.01 | -3.51 | -1.41 | -0.95 | 0.04 |
| ('Fav blowout', 'fav') | 1.12 | 1.25 | 1.06 | 1.1 | 1.81 | 2.28 | 0.58 | 0.6 |
| ('Shootout', 'dog') | 1.4 | 1.17 | 1.27 | 1.23 | 5.97 | 1.52 | 2.36 | 1.33 |
| ('Shootout', 'fav') | 1.29 | 1.16 | 1.17 | 1.28 | 5.04 | 1.57 | 1.64 | 1.79 |
| ('Slog', 'dog') | 0.77 | 0.89 | 0.87 | 0.93 | -3.2 | -0.93 | -1.03 | -0.38 |
| ('Slog', 'fav') | 0.75 | 0.9 | 0.86 | 0.95 | -3.86 | -0.88 | -1.27 | -0.31 |
| ('Standard', 'dog') | 1.04 | 1.02 | 1.05 | 1.16 | 0.6 | 0.14 | 0.43 | 0.88 |
| ('Standard', 'fav') | 0.99 | 1.03 | 1.0 | 1.11 | -0.14 | 0.24 | -0.04 | 0.68 |
| ('Upset', 'dog') | 1.18 | 1.17 | 1.07 | 1.09 | 2.5 | 1.49 | 0.6 | 0.5 |
| ('Upset', 'fav') | 0.86 | 0.9 | 0.95 | 1.05 | -2.31 | -0.94 | -0.44 | 0.31 |

**rel** -- actual/FFA ratio (left) and mean residual in PPR points (right)

| script, side | QB | RB | WR | TE | QB pts | RB pts | WR pts | TE pts |
|---|---|---|---|---|---|---|---|---|
| ('Fav blowout', 'dog') | 0.72 | 0.83 | 0.88 | 1.01 | -3.51 | -1.41 | -0.95 | 0.04 |
| ('Fav blowout', 'fav') | 1.12 | 1.25 | 1.06 | 1.1 | 1.81 | 2.28 | 0.58 | 0.6 |
| ('Shootout', 'dog') | 1.41 | 1.18 | 1.28 | 1.25 | 5.94 | 1.53 | 2.32 | 1.44 |
| ('Shootout', 'fav') | 1.32 | 1.16 | 1.19 | 1.31 | 5.38 | 1.55 | 1.73 | 1.96 |
| ('Slog', 'dog') | 0.78 | 0.9 | 0.88 | 0.95 | -3.13 | -0.88 | -1.04 | -0.29 |
| ('Slog', 'fav') | 0.76 | 0.9 | 0.86 | 0.95 | -3.92 | -0.94 | -1.31 | -0.28 |
| ('Standard', 'dog') | 1.07 | 1.02 | 1.07 | 1.15 | 0.97 | 0.19 | 0.62 | 0.82 |
| ('Standard', 'fav') | 1.0 | 1.04 | 1.01 | 1.1 | 0.02 | 0.4 | 0.06 | 0.61 |
| ('Upset', 'dog') | 1.18 | 1.17 | 1.07 | 1.09 | 2.5 | 1.49 | 0.6 | 0.5 |
| ('Upset', 'fav') | 0.86 | 0.9 | 0.95 | 1.05 | -2.31 | -0.94 | -0.44 | 0.31 |

## b) Can the script be called before kickoff?

Walk-forward by season (train on all earlier seasons, test on the season shown; first test season needs 2 years of history). Features: |spread|, total line, home-favourite flag (vegas); plus each side's FFA team projections by position and pass/rush yards (vegas+ffa); plus each team's trailing-8-game mean total, margin, |margin| and fantasy residual (+history; the GBM uses this full set). base = always pick the most common script. The pregame pick used in (c) is the +history logit's most likely class.

**traj**

| season | n | base_acc | base_ll | vegas_acc | vegas_ll | vegas+ffa_acc | vegas+ffa_ll | +history_acc | +history_ll | gbm_acc | gbm_ll |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 256.0 | 0.23 | 1.6002 | 0.293 | 1.5568 | 0.246 | 1.6372 | 0.266 | 1.6596 | 0.273 | 1.8312 |
| 2019 | 256.0 | 0.293 | 1.5795 | 0.273 | 1.5554 | 0.281 | 1.5952 | 0.27 | 1.6292 | 0.203 | 1.9447 |
| 2020 | 224.0 | 0.237 | 1.611 | 0.33 | 1.5459 | 0.29 | 1.6044 | 0.29 | 1.62 | 0.223 | 1.8756 |
| 2021 | 272.0 | 0.213 | 1.6027 | 0.261 | 1.5775 | 0.239 | 1.6134 | 0.239 | 1.6315 | 0.232 | 1.8083 |
| 2022 | 271.0 | 0.284 | 1.587 | 0.299 | 1.5483 | 0.306 | 1.5657 | 0.303 | 1.5858 | 0.214 | 1.704 |
| 2023 | 272.0 | 0.298 | 1.5786 | 0.276 | 1.5408 | 0.309 | 1.5427 | 0.294 | 1.5458 | 0.243 | 1.6504 |
| 2024 | 272.0 | 0.246 | 1.5993 | 0.276 | 1.5662 | 0.287 | 1.5712 | 0.287 | 1.5819 | 0.265 | 1.6807 |
| 2025 | 272.0 | 0.25 | 1.5948 | 0.268 | 1.5709 | 0.279 | 1.5717 | 0.276 | 1.5796 | 0.224 | 1.6489 |
| ALL | 2095.0 | 0.256 | 1.594 | 0.284 | 1.558 | 0.28 | 1.588 | 0.278 | 1.604 | 0.235 | 1.768 |

Top predicted probability (+history logit): median 0.32, above 0.5 in 4% of games, above 0.6 in 1%.

**abs**

| season | n | base_acc | base_ll | vegas_acc | vegas_ll | vegas+ffa_acc | vegas+ffa_ll | +history_acc | +history_ll | gbm_acc | gbm_ll |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 256.0 | 0.281 | 1.5806 | 0.316 | 1.5194 | 0.328 | 1.5667 | 0.277 | 1.5873 | 0.293 | 1.8254 |
| 2019 | 256.0 | 0.281 | 1.566 | 0.355 | 1.5111 | 0.328 | 1.5286 | 0.324 | 1.5664 | 0.301 | 1.6662 |
| 2020 | 224.0 | 0.246 | 1.6013 | 0.321 | 1.5131 | 0.277 | 1.5903 | 0.295 | 1.6089 | 0.362 | 1.6221 |
| 2021 | 272.0 | 0.276 | 1.5635 | 0.397 | 1.4955 | 0.371 | 1.5017 | 0.335 | 1.512 | 0.32 | 1.6518 |
| 2022 | 271.0 | 0.207 | 1.6339 | 0.284 | 1.5689 | 0.28 | 1.591 | 0.284 | 1.5898 | 0.251 | 1.6841 |
| 2023 | 272.0 | 0.29 | 1.569 | 0.29 | 1.5627 | 0.276 | 1.574 | 0.279 | 1.5768 | 0.32 | 1.589 |
| 2024 | 272.0 | 0.257 | 1.5849 | 0.294 | 1.5279 | 0.279 | 1.5372 | 0.287 | 1.5484 | 0.316 | 1.5838 |
| 2025 | 272.0 | 0.272 | 1.5603 | 0.342 | 1.5232 | 0.305 | 1.5301 | 0.342 | 1.5354 | 0.331 | 1.5527 |
| ALL | 2095.0 | 0.264 | 1.582 | 0.325 | 1.528 | 0.306 | 1.552 | 0.303 | 1.566 | 0.312 | 1.647 |

Top predicted probability (+history logit): median 0.32, above 0.5 in 9% of games, above 0.6 in 3%.

**rel**

| season | n | base_acc | base_ll | vegas_acc | vegas_ll | vegas+ffa_acc | vegas+ffa_ll | +history_acc | +history_ll | gbm_acc | gbm_ll |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 256.0 | 0.281 | 1.5828 | 0.312 | 1.596 | 0.301 | 1.6236 | 0.285 | 1.6333 | 0.289 | 1.8653 |
| 2019 | 256.0 | 0.281 | 1.561 | 0.367 | 1.5345 | 0.344 | 1.5587 | 0.297 | 1.6039 | 0.309 | 1.7074 |
| 2020 | 224.0 | 0.246 | 1.6053 | 0.339 | 1.5707 | 0.281 | 1.6459 | 0.277 | 1.6589 | 0.335 | 1.67 |
| 2021 | 272.0 | 0.276 | 1.558 | 0.39 | 1.5005 | 0.379 | 1.5133 | 0.357 | 1.5262 | 0.279 | 1.6671 |
| 2022 | 271.0 | 0.207 | 1.624 | 0.277 | 1.6077 | 0.262 | 1.635 | 0.284 | 1.6244 | 0.306 | 1.6863 |
| 2023 | 272.0 | 0.29 | 1.5637 | 0.29 | 1.562 | 0.294 | 1.5807 | 0.301 | 1.5883 | 0.309 | 1.5972 |
| 2024 | 272.0 | 0.257 | 1.5842 | 0.298 | 1.5472 | 0.287 | 1.5543 | 0.283 | 1.5617 | 0.294 | 1.6194 |
| 2025 | 272.0 | 0.272 | 1.5601 | 0.375 | 1.5286 | 0.342 | 1.5354 | 0.327 | 1.5418 | 0.338 | 1.566 |
| ALL | 2095.0 | 0.264 | 1.58 | 0.331 | 1.556 | 0.311 | 1.581 | 0.301 | 1.592 | 0.307 | 1.672 |

Top predicted probability (+history logit): median 0.31, above 0.5 in 8% of games, above 0.6 in 3%.

What the closing line already implies (abs taxonomy, % of games):

| spread | Fav blowout | Shootout | Slog | Standard | Upset |
|---|---|---|---|---|---|
| 0-2.5 | 13 | 18 | 24 | 25 | 19 |
| 3-6.5 | 23 | 14 | 13 | 18 | 33 |
| 7-10 | 37 | 15 | 10 | 18 | 21 |
| 10.5+ | 52 | 8 | 11 | 17 | 13 |

| total line | Fav blowout | Shootout | Slog | Standard | Upset |
|---|---|---|---|---|---|
| <=41.5 | 25 | 8 | 21 | 18 | 28 |
| 42-45.5 | 28 | 11 | 18 | 20 | 23 |
| 46-49.5 | 27 | 17 | 11 | 20 | 25 |
| 50+ | 24 | 26 | 5 | 18 | 27 |

## c) The oracle: MAE if the script were known before kickoff

Per (script, side, position) multiplier on the FFA line = sum(actual)/sum(FFA) in the training seasons, shrunk toward 1 with a 50-row pseudo-count, cells under 40 rows left at 1. Fitted on seasons before the test season, applied with the TRUE script (oracle) and with the pregame model's pick (what a line-reading fan would do). Rows with an FFA line of 0.5 or less are never touched.

**traj**

| season | n | base | oracle | pregame_pick | oracle_gain_% | pick_gain_% |
|---|---|---|---|---|---|---|
| 2018 | 4753.0 | 4.7696 | 4.6472 | 4.8441 | 2.57 | -1.56 |
| 2019 | 4622.0 | 4.8739 | 4.7863 | 4.9438 | 1.8 | -1.43 |
| 2020 | 3820.0 | 5.03 | 4.9931 | 5.1841 | 0.73 | -3.06 |
| 2021 | 4976.0 | 4.7963 | 4.7294 | 4.8578 | 1.39 | -1.28 |
| 2022 | 4947.0 | 4.5948 | 4.5006 | 4.6312 | 2.05 | -0.79 |
| 2023 | 5102.0 | 4.3756 | 4.2462 | 4.3579 | 2.96 | 0.4 |
| 2024 | 4994.0 | 4.5391 | 4.4205 | 4.5693 | 2.61 | -0.67 |
| 2025 | 5150.0 | 4.4427 | 4.3072 | 4.49 | 3.05 | -1.06 |
| ALL | 38364.0 | 4.6777 | 4.5788 | 4.7348 | 2.11 | -1.22 |

By position (pooled test seasons, oracle):

| position | base | oracle | gain_% |
|---|---|---|---|
| QB | 5.567 | 5.044 | 9.39 |
| RB | 4.68 | 4.602 | 1.67 |
| WR | 4.858 | 4.807 | 1.05 |
| TE | 3.702 | 3.733 | -0.84 |

By FFA line tier:

| FFA line | base | oracle | gain_% |
|---|---|---|---|
| 0-5 | 2.815 | 2.807 | 0.28 |
| 5-10 | 4.618 | 4.583 | 0.76 |
| 10-15 | 5.894 | 5.727 | 2.83 |
| 15+ | 6.756 | 6.456 | 4.44 |

Reader accuracy curve: expected MAE when the reader names the right script a% of the time and otherwise picks in proportion to the pregame model's probabilities over the other scripts. Baseline 4.6627; break-even accuracy 54%; base rate of the most common script 26%; pregame model accuracy is in (b). The baseline here is row-pooled across test seasons, the per-season table above averages season MAEs, hence the small difference.

| reader accuracy | 0% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| MAE | 4.7827 | 4.7385 | 4.7164 | 4.6943 | 4.6722 | 4.6501 | 4.628 | 4.6059 | 4.5838 | 4.5617 |

**abs**

| season | n | base | oracle | pregame_pick | oracle_gain_% | pick_gain_% |
|---|---|---|---|---|---|---|
| 2018 | 4753.0 | 4.7696 | 4.6863 | 4.8703 | 1.75 | -2.11 |
| 2019 | 4622.0 | 4.8739 | 4.8075 | 4.9669 | 1.36 | -1.91 |
| 2020 | 3820.0 | 5.03 | 4.9718 | 5.1648 | 1.16 | -2.68 |
| 2021 | 4976.0 | 4.7963 | 4.7074 | 4.9268 | 1.85 | -2.72 |
| 2022 | 4947.0 | 4.5948 | 4.4838 | 4.6636 | 2.42 | -1.5 |
| 2023 | 5102.0 | 4.3756 | 4.2033 | 4.4023 | 3.94 | -0.61 |
| 2024 | 4994.0 | 4.5391 | 4.4218 | 4.6145 | 2.58 | -1.66 |
| 2025 | 5150.0 | 4.4427 | 4.3149 | 4.5163 | 2.88 | -1.66 |
| ALL | 38364.0 | 4.6777 | 4.5746 | 4.7657 | 2.2 | -1.88 |

By position (pooled test seasons, oracle):

| position | base | oracle | gain_% |
|---|---|---|---|
| QB | 5.567 | 5.011 | 9.99 |
| RB | 4.68 | 4.604 | 1.62 |
| WR | 4.858 | 4.804 | 1.11 |
| TE | 3.702 | 3.735 | -0.89 |

By FFA line tier:

| FFA line | base | oracle | gain_% |
|---|---|---|---|
| 0-5 | 2.815 | 2.814 | 0.04 |
| 5-10 | 4.618 | 4.595 | 0.5 |
| 10-15 | 5.894 | 5.707 | 3.17 |
| 15+ | 6.756 | 6.427 | 4.87 |

Reader accuracy curve: expected MAE when the reader names the right script a% of the time and otherwise picks in proportion to the pregame model's probabilities over the other scripts. Baseline 4.6627; break-even accuracy 63%; base rate of the most common script 26%; pregame model accuracy is in (b). The baseline here is row-pooled across test seasons, the per-season table above averages season MAEs, hence the small difference.

| reader accuracy | 0% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| MAE | 4.844 | 4.7867 | 4.758 | 4.7294 | 4.7007 | 4.6721 | 4.6434 | 4.6148 | 4.5861 | 4.5574 |

**rel**

| season | n | base | oracle | pregame_pick | oracle_gain_% | pick_gain_% |
|---|---|---|---|---|---|---|
| 2018 | 4753.0 | 4.7696 | 4.6875 | 4.8651 | 1.72 | -2.0 |
| 2019 | 4622.0 | 4.8739 | 4.801 | 4.9461 | 1.5 | -1.48 |
| 2020 | 3820.0 | 5.03 | 4.9374 | 5.1404 | 1.84 | -2.19 |
| 2021 | 4976.0 | 4.7963 | 4.7144 | 4.919 | 1.71 | -2.56 |
| 2022 | 4947.0 | 4.5948 | 4.496 | 4.676 | 2.15 | -1.77 |
| 2023 | 5102.0 | 4.3756 | 4.2184 | 4.4176 | 3.59 | -0.96 |
| 2024 | 4994.0 | 4.5391 | 4.417 | 4.6081 | 2.69 | -1.52 |
| 2025 | 5150.0 | 4.4427 | 4.3141 | 4.502 | 2.89 | -1.33 |
| ALL | 38364.0 | 4.6777 | 4.5732 | 4.7593 | 2.23 | -1.74 |

By position (pooled test seasons, oracle):

| position | base | oracle | gain_% |
|---|---|---|---|
| QB | 5.567 | 4.994 | 10.29 |
| RB | 4.68 | 4.604 | 1.62 |
| WR | 4.858 | 4.809 | 1.01 |
| TE | 3.702 | 3.736 | -0.92 |

By FFA line tier:

| FFA line | base | oracle | gain_% |
|---|---|---|---|
| 0-5 | 2.815 | 2.815 | 0.0 |
| 5-10 | 4.618 | 4.603 | 0.32 |
| 10-15 | 5.894 | 5.702 | 3.26 |
| 15+ | 6.756 | 6.416 | 5.03 |

Reader accuracy curve: expected MAE when the reader names the right script a% of the time and otherwise picks in proportion to the pregame model's probabilities over the other scripts. Baseline 4.6627; break-even accuracy 64%; base rate of the most common script 26%; pregame model accuracy is in (b). The baseline here is row-pooled across test seasons, the per-season table above averages season MAEs, hence the small difference.

| reader accuracy | 0% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| MAE | 4.8493 | 4.7908 | 4.7616 | 4.7324 | 4.7032 | 4.674 | 4.6448 | 4.6156 | 4.5863 | 4.5571 |

### Upper bounds: exact final score, and the team's exact fantasy total

| season | n | base | exact_score | exact_team_fantasy_total | score_gain_% | team_total_gain_% |
|---|---|---|---|---|---|---|
| 2018 | 4753.0 | 4.7696 | 4.532 | 4.2905 | 4.98 | 10.04 |
| 2019 | 4622.0 | 4.8739 | 4.6659 | 4.3502 | 4.27 | 10.74 |
| 2020 | 3820.0 | 5.03 | 4.7743 | 4.4716 | 5.08 | 11.1 |
| 2021 | 4976.0 | 4.7963 | 4.5829 | 4.2724 | 4.45 | 10.92 |
| 2022 | 4947.0 | 4.5948 | 4.4161 | 4.1274 | 3.89 | 10.17 |
| 2023 | 5102.0 | 4.3756 | 4.192 | 3.931 | 4.2 | 10.16 |
| 2024 | 4994.0 | 4.5391 | 4.3756 | 4.1054 | 3.6 | 9.55 |
| 2025 | 5150.0 | 4.4427 | 4.2399 | 3.9585 | 4.56 | 10.9 |
| ALL | 38364.0 | 4.6777 | 4.4723 | 4.1884 | 4.39 | 10.46 |

exact_score = ridge on the realized margin, total and total-surprise (x position, x FFA line): the most a perfectly-called final score could give. exact_team_fantasy_total = every player on a side scaled by that side's realized fantasy total / projected total: the ceiling for ANY team-level information, not attainable.

### Does the oracle gain survive on Model_Burke? (league eval rows 2023-24, n=10096, factors fitted 2016-22)

| taxonomy | ffa | ffa_oracle | model_burke | model_burke_oracle |
|---|---|---|---|---|
| traj | 4.4565 | 4.332 | 4.4243 | 4.3016 |
| abs | 4.4565 | 4.3114 | 4.4243 | 4.2852 |
| rel | 4.4565 | 4.3169 | 4.4243 | 4.2858 |

## Summary

| taxonomy | base | oracle | oracle_gain_pct | pregame_pick | pick_gain_pct |
|---|---|---|---|---|---|
| traj | 4.6777 | 4.5788 | 2.11 | 4.7348 | -1.22 |
| abs | 4.6777 | 4.5746 | 2.2 | 4.7657 | -1.88 |
| rel | 4.6777 | 4.5732 | 2.23 | 4.7593 | -1.74 |

