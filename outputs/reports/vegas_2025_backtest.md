# Our model vs Vegas — 2025 backtest

*4724 player-weeks, 543 players, weeks 5-22 of 2025. Walk-forward: every prediction is made from strictly earlier weeks. Target is weekly PPR.*

`A_football` is our model with no market data. `B_market` is built from the Vegas player prop lines and nothing else. Same rows, same target, so the comparison is like-for-like.

## Answer: a dead heat

Our football-only model and the Vegas-lines model are **statistically indistinguishable** on last season: MAE 4.657 vs 4.658, a paired difference of **-0.001** (95% CI -0.100 to +0.096, clustered by player). Restricted to the top 150 players by season total the answer is the same (+0.030, CI -0.141 to +0.196).

That is a genuinely good result — the market prices these players with far more information than we have — but see the warning at the bottom before reading it as an edge.

## Accuracy, all 2025 player-weeks

| model | MAE | RMSE | Spearman | R² | bias |
|---|---:|---:|---:|---:|---:|
| **ours** (football only) | 4.657 | 6.359 | 0.663 | 0.369 | +0.433 |
| **Vegas** (prop lines only) | 4.658 | 6.285 | 0.620 | 0.384 | +0.091 |
| ours + Vegas | 4.462 | 6.137 | 0.688 | 0.413 | +0.242 |
| residual (2-stage) | 4.546 | 6.351 | 0.669 | 0.371 | +0.055 |
| market baseline | 4.658 | 6.285 | 0.620 | 0.384 | +0.091 |
| rolling average | 4.914 | 7.050 | 0.625 | 0.225 | +0.085 |

## They know different things

| comparison | ΔMAE | 95% CI | significant |
|---|---:|---:|:--:|
| ours − Vegas | -0.001 | -0.100 to +0.096 | no |
| ours+Vegas − ours | -0.194 | -0.247 to -0.144 | yes |
| ours+Vegas − Vegas | -0.196 | -0.273 to -0.123 | yes |
| ours − rolling baseline | -0.257 | -0.345 to -0.167 | yes |
| ours − Vegas (top 150) | +0.030 | -0.141 to +0.196 | no |
| ours+Vegas − ours (top 150) | -0.155 | -0.258 to -0.055 | yes |

Combining the two beats **either** alone, significantly. Vegas carries information our features miss, and our features carry information the lines miss — they are complementary, not redundant.

## Where each one wins

| position | n | ours | Vegas | ours+Vegas |
|---|---:|---:|---:|---:|
| QB | 505 | 6.61 | 6.45 | 6.34 |
| RB | 1237 | 4.67 | 4.87 | 4.50 |
| WR | 1965 | 4.59 | 4.50 | 4.35 |
| TE | 1017 | 3.80 | 3.82 | 3.69 |

Vegas is better on QB, we are better on RB, WR and TE are close.

## Ranking vs calibration

| model | mean within-week Spearman |
|---|---:|
| A_football | 0.668 |
| B_market | 0.632 |
| C_full | 0.696 |
| D_residual | 0.677 |
| baseline_rolling | 0.624 |

We **order** players better (within-week Spearman 0.668 vs 0.632), which is what a draft board needs. Vegas is better **calibrated** on magnitude (RMSE 6.285 vs 6.359, bias +0.091 vs +0.433). Our model systematically over-predicts by about +0.43 PPR a week.

## What this does NOT say

- **It is not a betting result.** Matching the market on MAE is not beating it for money: you have to clear the vig, and the settled-bet study (`outputs/reports/prop_ev_model.md`) found the model is **not +EV** against closing prices — the line won all six markets on MAE there, our P(over) was poorly calibrated, and essentially all realized profit came from line shopping.

- **`B_market` is a model of the lines, not the lines themselves.** It re-fits prop lines to weekly PPR, which adds its own error. A pure closing-line benchmark needs `db/nfl_odds.db`, which is not in this container.

- **No outcome-conditioned subsets.** Slicing on "weeks the player actually scored 8+" made us look 0.31 better, but that is selection on the dependent variable and it simply rewards our upward bias. It is excluded deliberately.

