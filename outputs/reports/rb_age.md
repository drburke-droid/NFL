# RB age — one rejected fix, one shipped

Walk-forward on `season_ffa_predictions` joined to `season_dataset.age` (582 RB player-seasons, 2016-2025).

## The bias is real

Our model's RB relative error drifts **-1.87% per year of age** (p=0.0108) — young RBs under-projected, 28+ over-projected. The model inherits this from FFA almost exactly (FFA -1.93%/yr), so Blend does not fix it.

## REJECTED: an age multiplier on RB projections

| lambda | mean MAE | vs board | MAE wins | mean rank | rank wins |
|---|---|---|---|---|---|
| 0.00 | 3.0014 | +0.00% | 0/7 | 0.6276 | 0/7 |
| 0.15 | 2.9996 | +0.06% | 4/7 | 0.6276 | 3/7 |
| 0.25 | 3.0027 | -0.04% | 4/7 | 0.6275 | 2/7 |
| 0.35 | 3.0090 | -0.25% | 4/7 | 0.6289 | 4/7 |
| 0.50 | 3.0262 | -0.83% | 4/7 | 0.6314 | 4/7 |
| 0.75 | 3.0813 | -2.66% | 3/7 | 0.6292 | 5/7 |
| 1.00 | 3.1466 | -4.84% | 3/7 | 0.6293 | 4/7 |

Every dose either does nothing or hurts. **Not shipped.**

## SHIPPED: an age tiebreaker

Two players, same season, projections within 10%, age gap >= 4 years — how often does the younger outscore the older?

| pos | younger wins | rate | p |
|---|---|---|---|
| RB | 501/854 | 58.7% | 0.0000 |
| QB | 526/974 | 54.0% | 0.0136 |
| WR | 1046/1990 | 52.6% | 0.0235 |
| TE | 220/420 | 52.4% | 0.3539 |

Among draftable RBs (proj >= 10 PPG) it holds: **238/409 = 58.2%**, p=0.0011.

`docs/index.html` `rbAgeTag()` surfaces this as **⬆ YOUTH** (RB <= 24) and **⚠ RB AGE** (RB >= 28). Context only — projections and prices are unchanged.
