# Anatomy of the largest misses (2017-2025)

Played rows with an incumbent prediction: 43,564. A big miss = top 5% of |actual - model| within each season (threshold 12.5-13.7 points). 2,183 big misses: 1,854 booms (actual above the model) and 329 busts.

- They are 5.0% of rows and 18.7% of all absolute error; mean size 17.2 points (booms 17.6, busts 14.9).
- Booms outnumber busts 85% to 15%: the right tail is where the big errors live. The biggest single misses: devon achane 2023 wk3 51 vs 4; will fuller 2019 wk5 54 vs 12; derrick henry 2018 wk14 48 vs 8; tyreek hill 2020 wk12 58 vs 19; joe mixon 2022 wk9 55 vs 16.
- FFA misses the same rows: |actual - FFA| clears the same threshold on 91% of them, so these are not correction errors; the consensus missed too.

## Who the big misses are (share of big misses vs share of all rows; lift > 1 = over-represented)

**position**

| position | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| QB | 13% | 1.21 | 3.38 | 7.7% |
| RB | 26% | 1.11 | 0.78 | 5.3% |
| WR | 40% | 1.05 | 0.86 | 5.1% |
| TE | 21% | 0.64 | 0.10 | 2.8% |

**projection**

| projection | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| <5 | 34% | 0.47 | 0.00 | 2.0% |
| 5-10 | 27% | 0.98 | 0.01 | 4.2% |
| 10-15 | 22% | 1.39 | 0.59 | 6.4% |
| 15-20 | 14% | 1.53 | 4.36 | 9.8% |
| 20+ | 3% | 1.77 | 8.04 | 13.6% |

**tier within position**

| tier within position | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| top6 | 9% | 1.72 | 5.48 | 11.4% |
| 7-12 | 9% | 1.54 | 2.73 | 8.6% |
| 13-24 | 17% | 1.24 | 1.39 | 6.3% |
| 25-36 | 17% | 1.11 | 0.28 | 4.9% |
| 37+ | 49% | 0.65 | 0.00 | 2.8% |

**week**

| week | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| wk1-2 | 12% | 1.01 | 1.00 | 5.1% |
| wk3-4 | 12% | 1.16 | 0.73 | 5.5% |
| wk5-13 | 50% | 0.94 | 1.04 | 4.8% |
| wk14+ | 26% | 1.03 | 1.04 | 5.2% |

**injury tag**

| injury tag | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| not Q | 97% | 1.00 | 0.99 | 5.0% |
| Q | 3% | 1.02 | 1.31 | 5.3% |

**recent volatility**

| recent volatility | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| volatile | 27% | 1.33 | 1.93 | 7.1% |
| steady | 27% | 0.67 | 0.34 | 3.1% |
| mid | 27% | 1.07 | 0.88 | 5.2% |
| no history | 18% | 0.89 | 0.76 | 4.4% |

**experience**

| experience | rows | boom lift | bust lift | big-miss rate |
|---|---|---|---|---|
| new | 50% | 0.92 | 0.79 | 4.5% |
| established | 50% | 1.08 | 1.21 | 5.5% |

**Repeat offenders** (players with 30+ rows, ranked by big misses above what their projection level predicts):

| player | rows | big misses | expected | of which booms |
|---|---|---|---|---|
| lamar jackson | 114 | 20 | 11.8 | 12 |
| derrick henry | 138 | 20 | 11.9 | 17 |
| amari cooper | 119 | 18 | 10.2 | 14 |
| amonra st brown | 82 | 16 | 8.9 | 12 |
| aj brown | 104 | 17 | 10.4 | 12 |
| jamarr chase | 78 | 16 | 9.5 | 12 |
| josh allen | 125 | 20 | 13.5 | 17 |
| jonathan taylor | 82 | 15 | 8.5 | 11 |
| mike evans | 128 | 20 | 13.6 | 14 |
| adam thielen | 127 | 17 | 10.6 | 14 |

Season-to-season correlation of a player's big-miss rate (players with 8+ games both seasons, n = 1,889): +0.22. Players in the top volatility tercile last season have a big-miss rate of 7.1% vs 3.1% for the steadiest.

## What happened on the day

**Booms** (actual far above the model): where the extra points came from, per row, vs a normal row of the same projection band.

| projection | n booms | actual | model | TD+other pts (boom) | TD+other pts (normal) | touches (boom) | touches (normal) | yards/touch (boom) | (normal) |
|---|---|---|---|---|---|---|---|---|---|
| <5 | 294 | 19.6 | 2.8 | 7.8 | 0.5 | 11.5 | 3.1 | 9.0 | 4.5 |
| 5-10 | 490 | 24.7 | 7.2 | 8.8 | 1.1 | 11.3 | 5.9 | 9.9 | 5.4 |
| 10-15 | 563 | 29.4 | 12.0 | 10.5 | 2.0 | 19.1 | 14.1 | 8.3 | 4.8 |
| 15-20 | 405 | 34.7 | 16.4 | 13.3 | 3.8 | 24.8 | 22.0 | 6.9 | 4.2 |
| 20+ | 102 | 39.0 | 20.8 | 14.6 | 5.9 | 28.5 | 27.0 | 7.3 | 4.3 |

Across all booms the excess over a normal row of the same band is 17.6 points, of which touchdowns/other account for 8.3 (47%); the rest is volume and yardage. 51% of booms had two or more touchdowns' worth of TD points.

**Busts** (actual far below the model): volume collapse, early exit, or efficiency?

| projection | n busts | model | actual | touches (bust) | touches (normal) | snap share vs own norm (bust) | (normal) | left early (<50% of norm) | zero-touch |
|---|---|---|---|---|---|---|---|---|---|
| 10-15 | 42 | 13.2 | -1.0 | 14.0 | 14.1 | 0.72 | 1.02 | 29% | 7% |
| 15-20 | 204 | 17.0 | 2.3 | 11.7 | 22.0 | 0.84 | 1.01 | 28% | 18% |
| 20+ | 82 | 21.4 | 5.5 | 18.4 | 27.0 | 0.98 | 1.01 | 17% | 6% |

Busts with a snap-share record: 97%. Of those, 25% played under half their usual snaps (an in-game injury or a benching the pre-game data could not see) and 24% were between 50-90%; 48% played a full complement and simply produced nothing.

## Can a big miss be seen coming?

Walk-forward logistic regression on pre-game features (projection, volatility, recent points, Q-tag, week, spread/total/home, experience, deep share, route share, target share, FFA stat spreads, usual snap share), 2019-2025:

| target | AUC (all features) | AUC (projection only) | big-miss rate in the model's top decile | base rate |
|---|---|---|---|---|
| any big miss | 0.679 | 0.682 | 12.1% | 5.0% |
| boom | 0.634 | 0.636 | 7.7% | 4.2% |
| bust | 0.909 | 0.909 | 4.6% | 0.8% |

Big-miss rate by the strongest single pre-game flags (all rows):

| flag | rows | big-miss rate | boom rate | bust rate |
|---|---|---|---|---|
| projection 20+ | 1,351 | 13.6% | 7.5% | 6.1% |
| projection 15-20 | 6,202 | 9.8% | 6.5% | 3.3% |
| projection < 5 | 14,812 | 2.0% | 2.0% | 0.0% |
| volatile (top tercile std3) | 11,919 | 7.1% | 5.7% | 1.5% |
| steady (bottom tercile) | 11,919 | 3.1% | 2.9% | 0.3% |
| Q tag | 1,110 | 5.3% | 4.3% | 1.0% |
| weeks 1-2 | 5,302 | 5.1% | 4.3% | 0.8% |
| game total 50+ | 6,059 | 6.1% | 4.8% | 1.4% |
| game total < 40 | 4,679 | 4.2% | 3.8% | 0.4% |
| spread |7|+ underdog | 6,192 | 4.2% | 3.6% | 0.6% |
| favourite by 7+ | 6,396 | 5.5% | 4.4% | 1.1% |
| new (< 6 games) | 21,869 | 4.5% | 3.9% | 0.6% |
| usual snap share < 50% | 15,268 | 3.1% | 3.0% | 0.1% |
| usual snap share 90%+ | 5,999 | 9.0% | 6.4% | 2.6% |

Market disagreement (2023-25 rows with a DK receiving-yards line, n = 9,092): big-miss rate and which way the miss went.

| DK line vs FFA rec yds | rows | big-miss rate | boom | bust | mean residual vs model |
|---|---|---|---|---|---|
| DK 10+ under FFA | 826 | 5.2% | 4.6% | 0.6% | -0.16 |
| DK 4-10 under | 4,143 | 5.6% | 5.1% | 0.5% | +0.52 |
| agree | 3,954 | 6.3% | 5.9% | 0.4% | +1.11 |
| DK 4-10 over | 142 | 4.9% | 4.2% | 0.7% | +1.83 |
