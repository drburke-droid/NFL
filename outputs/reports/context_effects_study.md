# Context effects study (2026-09-09)

Box scores 2011-25 REG: 83,255 player-weeks. FFA-baseline sample 2023-25: 10,193 player-weeks (FFA >= 3). Own-r6 sample: 59,339.

## A. Opponent points allowed by position (prior-6, z-scored within week)

Residual vs opponent DvP z-score. Slope = residual points per 1 SD of opponent generosity.

| position | baseline | slope | corr | n |
|---|---|---|---|---|
| QB | FFA | +0.168 | +0.021 | 1,179 |
| QB | own-r6 | +0.792 | +0.096 | 5,981 |
| RB | FFA | +0.256 | +0.038 | 2,311 |
| RB | own-r6 | +0.565 | +0.075 | 12,172 |
| WR | FFA | -0.067 | -0.010 | 3,775 |
| WR | own-r6 | +0.298 | +0.040 | 19,592 |
| TE | FFA | +0.239 | +0.041 | 1,509 |
| TE | own-r6 | +0.329 | +0.053 | 8,557 |

FFA residual by opponent-DvP quintile (all positions):

| bin | mean residual | n |
|---|---|---|
| (-2.984, -0.869] | +0.32 | 1757 |
| (-0.869, -0.287] | +0.14 | 1755 |
| (-0.287, 0.196] | -0.02 | 1753 |
| (0.196, 0.848] | +0.26 | 1757 |
| (0.848, 3.248] | +0.78 | 1752 |

WR only:

| bin | mean residual | n |
|---|---|---|
| (-2.82, -0.871] | +0.19 | 755 |
| (-0.871, -0.266] | -0.13 | 756 |
| (-0.266, 0.243] | +0.05 | 754 |
| (0.243, 0.857] | -0.04 | 757 |
| (0.857, 3.104] | +0.28 | 753 |

RB only:

| bin | mean residual | n |
|---|---|---|
| (-2.4699999999999998, -0.871] | +0.18 | 463 |
| (-0.871, -0.32] | +0.21 | 462 |
| (-0.32, 0.143] | -0.31 | 462 |
| (0.143, 0.826] | +0.12 | 462 |
| (0.826, 3.055] | +1.13 | 462 |

## B. Vegas game line

| position | baseline | x | slope (pts per unit) | corr | n |
|---|---|---|---|---|---|
| QB | FFA | itt | +0.087 | +0.043 | 1,372 |
| QB | FFA | spread | -0.043 | -0.034 | 1,372 |
| QB | FFA | game_total | +0.045 | +0.026 | 1,372 |
| QB | own-r6 | itt | +0.014 | +0.007 | 5,984 |
| QB | own-r6 | spread | -0.022 | -0.017 | 5,984 |
| QB | own-r6 | game_total | -0.026 | -0.014 | 5,984 |
| RB | FFA | itt | +0.120 | +0.069 | 2,688 |
| RB | FFA | spread | -0.061 | -0.057 | 2,688 |
| RB | FFA | game_total | +0.058 | +0.039 | 2,688 |
| RB | own-r6 | itt | +0.081 | +0.043 | 12,177 |
| RB | own-r6 | spread | -0.051 | -0.045 | 12,177 |
| RB | own-r6 | game_total | +0.018 | +0.011 | 12,177 |
| WR | FFA | itt | +0.028 | +0.016 | 4,373 |
| WR | FFA | spread | -0.006 | -0.005 | 4,373 |
| WR | FFA | game_total | +0.031 | +0.020 | 4,373 |
| WR | own-r6 | itt | +0.012 | +0.006 | 19,599 |
| WR | own-r6 | spread | -0.006 | -0.006 | 19,599 |
| WR | own-r6 | game_total | +0.005 | +0.003 | 19,599 |
| TE | FFA | itt | +0.053 | +0.035 | 1,760 |
| TE | FFA | spread | -0.026 | -0.028 | 1,760 |
| TE | FFA | game_total | +0.029 | +0.022 | 1,760 |
| TE | own-r6 | itt | +0.013 | +0.008 | 8,561 |
| TE | own-r6 | spread | -0.011 | -0.012 | 8,561 |
| TE | own-r6 | game_total | -0.003 | -0.002 | 8,561 |

FFA residual by implied team total quintile (all positions):

| bin | mean residual | n |
|---|---|---|
| (11.499, 19.0] | -0.05 | 2272 |
| (19.0, 21.25] | +0.23 | 1937 |
| (21.25, 23.25] | +0.36 | 2025 |
| (23.25, 25.5] | +0.13 | 2189 |
| (25.5, 32.25] | +0.59 | 1770 |

FFA residual by spread quintile (negative = favoured):

| bin | mean residual | n |
|---|---|---|
| (-19.501, -5.5] | +0.45 | 2170 |
| (-5.5, -2.5] | +0.44 | 2328 |
| (-2.5, 2.5] | +0.03 | 1874 |
| (2.5, 5.5] | +0.26 | 2005 |
| (5.5, 19.5] | -0.10 | 1816 |

### Walk-forward: does correcting the FFA baseline with these help? (fit on prior weeks, test 2024-25)

| position | features | MAE baseline | MAE corrected | gain |
|---|---|---|---|---|
| QB | itt only | 6.283 | 6.271 | +0.18% |
| QB | dvp only | 6.283 | 6.295 | -0.20% |
| QB | itt+spread | 6.283 | 6.276 | +0.11% |
| QB | itt+spread+dvp | 6.283 | 6.285 | -0.04% |
| RB | itt only | 5.139 | 5.172 | -0.64% |
| RB | dvp only | 5.139 | 5.187 | -0.94% |
| RB | itt+spread | 5.139 | 5.167 | -0.55% |
| RB | itt+spread+dvp | 5.139 | 5.168 | -0.58% |
| WR | itt only | 4.979 | 5.004 | -0.51% |
| WR | dvp only | 4.979 | 5.007 | -0.58% |
| WR | itt+spread | 4.979 | 5.012 | -0.67% |
| WR | itt+spread+dvp | 4.979 | 5.017 | -0.77% |
| TE | itt only | 4.448 | 4.548 | -2.25% |
| TE | dvp only | 4.448 | 4.541 | -2.08% |
| TE | itt+spread | 4.448 | 4.554 | -2.37% |
| TE | itt+spread+dvp | 4.448 | 4.546 | -2.20% |

## C. When a starter is absent, where do the points go?

Starter = team's leader at the position by prior-3-game usage (carries for RB, targets for WR/TE, attempts for QB), who PLAYED the previous team game. Absence = no box-score row that week while the team played. Deltas are vs the team's / player's own prior-3-game averages (2012-25 REG).


### QB1 absent — 520 team-games (vs 5,997 with the starter playing); avg vacated = 14.3 PPR

| quantity | starter absent | starter plays | difference |
|---|---|---|---|
| next man up, gain vs his own r3 (PPR) | +7.12 | +0.10 | +7.02 |
| other same-pos players, gain | -0.13 | -0.00 | -0.12 |
| RBs total gain | +1.22 | +0.62 | +0.60 |
| WRs total gain | +0.04 | +0.27 | -0.22 |
| TEs total gain | +0.58 | +0.25 | +0.34 |
| team pass attempts, delta | -0.73 | -0.16 | -0.57 |
| team carries, delta | -0.01 | +0.11 | -0.12 |
| team pass rate, delta | -0.01 | -0.00 | -0.00 |
| team total PPR, delta | -2.87 | -0.60 | -2.27 |

Share of the vacated QB1 points (net of the normal-week drift): next man up **55%**, other QBs **-1%**, other positions **14%**, team total change -16% of vacated.
Median next-man share 47%; next man absorbs >=50% in 49% of cases.

### RB1 absent — 529 team-games (vs 5,858 with the starter playing); avg vacated = 13.7 PPR

| quantity | starter absent | starter plays | difference |
|---|---|---|---|
| next man up, gain vs his own r3 (PPR) | +8.30 | +1.46 | +6.84 |
| other same-pos players, gain | +1.04 | -1.07 | +2.11 |
| QBs total gain | +0.71 | +0.33 | +0.38 |
| WRs total gain | +0.95 | +0.28 | +0.66 |
| TEs total gain | +0.46 | +0.28 | +0.18 |
| team pass attempts, delta | -0.25 | -0.14 | -0.10 |
| team carries, delta | -0.36 | +0.07 | -0.43 |
| team pass rate, delta | +0.00 | -0.00 | +0.00 |
| team total PPR, delta | -1.92 | -0.42 | -1.50 |

Share of the vacated RB1 points (net of the normal-week drift): next man up **60%**, other RBs **17%**, other positions **5%**, team total change -11% of vacated.
Median next-man share 45%; next man absorbs >=50% in 47% of cases.

### WR1 absent — 435 team-games (vs 6,339 with the starter playing); avg vacated = 14.3 PPR

| quantity | starter absent | starter plays | difference |
|---|---|---|---|
| next man up, gain vs his own r3 (PPR) | +7.05 | +4.15 | +2.90 |
| other same-pos players, gain | +0.30 | -3.25 | +3.55 |
| QBs total gain | +0.29 | +0.38 | -0.08 |
| RBs total gain | +1.54 | +0.62 | +0.92 |
| TEs total gain | +0.86 | +0.29 | +0.57 |
| team pass attempts, delta | -1.05 | -0.12 | -0.93 |
| team carries, delta | +0.22 | +0.12 | +0.10 |
| team pass rate, delta | -0.01 | -0.00 | -0.01 |
| team total PPR, delta | -5.18 | -0.21 | -4.97 |

Share of the vacated WR1 points (net of the normal-week drift): next man up **20%**, other WRs **24%**, other positions **7%**, team total change -35% of vacated.
Median next-man share 16%; next man absorbs >=50% in 28% of cases.

### TE1 absent — 464 team-games (vs 5,161 with the starter playing); avg vacated = 9.5 PPR

| quantity | starter absent | starter plays | difference |
|---|---|---|---|
| next man up, gain vs his own r3 (PPR) | +3.40 | +0.71 | +2.70 |
| other same-pos players, gain | -0.16 | -0.48 | +0.32 |
| QBs total gain | +0.95 | +0.32 | +0.63 |
| RBs total gain | +1.21 | +0.58 | +0.63 |
| WRs total gain | +2.24 | +0.33 | +1.90 |
| team pass attempts, delta | -2.05 | -0.33 | -1.72 |
| team carries, delta | +0.83 | +0.12 | +0.71 |
| team pass rate, delta | -0.02 | -0.00 | -0.02 |
| team total PPR, delta | -2.98 | -0.55 | -2.43 |

Share of the vacated TE1 points (net of the normal-week drift): next man up **36%**, other TEs **4%**, other positions **46%**, team total change -26% of vacated.
Median next-man share 22%; next man absorbs >=50% in 30% of cases.
