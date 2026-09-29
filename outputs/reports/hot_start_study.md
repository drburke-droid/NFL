# Do the first three weeks predict the rest of the season?

```
rows: 2,889 player-seasons, 2016-2025, weeks 1-3 predicting 4-17, league scoring
naive rule = he keeps scoring his first-3-week average; mean |regression| = 3.36 ppg
early scoring alone                    n= 2601  MAE 3.082 vs naive 3.387  (+9.0%)  corr +0.476
luck alone (points above volume)       n= 2601  MAE 2.901 vs naive 3.387  (+14.4%)  corr +0.522
touchdowns per game alone              n= 2601  MAE 3.110 vs naive 3.387  (+8.2%)  corr +0.413
volume alone                           n= 2601  MAE 3.297 vs naive 3.387  (+2.7%)  corr +0.243
early scoring + luck                   n= 2601  MAE 2.812 vs naive 3.387  (+17.0%)  corr +0.588
scoring + luck + volume                n= 2601  MAE 2.809 vs naive 3.387  (+17.1%)  corr +0.599
the lot, with last season              n= 2601  MAE 2.704 vs naive 3.387  (+20.2%)  corr +0.636

RB: scoring + luck + volume + prior    n=  542  MAE 2.887 vs naive 3.661  (+21.1%)  corr +0.620
WR: scoring + luck + volume + prior    n=  982  MAE 2.522 vs naive 3.204  (+21.3%)  corr +0.654
TE: scoring + luck + volume + prior    n=  359  MAE 1.927 vs naive 2.613  (+26.3%)  corr +0.665

luck quintile (points above what volume implies, weeks 1-3):
                n  early    ros   gap
luck_q                               
most unlucky  593   8.03  10.21  2.19
unlucky       577   6.64   8.12  1.48
even          570   6.28   7.50  1.23
lucky         562   9.39   9.06 -0.33
most lucky    587  15.23  11.63 -3.60

volume quintile (targets + carries per game, weeks 1-3):
           n  early    ros   gap
vol_q                           
lowest   643   3.28   4.92  1.63
low      551   5.88   6.93  1.05
mid      575   8.84   9.09  0.25
high     571  11.96  11.47 -0.49
highest  549  16.62  14.92 -1.71
SELL corner: hot but lightly used    n= 285  early 11.69 -> rest  8.90 ppg (-2.80)
BUY corner: heavily used, cold       n= 322  early 10.67 -> rest 12.33 ppg (+1.66)
```
