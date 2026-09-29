# Do the first three weeks predict the rest of the season?

```
rows: 2,578 player-seasons, 2016-2025, weeks 1-2 predicting 3-17, league scoring
naive rule = he keeps scoring his first-2-week average; mean |regression| = 3.88 ppg
early scoring alone                    n= 2319  MAE 3.383 vs naive 3.882  (+12.9%)  corr +0.561
luck alone (points above volume)       n= 2319  MAE 3.146 vs naive 3.882  (+19.0%)  corr +0.609
touchdowns per game alone              n= 2319  MAE 3.404 vs naive 3.882  (+12.3%)  corr +0.499
volume alone                           n= 2319  MAE 3.754 vs naive 3.882  (+3.3%)  corr +0.249
early scoring + luck                   n= 2319  MAE 3.009 vs naive 3.882  (+22.5%)  corr +0.674
scoring + luck + volume                n= 2319  MAE 3.010 vs naive 3.882  (+22.5%)  corr +0.683
the lot, with last season              n= 2319  MAE 2.828 vs naive 3.882  (+27.2%)  corr +0.722

RB: scoring + luck + volume + prior    n=  490  MAE 3.001 vs naive 3.821  (+21.5%)  corr +0.648
WR: scoring + luck + volume + prior    n=  896  MAE 2.625 vs naive 3.731  (+29.6%)  corr +0.749
TE: scoring + luck + volume + prior    n=  247  MAE 2.066 vs naive 3.105  (+33.4%)  corr +0.745

luck quintile (points above what volume implies, weeks 1-3):
                n  early    ros   gap
luck_q                               
most unlucky  534   7.77  10.74  2.97
unlucky       505   7.41   9.39  1.98
even          523   6.20   7.45  1.26
lucky         492  10.13   9.73 -0.40
most lucky    524  16.79  12.00 -4.79

volume quintile (targets + carries per game, weeks 1-3):
           n  early    ros   gap
vol_q                           
lowest   614   3.83   5.59  1.76
low      508   6.90   7.97  1.07
mid      480   9.28   9.88  0.61
high     474  12.69  12.04 -0.64
highest  502  17.12  14.98 -2.14
SELL corner: hot but lightly used    n= 253  early 12.90 -> rest  9.65 ppg (-3.25)
BUY corner: heavily used, cold       n= 276  early 10.19 -> rest 12.45 ppg (+2.26)
```
