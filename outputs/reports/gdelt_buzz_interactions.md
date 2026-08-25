# GDELT x wiki-buzz interaction study

Universe: late-breakout frame 2016-2025 (6783 player-seasons). Signals measured in August of the season, pre-draft. Outcomes: `hit` (late-breakout label), `overperf` = actual pts - FFA projection.


## H1 — RB1 bad news x RB2 wiki buzz

### neg-tone Q1 (tone <= 25th pctl of P1s, >=3 articles) — pairs n=288
```
               n    hit    over   beat
bad1  buzz2                           
False False  119  0.202  10.891  0.542
      True   108  0.157  13.284  0.549
True  False   35  0.114   7.046  0.517
      True    26  0.308  27.528  0.640
interaction delta (hit): +0.238, bootstrap p=0.038
```

### high avg_neg (neg >= 75th pctl of P1s, >=3 articles) — pairs n=288
```
               n    hit    over   beat
bad1  buzz2                           
False False  117  0.214  14.030  0.570
      True   110  0.155  15.068  0.570
True  False   37  0.081  -1.715  0.438
      True    24  0.333  21.552  0.565
interaction delta (hit): +0.311, bootstrap p=0.008
```

### news storm (articles >= 75th pctl AND tone below median) — pairs n=288
```
               n    hit    over   beat
bad1  buzz2                           
False False  135  0.185   9.718  0.532
      True   114  0.167  11.218  0.561
True  False   19  0.158  12.229  0.571
      True    20  0.300  44.317  0.611
interaction delta (hit): +0.161, bootstrap p=0.241
```

## H2 — WR1 bad news x WR2 wiki buzz

### neg-tone Q1 (tone <= 25th pctl of P1s, >=3 articles) — pairs n=279
```
               n    hit    over   beat
bad1  buzz2                           
False False  170  0.371  21.347  0.605
      True    55  0.236  28.404  0.723
True  False   43  0.233  15.225  0.647
      True    11  0.182  13.982  0.556
interaction delta (hit): +0.083, bootstrap p=0.614
```

### high avg_neg (neg >= 75th pctl of P1s, >=3 articles) — pairs n=279
```
               n    hit    over   beat
bad1  buzz2                           
False False  171  0.380  21.705  0.607
      True    54  0.222  24.904  0.711
True  False   42  0.190  12.901  0.645
      True    12  0.250  30.922  0.636
interaction delta (hit): +0.217, bootstrap p=0.172
```

### news storm (articles >= 75th pctl AND tone below median) — pairs n=279
```
               n    hit    over   beat
bad1  buzz2                           
False False  184  0.348  21.900  0.613
      True    61  0.246  26.145  0.673
True  False   29  0.310  10.048  0.615
      True     5  0.000  25.325  1.000
interaction delta (hit): -0.208, bootstrap p=0.271
```

## H3 — team-level August buzz
```
team overperf by Aug team-buzz quintile:
          n    over  beat
spike_q                  
0        70   90.12  0.58
1        60  101.50  0.61
2        70  116.42  0.60
3        60  117.82  0.59
4        60  126.70  0.62
corr(team median spike, team sum overperf) = +0.003
```
```
cheap-player hit rate by team-buzz quartile:
        n    hit
tmq             
0    1305  0.019
1    1295  0.027
2    1297  0.019
3    1291  0.025
cheap + own buzz >=75pctl by team-buzz quartile:
       n    hit
tmq            
0    285  0.018
1    355  0.023
2    425  0.024
3    540  0.022
```
