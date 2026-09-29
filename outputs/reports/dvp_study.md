# Does the opponent matter, and can you see it after 3 weeks?

```
persistence of defence-vs-position, first 3 weeks -> rest of season (n=1280 defence-seasons-positions)
  QB   n= 320   raw r=+0.147   opponent-adjusted r=+0.171
  RB   n= 320   raw r=+0.169   opponent-adjusted r=+0.212
  WR   n= 320   raw r=+0.097   opponent-adjusted r=+0.134
  TE   n= 320   raw r=+0.200   opponent-adjusted r=+0.219
  ALL  n=1280   raw r=+0.128   opponent-adjusted r=+0.159

spread of true rest-of-season defence-vs-position (sd of points allowed vs average):
  QB    2.86 pts/game
  RB    3.40 pts/game
  TE    2.56 pts/game
  WR    4.33 pts/game

predicting a player's ACTUAL weekly points, weeks 4-17 (n=33,038 player-weeks)
  player rate alone      MAE 5.6095
  raw                    MAE 5.6098   vs baseline -0.006%   avg weight on the matchup term +0.021
  opponent-adjusted      MAE 5.6097   vs baseline -0.004%   avg weight on the matchup term +0.023
```
