# How much do actual points move with the defence faced?

```
38,636 player-games, 2016-2025, league scoring. No projections used: the baseline is the player's own year-to-date points per game from the weeks BEFORE each game.
'gap' = his actual points that week minus his own year-to-date rate.

KNOWABLE (defence ranked on what it had allowed before this game):
          games  ytd_rate  actual   gap
bucket                                 
toughest   7773      9.12    8.68 -0.44
hard       7725      8.91    8.90 -0.01
average    7720      8.76    8.89  0.13
soft       7711      8.68    9.04  0.36
softest    7707      8.89    9.36  0.47
  softest minus toughest: +0.91 pts/game

  by position: QB +2.29   RB +1.26   WR +0.44   TE +0.72

ORACLE (defence ranked on its whole season: unknowable, the ceiling):
          games  ytd_rate  actual   gap
bucket                                 
toughest   8325      8.89    7.50 -1.39
hard       7705      8.89    8.43 -0.46
average    7822      8.87    9.06  0.19
soft       7482      8.79    9.47  0.68
softest    7302      8.92   10.64  1.71
  softest minus toughest: +3.10 pts/game

  by position: QB +7.01   RB +3.18   WR +2.34   TE +2.58

a manager who could always pick the softest-fifth matchup instead of an average one would gain +1.52 pts per game per player, with perfect foreknowledge
for scale: the typical week-to-week swing in a player's own scoring is about 4.94 pts, and a bye week costs the whole starter
```
