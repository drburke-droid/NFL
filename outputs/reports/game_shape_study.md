# Fantasy-point shape of an NFL game — 816 games, 1632 team-games, seasons 2023-2025

Scoring is PPR (fantasy_points_ppr). Shape features only; margin/spread/total are held out.

## Are there discrete game types? No.

| k | silhouette | cluster sizes |
|---|---|---|
| 2 | 0.198 | 983, 649 |
| 3 | 0.159 | 643, 540, 449 |
| 4 | 0.164 | 603, 447, 313, 269 |
| 5 | 0.157 | 494, 322, 316, 254, 246 |
| 6 | 0.149 | 389, 301, 263, 254, 249, 176 |
| 7 | 0.147 | 303, 265, 262, 244, 219, 182, 157 |
| 8 | 0.144 | 258, 254, 245, 221, 202, 175, 154, 123 |

A silhouette this low means one continuous cloud, not separable groups. Use the axes below.

## The axes that actually exist

Variance explained: 31%, 21%, 19%, 12% (first four: 83%)

| feature | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| tot | 0.48 | -0.33 | -0.17 | -0.21 |
| qb_sh | 0.37 | -0.24 | -0.18 | 0.65 |
| rb_sh | -0.47 | -0.3 | -0.22 | -0.43 |
| wr_sh | 0.26 | 0.63 | -0.26 | -0.08 |
| te_sh | -0.03 | -0.28 | 0.71 | 0.18 |
| top1 | -0.27 | 0.0 | -0.39 | 0.39 |
| n10 | 0.47 | -0.33 | -0.12 | -0.37 |
| pass_share | 0.23 | 0.4 | 0.39 | -0.16 |

Correlation with game context (held out of the fit):

| axis | margin | game_total | pts_for | spread | total_line | att | car |
|---|---|---|---|---|---|---|---|
| PC1 | 0.198 | 0.46 | 0.459 | 0.163 | 0.235 | 0.491 | -0.149 |
| PC2 | -0.497 | -0.27 | -0.548 | -0.244 | -0.075 | 0.24 | -0.58 |
| PC3 | -0.378 | -0.177 | -0.397 | -0.167 | -0.087 | 0.344 | -0.421 |
| PC4 | -0.066 | -0.107 | -0.122 | -0.095 | -0.052 | -0.282 | 0.006 |

## Game-script asymmetry: winner vs loser by final margin

| margin | side | n | team tot | QB | RB | WR | TE | carries | att | QB2 | RB1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-3 | W | 205 | 85.2 | 17.3 | 22.3 | 33.9 | 11.7 | 28.5 | 33.2 | 0.17 | 16.7 |
| 0-3 | L | 205 | 83.1 | 16.9 | 20.6 | 32.6 | 12.9 | 25.5 | 34.1 | 0.13 | 14.8 |
| 4-10 | W | 282 | 86.9 | 17.8 | 22.8 | 33.7 | 12.6 | 29.9 | 30.7 | 0.14 | 16.4 |
| 4-10 | L | 282 | 79.9 | 15.3 | 18.8 | 32.9 | 12.8 | 23.7 | 36.1 | 0.29 | 13.6 |
| 11-17 | W | 140 | 93.4 | 19.0 | 26.1 | 35.6 | 12.7 | 31.5 | 29.7 | 0.10 | 19.3 |
| 11-17 | L | 140 | 71.8 | 12.5 | 18.1 | 28.3 | 12.9 | 21.4 | 35.6 | 0.34 | 12.9 |
| 18+ | W | 188 | 102.3 | 21.6 | 30.2 | 36.8 | 13.6 | 32.9 | 28.7 | 0.30 | 20.3 |
| 18+ | L | 188 | 61.4 | 9.7 | 15.7 | 25.6 | 10.4 | 21.0 | 33.5 | 0.39 | 10.6 |

The losing side throws MORE and its receivers score LESS -- extra attempts arrive with sacks, incompletions and picks attached. In share terms the loser's WRs do gain (blowouts: wr_sh 0.417 vs winner 0.356), but within a much smaller pie.

## Scoring environment: by combined game total

| total | n | team tot | QB | RB | WR | TE | n>=10 | n>=20 | top1_sh | hhi |
|---|---|---|---|---|---|---|---|---|---|---|
| <=34 | 352 | 63.9 | 10.7 | 18.1 | 24.8 | 10.3 | 2.35 | 0.39 | 0.275 | 0.180 |
| 35-44 | 480 | 77.2 | 14.6 | 20.6 | 30.4 | 11.6 | 3.11 | 0.70 | 0.273 | 0.178 |
| 45-54 | 432 | 88.3 | 18.1 | 21.8 | 34.8 | 13.5 | 3.60 | 1.17 | 0.271 | 0.178 |
| 55+ | 368 | 103.3 | 22.1 | 26.4 | 40.4 | 14.5 | 4.18 | 1.65 | 0.270 | 0.178 |

Concentration is invariant: top1_sh and hhi barely move across a pie that grows ~60%. A shootout scales the distribution rather than reshaping it -- what rises is breadth at a fixed threshold (n>=10, n>=20). QB share climbs and RB share falls; WR share is flat.

## Level is the game's; composition is the team's

| target | team-season identity | game context | both |
|---|---|---|---|
| tot | 0.177 | 0.635 | 0.675 |
| qb_sh | 0.188 | 0.162 | 0.310 |
| rb_sh | 0.140 | 0.028 | 0.172 |
| wr_sh | 0.241 | 0.023 | 0.267 |
| te_sh | 0.243 | 0.026 | 0.256 |
| top1 | 0.102 | 0.001 | 0.103 |
| pass_share | 0.171 | 0.241 | 0.377 |

Identity R2 is in-sample with 96 dummies on 1632 rows, so inflated by roughly 0.059; the ratio against game context still holds.

## How much is knowable before kickoff

| target | pregame (spread, total_line) | realized (margin, game total) |
|---|---|---|
| tot | 0.172 | 0.635 |
| qb | 0.113 | 0.492 |
| rb | 0.073 | 0.288 |
| wr | 0.075 | 0.241 |
| te | 0.013 | 0.044 |
| rb_sh | 0.010 | 0.028 |
| wr_sh | 0.006 | 0.023 |
| qb_sh | 0.031 | 0.162 |
| PC1 | 0.082 | 0.251 |
| PC2 | 0.065 | 0.320 |

- market total_line vs game_total: r=0.305 (R2=0.093)
- market spread vs margin: r=0.494 (R2=0.244)

The market is the ceiling: a shape cannot be forecast better than the game itself, and composition is near-unforecastable from anything -- ~1% pregame, ~3% with hindsight.

## The one monotone gradient: the favourite's running back

| pregame spread | n | RB pts | RB1 | carries | rb_sh | team tot | win% |
|---|---|---|---|---|---|---|---|
| dog 8+ | 167 | 17.8 | 12.5 | 23.8 | 0.266 | 70.0 | 21.0 |
| dog 4-7 | 327 | 20.2 | 14.6 | 25.1 | 0.262 | 79.6 | 29.4 |
| dog 1-3 | 322 | 20.1 | 14.6 | 26.6 | 0.264 | 79.1 | 39.8 |
| fav 1-3 | 413 | 22.5 | 16.1 | 27.3 | 0.273 | 85.4 | 62.7 |
| fav 4-7 | 279 | 24.2 | 17.2 | 28.7 | 0.272 | 90.1 | 68.8 |
| fav 8+ | 124 | 26.6 | 18.2 | 30.2 | 0.278 | 97.4 | 84.7 |

Monotone in every step, but note rb_sh barely moves: this is a level effect, not a mix effect. Check whether the model already prices it before treating it as an edge.

## Cross-tab against the quarter-trajectory labels (`game_scripts`)

`db/nfl_odds.db` not present — skipped. Run on the PC that has the DB.

