# SaberSim send accuracy — 2026 (graded 2026-09-13T21:28Z)

latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 10 | 296 | 3.863 | 5.47 | -0.268 | 0.779 | 0.824 | 4.201 (model 4.176) | — |

**Overall:** n 296 · MAE 3.863 · RMSE 5.47 · bias -0.268 · Spearman 0.779 · 80% coverage 0.824

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 20 | 3.331 | 4.239 | +0.325 | 0.85 |
| QB | 39 | 4.507 | 6.822 | +0.382 | 0.846 |
| RB | 69 | 4.021 | 5.796 | +0.251 | 0.826 |
| TE | 65 | 3.081 | 4.164 | -0.891 | 0.831 |
| WR | 103 | 4.109 | 5.605 | -0.583 | 0.806 |

## Scale (2025 reference, full slate pool)

Bands are for the full slate pool (every projected skill player, ~10 per team). MAE is dominated by outcome noise: a hindsight oracle that knows each player's true season average still scores MAE 4.04 on this pool, so ~4.0 is the floor and 3.0 is not attainable. Starters-only pools run 1.5-2 points higher (FFA 5.95 on players projected 8+). A single 30-player slate has an MAE standard deviation of ~0.8, so judge on 5+ weeks (~1,500 player-games) and on the same-rows comparison with FFA and DraftKings.

| projection | MAE | RMSE | Spearman |
|---|---|---|---|
| previous game points | 5.73 | 8.25 | 0.532 |
| trailing 4-game mean | 4.78 | 6.65 | 0.625 |
| season-to-date mean | 4.67 | 6.58 | 0.641 |
| DK market-implied (rows with a line) | 4.74 | 6.4 | 0.608 |
| FFA consensus | 4.17 | 5.85 | 0.719 |
| Model_Burke (walk-forward) | 4.12 | 5.9 | 0.714 |
| oracle: true season mean, hindsight | 4.04 | 5.65 | 0.737 |

Bands (upper edge): MAE elite ≤4.05 · top ≤4.15 · consensus ≤4.30 · fair ≤4.80 · poor above. RMSE 5.65/5.85/6.05/6.70. Spearman ≥0.74/0.72/0.69/0.60. |bias| ≤0.15/0.30/0.50/0.80. 80% coverage within ±0.02/0.04/0.06/0.10 of 0.80. Model÷FFA MAE on the same rows ≤0.97/0.99/1.01/1.04.

## Subvertadown check

- On 136 RB/WR/TE player-games where Subvertadown flagged a matchup of at least ±0.5 team points, the direction of our error matched the flag 48% of the time (50% = coin flip).
- Adding the full team bonus, shared by projection, would have moved MAE from 3.80 to 3.79; half of it: 3.79.
- Correlation between the bonus and our error: +0.05.
- Directional only — a signal needs several hundred player-games before ±0.1 MAE means anything; prior studies found opponent-matchup features add nothing on top of FFA + DK, so the bar is 'consistently right direction', not one good week.
- QB: on 20 graded starters Subvertadown's projection MAE was 7.84 vs ours 6.96; a 50/50 blend 7.39.

Largest misses:

- wk1 Jalen Coker (WR CAR): proj 9.5, actual 33.8
- wk1 Caleb Williams (QB CHI): proj 17.2, actual 37.3
- wk1 Derrick Henry (RB BAL): proj 15.9, actual 35.3
- wk1 D'Andre Swift (RB CHI): proj 13.8, actual 32.4
- wk1 Ja'Marr Chase (WR CIN): proj 21.1, actual 3.2
- wk1 Josh Allen (QB BUF): proj 19.3, actual 35.7
- wk1 Bryce Young (QB CAR): proj 15.2, actual 31.4
- wk1 David Montgomery (RB HOU): proj 12.9, actual 28.9
- wk1 Sam Darnold (QB SEA): proj 14.5, actual 0.5
- wk1 Chuba Hubbard (RB CAR): proj 10.7, actual 23.7
- wk1 Colston Loveland (TE CHI): proj 12.9, actual 0.0
- wk1 Kyle Monangai (RB CHI): proj 7.8, actual 20.4
