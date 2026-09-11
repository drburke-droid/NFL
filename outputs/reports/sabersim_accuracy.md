# SaberSim send accuracy — 2026 (graded 2026-09-11T14:26Z)

latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 62 | 4.097 | 5.186 | -1.769 | 0.643 | 0.823 | — | — |

**Overall:** n 62 · MAE 4.097 · RMSE 5.186 · bias -1.769 · Spearman 0.643 · 80% coverage 0.823

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 4 | 3.44 | 4.092 | -1.725 | 0.75 |
| QB | 8 | 6.639 | 8.526 | -2.576 | 0.625 |
| RB | 14 | 2.567 | 2.856 | -1.354 | 0.857 |
| TE | 13 | 3.189 | 3.588 | -2.2 | 0.923 |
| WR | 23 | 4.771 | 5.663 | -1.506 | 0.826 |

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

Largest misses:

- wk1 Sam Darnold (QB SEA): proj 14.5, actual 0.5
- wk1 Matthew Stafford (QB LA): proj 17.5, actual 4.1
- wk1 Drew Lock (QB SEA): proj 0.9, actual 12.8
- wk1 A.J. Brown (WR NE): proj 14.8, actual 5.6
- wk1 Puka Nacua (WR LA): proj 21.4, actual 12.4
- wk1 Davante Adams (WR LA): proj 14.2, actual 5.6
- wk1 Romeo Doubs (WR NE): proj 8.5, actual 0.0
- wk1 Jaxon Smith-Njigba (WR SEA): proj 17.8, actual 26.2
- wk1 Demarcus Robinson (WR SF): proj 4.7, actual 13.0
- wk1 Deebo Samuel (WR SF): proj 10.1, actual 18.0
- wk1 De'Zhaun Stribling (WR SF): proj 7.3, actual 0.0
- wk1 George Kittle (TE SF): proj 10.4, actual 3.2
