# SaberSim send accuracy — 2026 (graded 2026-09-10T21:59Z)

latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 29 | 4.056 | 5.301 | -1.857 | 0.585 | 0.862 | — | — |

**Overall:** n 29 · MAE 4.056 · RMSE 5.301 · bias -1.857 · Spearman 0.585 · 80% coverage 0.862

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 2 | 1.64 | 1.66 | -1.64 | 1.0 |
| QB | 4 | 8.255 | 9.75 | -2.335 | 0.5 |
| RB | 7 | 2.11 | 2.518 | -2.11 | 1.0 |
| TE | 5 | 3.132 | 3.239 | -1.344 | 1.0 |
| WR | 11 | 4.626 | 5.496 | -1.795 | 0.818 |

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
- wk1 Drew Lock (QB SEA): proj 0.9, actual 12.8
- wk1 A.J. Brown (WR NE): proj 14.8, actual 5.6
- wk1 Romeo Doubs (WR NE): proj 8.5, actual 0.0
- wk1 Jaxon Smith-Njigba (WR SEA): proj 17.8, actual 26.2
- wk1 Drake Maye (QB NE): proj 16.5, actual 9.8
- wk1 Rashid Shaheed (WR SEA): proj 7.8, actual 1.4
- wk1 Mack Hollins (WR NE): proj 3.6, actual 9.1
- wk1 Eli Raridon (TE NE): proj 2.7, actual 7.2
- wk1 Tory Horton (WR SEA): proj 4.2, actual 0.0
- wk1 Emanuel Wilson (RB SEA): proj 4.1, actual 0.3
- wk1 Jadarian Price (RB SEA): proj 11.6, actual 7.8
