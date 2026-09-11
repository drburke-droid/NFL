# SaberSim send accuracy — 2026 (graded 2026-09-11T03:33Z)

latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 33 | 6.566 | 8.643 | -6.566 | nan | 0.515 | 6.792 (model 7.08) | — |

**Overall:** n 33 · MAE 6.566 · RMSE 8.643 · bias -6.566 · Spearman nan · 80% coverage 0.515

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 2 | 7.965 | 7.968 | -7.965 | 0.0 |
| QB | 4 | 8.853 | 11.719 | -8.853 | 0.5 |
| RB | 7 | 7.327 | 9.535 | -7.327 | 0.571 |
| TE | 8 | 4.372 | 5.002 | -4.372 | 0.5 |
| WR | 12 | 6.588 | 8.908 | -6.588 | 0.583 |

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

- wk1 Puka Nacua (WR LA): proj 21.2, actual 0.0
- wk1 Christian McCaffrey (RB SF): proj 19.4, actual 0.0
- wk1 Brock Purdy (QB SF): proj 16.6, actual 0.0
- wk1 Matthew Stafford (QB LA): proj 16.4, actual 0.0
- wk1 Davante Adams (WR LA): proj 14.3, actual 0.0
- wk1 Kyren Williams (RB LA): proj 12.9, actual 0.0
- wk1 Mike Evans (WR SF): proj 11.2, actual 0.0
- wk1 George Kittle (TE SF): proj 9.3, actual 0.0
- wk1 Deebo Samuel (WR SF): proj 8.7, actual 0.0
- wk1 Harrison Mevis (K LA): proj 8.2, actual 0.0
- wk1 Eddy Pineiro (K SF): proj 7.8, actual 0.0
- wk1 Blake Corum (RB LA): proj 7.6, actual 0.0
