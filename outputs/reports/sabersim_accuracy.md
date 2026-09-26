# SaberSim send accuracy — 2026 (graded 2026-09-26T14:00Z)

latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 16 | 470 | 3.876 | 5.476 | -0.379 | 0.776 | 0.826 | 4.171 (model 4.139) | 4.466 (model 4.625) |
| 2 | 6 | 16 | 464 | 3.6 | 5.11 | -0.611 | 0.771 | 0.8 | 3.929 (model 3.764) | 4.081 (model 4.208) |
| 3 | 1 | 1 | 24 | 4.257 | 6.02 | +1.982 | 0.766 | 0.75 | 4.571 (model 4.193) | 4.362 (model 4.393) |

**Overall:** n 958 · MAE 3.752 · RMSE 5.317 · bias -0.432 · Spearman 0.772 · 80% coverage 0.811

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 66 | 3.373 | 4.257 | -0.18 | 0.788 |
| QB | 132 | 4.224 | 6.153 | -0.186 | 0.833 |
| RB | 221 | 3.536 | 5.076 | -0.545 | 0.824 |
| TE | 204 | 3.306 | 4.581 | -0.575 | 0.814 |
| WR | 335 | 4.055 | 5.707 | -0.418 | 0.797 |

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

- On 490 RB/WR/TE player-games where Subvertadown flagged a matchup of at least ±0.5 team points, the direction of our error matched the flag 53% of the time (50% = coin flip).
- Adding the full team bonus, shared by projection, would have moved MAE from 3.70 to 3.67; half of it: 3.68.
- Correlation between the bonus and our error: +0.07.
- Directional only — a signal needs several hundred player-games before ±0.1 MAE means anything; prior studies found opponent-matchup features add nothing on top of FFA + DK, so the bar is 'consistently right direction', not one good week.
- QB: on 68 graded starters Subvertadown's projection MAE was 7.41 vs ours 6.54; a 50/50 blend 6.96.

Largest misses:

- wk1 Jalen Coker (WR CAR): proj 9.5, actual 33.8
- wk2 Jaxon Smith-Njigba (WR SEA): proj 18.8, actual 42.5
- wk2 Davante Adams (WR LA): proj 17.4, actual 39.5
- wk1 Christian Watson (WR GB): proj 11.6, actual 32.7
- wk1 Caleb Williams (QB CHI): proj 17.2, actual 37.3
- wk1 Derrick Henry (RB BAL): proj 15.9, actual 35.3
- wk1 Kenneth Walker III (RB KC): proj 15.3, actual 34.1
- wk1 Isaiah Likely (TE NYG): proj 9.1, actual 27.8
- wk1 D'Andre Swift (RB CHI): proj 13.8, actual 32.4
- wk1 Ja'Marr Chase (WR CIN): proj 21.1, actual 3.2
- wk2 CeeDee Lamb (WR DAL): proj 18.3, actual 35.3
- wk2 Josh Allen (QB BUF): proj 23.9, actual 40.8
