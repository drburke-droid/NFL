# SaberSim send accuracy — 2026 (graded 2026-09-20T21:05Z)

latest send generated >= 75 min before kickoff, per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 16 | 470 | 3.876 | 5.476 | -0.379 | 0.776 | 0.826 | 4.171 (model 4.139) | 4.466 (model 4.625) |
| 2 | 2 | 8 | 235 | 3.327 | 4.707 | -0.865 | 0.779 | 0.817 | 3.681 (model 3.525) | 3.829 (model 3.966) |

**Overall:** n 705 · MAE 3.693 · RMSE 5.232 · bias -0.541 · Spearman 0.776 · 80% coverage 0.823

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 48 | 3.129 | 3.961 | -0.365 | 0.854 |
| QB | 100 | 4.017 | 6.033 | -0.297 | 0.85 |
| RB | 162 | 3.59 | 5.232 | -0.485 | 0.809 |
| TE | 152 | 3.274 | 4.59 | -0.699 | 0.809 |
| WR | 243 | 4.002 | 5.47 | -0.615 | 0.823 |

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

- On 352 RB/WR/TE player-games where Subvertadown flagged a matchup of at least ±0.5 team points, the direction of our error matched the flag 51% of the time (50% = coin flip).
- Adding the full team bonus, shared by projection, would have moved MAE from 3.68 to 3.67; half of it: 3.67.
- Correlation between the bonus and our error: +0.06.
- Directional only — a signal needs several hundred player-games before ±0.1 MAE means anything; prior studies found opponent-matchup features add nothing on top of FFA + DK, so the bar is 'consistently right direction', not one good week.
- QB: on 49 graded starters Subvertadown's projection MAE was 7.48 vs ours 6.38; a 50/50 blend 6.91.

Largest misses:

- wk1 Jalen Coker (WR CAR): proj 9.5, actual 33.8
- wk1 Christian Watson (WR GB): proj 11.6, actual 32.7
- wk1 Caleb Williams (QB CHI): proj 17.2, actual 37.3
- wk1 Derrick Henry (RB BAL): proj 15.9, actual 35.3
- wk1 Kenneth Walker III (RB KC): proj 15.3, actual 34.1
- wk1 Isaiah Likely (TE NYG): proj 9.1, actual 27.8
- wk1 D'Andre Swift (RB CHI): proj 13.8, actual 32.4
- wk1 Ja'Marr Chase (WR CIN): proj 21.1, actual 3.2
- wk2 Josh Allen (QB BUF): proj 23.9, actual 40.8
- wk1 Ashton Jeanty (RB LV): proj 15.8, actual 32.7
- wk1 Kyler Murray (QB MIN): proj 16.5, actual -0.4
- wk1 Josh Allen (QB BUF): proj 19.3, actual 35.7
