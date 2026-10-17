# SaberSim send accuracy — 2026 (graded 2026-10-17T14:05Z)

latest send generated >= 55 min before kickoff (>= 75 for kickoffs before 2026-09-27 20:10Z, when the send moved from T-80 to T-60), per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 16 | 470 | 3.876 | 5.476 | -0.379 | 0.776 | 0.826 | 4.171 (model 4.139) | 4.466 (model 4.625) |
| 2 | 6 | 16 | 464 | 3.6 | 5.11 | -0.611 | 0.771 | 0.8 | 3.929 (model 3.764) | 4.081 (model 4.208) |
| 3 | 6 | 16 | 421 | 3.733 | 5.387 | +0.9 | 0.776 | 0.781 | 4.294 (model 3.791) | 4.271 (model 4.162) |
| 4 | 7 | 16 | 429 | 3.791 | 5.858 | +0.899 | 0.785 | 0.797 | 4.436 (model 3.904) | 4.493 (model 4.473) |
| 5 | 7 | 15 | 407 | 3.663 | 5.537 | +0.813 | 0.795 | 0.803 | 4.29 (model 3.757) | 4.362 (model 4.31) |
| 6 | 1 | 1 | 28 | 2.848 | 3.86 | -0.047 | 0.814 | 0.893 | 3.134 (model 3.138) | 3.37 (model 3.394) |

**Overall:** n 2219 · MAE 3.723 · RMSE 5.456 · bias +0.285 · Spearman 0.769 · 80% coverage 0.803

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 160 | 3.622 | 4.509 | +0.419 | 0.781 |
| QB | 269 | 4.294 | 6.186 | +0.102 | 0.862 |
| RB | 528 | 3.53 | 5.336 | +0.306 | 0.782 |
| TE | 510 | 3.168 | 4.659 | +0.043 | 0.824 |
| WR | 752 | 4.051 | 5.924 | +0.472 | 0.787 |

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

- On 1324 RB/WR/TE player-games where Subvertadown flagged a matchup of at least ±0.5 team points, the direction of our error matched the flag 53% of the time (50% = coin flip).
- Adding the full team bonus, shared by projection, would have moved MAE from 3.65 to 3.65; half of it: 3.64.
- Correlation between the bonus and our error: -0.01.
- Good minus bad matchups (how much more the flagged-good players beat our projection than the flagged-bad ones, within each week, then averaged over weeks; 0 = no signal): RB +0.08, WR +0.11, TE +1.23; all +0.37.
- Directional only — a signal needs several hundred player-games before ±0.1 MAE means anything; prior studies found opponent-matchup features add nothing on top of FFA + DK, so the bar is 'consistently right direction', not one good week.
- QB direction: where the two projections differed by 1+ point (both had him starting), the result landed on Subvertadown's side 32 of 64 times (50%; 50% = no better than ours).
- QB: on 175 graded starters Subvertadown's projection MAE was 7.29 vs ours 5.60; a 50/50 blend 6.43.

Largest misses:

- wk4 Tetairoa McMillan (WR CAR): proj 16.1, actual 45.2
- wk4 CeeDee Lamb (WR DAL): proj 16.1, actual 41.3
- wk4 Kyren Williams (RB LA): proj 12.1, actual 36.7
- wk1 Jalen Coker (WR CAR): proj 9.5, actual 33.8
- wk4 Tyquan Thornton (WR KC): proj 3.8, actual 28.1
- wk2 Jaxon Smith-Njigba (WR SEA): proj 18.8, actual 42.5
- wk5 Isaiah Likely (TE NYG): proj 9.8, actual 32.4
- wk2 Davante Adams (WR LA): proj 17.4, actual 39.5
- wk4 Brian Robinson (RB ATL): proj 3.7, actual 25.7
- wk5 Drake Maye (QB NE): proj 18.1, actual 39.5
- wk4 Kyle Monangai (RB CHI): proj 6.7, actual 28.0
- wk1 Christian Watson (WR GB): proj 11.6, actual 32.7
