# SaberSim send accuracy — 2026 (graded 2026-11-12T14:06Z)

latest send generated >= 50 min before kickoff (>= 55 for kickoffs before 2026-10-25 13:30Z, when the T-55 fallback began; >= 75 before 2026-09-27 20:10Z, when the send moved from T-80 to T-60), per game and player; QB/RB/WR/TE scored PPR (4-pt pass TD, -2 INT), K = DK kicker scoring; DST not graded

| week | sends | games | n | MAE | RMSE | bias | Spearman | 80% cov | FFA MAE (same rows) | DK MAE (same rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 16 | 470 | 3.876 | 5.476 | -0.379 | 0.776 | 0.826 | 4.171 (model 4.139) | 4.466 (model 4.625) |
| 2 | 6 | 16 | 464 | 3.6 | 5.11 | -0.611 | 0.771 | 0.8 | 3.929 (model 3.764) | 4.081 (model 4.208) |
| 3 | 6 | 16 | 421 | 3.733 | 5.387 | +0.9 | 0.776 | 0.781 | 4.294 (model 3.791) | 4.271 (model 4.162) |
| 4 | 7 | 16 | 429 | 3.791 | 5.858 | +0.899 | 0.785 | 0.797 | 4.436 (model 3.904) | 4.493 (model 4.473) |
| 5 | 7 | 15 | 407 | 3.663 | 5.537 | +0.813 | 0.795 | 0.803 | 4.29 (model 3.757) | 4.362 (model 4.31) |
| 6 | 6 | 13 | 360 | 3.469 | 5.1 | +0.501 | 0.792 | 0.808 | 4.128 (model 3.459) | 4.063 (model 3.995) |
| 7 | 6 | 13 | 348 | 3.413 | 5.087 | +0.564 | 0.802 | 0.807 | 4.13 (model 3.544) | 4.157 (model 4.068) |
| 8 | 6 | 14 | 372 | 3.755 | 5.769 | +0.726 | 0.745 | 0.804 | 4.317 (model 3.819) | 4.36 (model 4.341) |
| 9 | 4 | 6 | 153 | 3.632 | 5.923 | +0.614 | 0.75 | 0.824 | 3.862 (model 3.589) | 4.321 (model 4.243) |

**Overall:** n 3424 · MAE 3.671 · RMSE 5.451 · bias +0.402 · Spearman 0.77 · 80% coverage 0.804

| pos | n | MAE | RMSE | bias | 80% cov |
|---|---|---|---|---|---|
| K | 250 | 3.654 | 4.526 | +0.306 | 0.78 |
| QB | 390 | 4.357 | 6.319 | -0.014 | 0.859 |
| RB | 820 | 3.511 | 5.435 | +0.539 | 0.774 |
| TE | 813 | 3.051 | 4.554 | +0.158 | 0.835 |
| WR | 1151 | 3.995 | 5.893 | +0.638 | 0.791 |

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

- On 2092 RB/WR/TE player-games where Subvertadown flagged a matchup of at least ±0.5 team points, the direction of our error matched the flag 51% of the time (50% = coin flip).
- Adding the full team bonus, shared by projection, would have moved MAE from 3.58 to 3.58; half of it: 3.58.
- Correlation between the bonus and our error: -0.01.
- Good minus bad matchups (how much more the flagged-good players beat our projection than the flagged-bad ones, within each week, then averaged over weeks; 0 = no signal): RB +0.08, WR -0.06, TE +0.79; all +0.19.
- Directional only — a signal needs several hundred player-games before ±0.1 MAE means anything; prior studies found opponent-matchup features add nothing on top of FFA + DK, so the bar is 'consistently right direction', not one good week.
- QB direction: where the two projections differed by 1+ point (both had him starting), the result landed on Subvertadown's side 48 of 109 times (44%; 50% = no better than ours).
- QB: on 275 graded starters Subvertadown's projection MAE was 7.40 vs ours 5.51; a 50/50 blend 6.43.

Largest misses:

- wk8 Tyler Allgeier (RB ARI): proj 4.4, actual 36.1
- wk4 Tetairoa McMillan (WR CAR): proj 16.1, actual 45.2
- wk9 Kayshon Boutte (WR HOU): proj 2.7, actual 28.5
- wk4 CeeDee Lamb (WR DAL): proj 16.1, actual 41.3
- wk4 Kyren Williams (RB LA): proj 12.1, actual 36.7
- wk1 Jalen Coker (WR CAR): proj 9.5, actual 33.8
- wk4 Tyquan Thornton (WR KC): proj 3.8, actual 28.1
- wk7 Tony Pollard (RB TEN): proj 12.4, actual 36.5
- wk2 Jaxon Smith-Njigba (WR SEA): proj 18.8, actual 42.5
- wk8 Emmett Johnson (RB KC): proj 2.0, actual 25.3
- wk5 Isaiah Likely (TE NYG): proj 9.8, actual 32.4
- wk8 Adonai Mitchell (WR NYJ): proj 4.9, actual 27.2
