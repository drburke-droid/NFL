# Spec follow-ups: ridge calibrator + mechanistic component test (2026-09-10)

## A. Low-capacity ridge calibrator on the FFA residual (walk-forward, 2025)

FFA baseline (RB/WR/TE 2025, n=5,208): MAE 4.017 · RMSE 5.677 · Spearman 0.683

| features | alpha | MAE | RMSE | Spearman | mean |corr| |
|---|---|---|---|---|---|
| intercept only (bias) | 1 | 4.082 (+1.61%) | 5.676 (-0.01%) | 0.683 (+0.000) | 0.31 |
| route_surprise | 1 | 4.085 (+1.69%) | 5.677 (-0.00%) | 0.682 (-0.001) | 0.34 |
| route_surprise | 30 | 4.084 (+1.66%) | 5.676 (-0.01%) | 0.682 (-0.001) | 0.32 |
| yprr_r6 | 1 | 4.088 (+1.77%) | 5.672 (-0.10%) | 0.684 (+0.000) | 0.37 |
| yprr_r6 | 30 | 4.088 (+1.75%) | 5.671 (-0.10%) | 0.684 (+0.000) | 0.36 |
| surprise + yprr | 1 | 4.093 (+1.89%) | 5.672 (-0.08%) | 0.683 (-0.000) | 0.41 |
| surprise + yprr | 30 | 4.091 (+1.83%) | 5.671 (-0.10%) | 0.683 (+0.000) | 0.38 |
| all five | 1 | 4.094 (+1.90%) | 5.672 (-0.09%) | 0.683 (-0.000) | 0.43 |
| all five | 30 | 4.091 (+1.82%) | 5.671 (-0.11%) | 0.683 (+0.000) | 0.41 |

## B. Mechanistic receiving component vs FFA line vs DK closing line (2025 RB/WR/TE)

props frame columns: ['event_id', 'player_name', 'season', 'week', 'baseline_proj', 'over_price', 'under_price', 'n_books', 'nname', 'player_id', 'player', 'position', 'team', 'opponent_team']

| projection | MAE rec_yds | RMSE | Spearman (within week) | bias | n |
|---|---|---|---|---|---|
| FFA rec_yds | 16.07 | 23.03 | 0.639 | -0.00 | 4,163 |
| mechanistic r6 (db×rs×yprr6) | 16.94 | 24.23 | 0.599 | -0.27 | 4,142 |
| mechanistic r16 (db×rs×yprr16) | 16.91 | 24.18 | 0.609 | +0.05 | 4,011 |
| 0.5 FFA + 0.5 mechanistic r16 | 16.41 | 23.45 | 0.637 | +0.06 | 4,011 |

| projection | MAE rec | RMSE | Spearman | bias | n |
|---|---|---|---|---|---|
| FFA rec | 1.220 | 1.644 | 0.660 | -0.017 | 4,162 |
| mechanistic r16 (db×rs×cprr16) | 1.289 | 1.741 | 0.625 | +0.004 | 4,010 |

corr(mech − FFA, actual − FFA) = +0.011  (positive = mechanism knows something FFA missed)
mean (actual − FFA rec_yds) by quintile of (mech − FFA): Q1 -0.8, Q2 -1.2, Q3 +1.0, Q4 -0.2, Q5 +0.8
