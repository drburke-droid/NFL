# Feature candidates for the SaberSim model (2026-09-10)

History frame: 22,527 player-weeks 2023-25 (FFA weekly baseline, Model_Burke lags).

## 1. Univariate: correlation with the FFA residual (2023-25), by position

| feature | coverage | QB | RB | WR | TE | all |
|---|---|---|---|---|---|---|
| qb_ttt_x_press | 7% | +0.021 | — | — | — | +0.021 |
| qb_sack_x_press | 6% | -0.005 | — | — | — | -0.005 |
| qb_scr_x_man | 6% | — | — | — | — | — |
| qb_adot_x_twohigh | 6% | -0.018 | — | — | — | -0.018 |
| qb_adot_x_press | 6% | +0.018 | — | — | — | +0.018 |
| qb_ttt_x_twohigh | 7% | -0.007 | — | — | — | -0.007 |
| wr_deep_x_twohigh | 37% | — | — | -0.011 | -0.009 | -0.011 |
| wr_deep_x_press | 37% | — | — | +0.018 | +0.002 | +0.014 |
| wr_deep_x_qbadot | 46% | — | — | -0.022 | -0.030 | -0.024 |
| wr_sep_x_man | 24% | — | — | +0.007 | +0.018 | +0.011 |
| wr_cushion_x_man | 24% | — | — | +0.026 | +0.024 | +0.025 |
| wr_pressgap_x_press | 37% | — | — | -0.023 | +0.023 | -0.012 |
| rb_tprr_x_press | 14% | — | -0.029 | — | — | -0.029 |
| rb_x_box | 16% | — | -0.035 | — | — | -0.035 |

### Mean FFA residual by quintile of the interaction (own position only)

| feature | Q1 | Q2 | Q3 | Q4 | Q5 | Q5−Q1 | n |
|---|---|---|---|---|---|---|---|
| qb_ttt_x_press | +0.14 | +0.44 | +0.35 | +0.70 | +0.59 | +0.45 | 1,606 |
| qb_sack_x_press | +0.23 | +1.17 | +0.15 | +0.68 | -0.05 | -0.28 | 1,443 |
| qb_scr_x_man | +0.30 | +0.27 | +1.40 | -0.06 | +0.26 | -0.04 | 1,443 |
| qb_adot_x_twohigh | +0.65 | +0.12 | +0.45 | +1.00 | +0.02 | -0.63 | 1,442 |
| qb_adot_x_press | +0.09 | +0.30 | +0.38 | +1.02 | +0.43 | +0.33 | 1,442 |
| qb_ttt_x_twohigh | +0.39 | +0.28 | +0.93 | +0.57 | +0.05 | -0.34 | 1,606 |
| wr_deep_x_twohigh | +0.39 | +0.41 | +0.50 | +0.20 | +0.19 | -0.20 | 8,255 |
| wr_deep_x_press | +0.13 | +0.27 | +0.37 | +0.43 | +0.49 | +0.36 | 8,255 |
| wr_deep_x_qbadot | +0.27 | +0.33 | +0.35 | +0.37 | +0.03 | -0.24 | 10,451 |
| wr_sep_x_man | +0.14 | -0.03 | +0.24 | +0.27 | +0.36 | +0.22 | 5,358 |
| wr_cushion_x_man | +0.08 | +0.10 | +0.17 | +0.01 | +0.62 | +0.54 | 5,358 |
| wr_pressgap_x_press | +0.51 | +0.23 | +0.54 | +0.20 | +0.21 | -0.30 | 8,255 |
| rb_tprr_x_press | +0.48 | +0.59 | +0.04 | +0.46 | +0.27 | -0.21 | 3,226 |
| rb_x_box | +0.66 | +0.42 | +0.11 | +0.07 | +0.01 | -0.65 | 3,561 |

### Main effects for reference (own position, quintile Q5−Q1 of the FFA residual)

- QB off_ttt_r6: Q1 +0.34 … Q5 -0.04 (Δ -0.38, r -0.020, n=2,104)
- QB qb_adot_r6: Q1 +0.60 … Q5 -0.01 (Δ -0.62, r -0.027, n=2,050)
- QB qb_scramble_rate_r6: Q1 -0.07 … Q5 +0.30 (Δ +0.37, r +nan, n=2,051)
- QB opp_press_r6_z: Q1 +0.50 … Q5 +0.51 (Δ +0.01, r -0.023, n=1,607)
- QB opp_two_high_r6_z: Q1 +0.26 … Q5 +0.45 (Δ +0.18, r +0.019, n=1,607)
- WR deep_share_r6: Q1 +0.35 … Q5 -0.37 (Δ -0.72, r -0.034, n=8,558)
- WR sep_prev: Q1 -0.09 … Q5 -0.09 (Δ +0.00, r +0.007, n=6,226)
- WR opp_man_r6_z: Q1 +0.25 … Q5 +0.27 (Δ +0.01, r -0.007, n=5,695)
- WR opp_two_high_r6_z: Q1 +0.18 … Q5 -0.14 (Δ -0.33, r -0.007, n=5,695)
- RB opp_press_r6_z: Q1 +0.44 … Q5 -0.13 (Δ -0.57, r -0.044, n=3,561)
- RB opp_box_r6_z: Q1 +0.66 … Q5 +0.01 (Δ -0.65, r -0.035, n=3,561)

## 2. Direct walk-forward residual GBM (2025; baseline + k*pred_resid)

The shipped package shrinks its correction to ~0 on this baseline, so this is the honest sensitivity test.

FFA baseline alone: MAE 4.190 · RMSE 5.872 · Spearman 0.714

| features | k | MAE | RMSE | weekly Spearman |
|---|---|---|---|---|
| base lags+context | 0.5 | 4.207 | 5.858 | 0.715 |
| base lags+context | 1.0 | 4.233 | 5.856 | 0.715 |
| + QB x pressure | 0.5 | 4.206 (-0.01%) | 5.858 (-0.01%) | 0.715 (-0.000) |
| + QB x pressure | 1.0 | 4.231 (-0.04%) | 5.854 (-0.02%) | 0.715 (+0.000) |
| + QB x man/two-high | 0.5 | 4.205 (-0.04%) | 5.858 (-0.01%) | 0.714 (-0.000) |
| + QB x man/two-high | 1.0 | 4.229 (-0.10%) | 5.854 (-0.03%) | 0.715 (-0.000) |
| + WR deep x D | 0.5 | 4.208 (+0.04%) | 5.858 (+0.00%) | 0.714 (-0.000) |
| + WR deep x D | 1.0 | 4.236 (+0.07%) | 5.856 (+0.00%) | 0.715 (-0.000) |
| + WR deep x QB aDOT | 0.5 | 4.207 (-0.00%) | 5.858 (-0.01%) | 0.715 (+0.000) |
| + WR deep x QB aDOT | 1.0 | 4.233 (-0.01%) | 5.855 (-0.02%) | 0.715 (+0.000) |
| + WR separation x man | 0.5 | 4.208 (+0.03%) | 5.860 (+0.02%) | 0.715 (-0.000) |
| + WR separation x man | 1.0 | 4.236 (+0.06%) | 5.858 (+0.04%) | 0.715 (-0.000) |
| + WR pressure gap | 0.5 | 4.209 (+0.05%) | 5.859 (+0.01%) | 0.715 (-0.000) |
| + WR pressure gap | 1.0 | 4.237 (+0.09%) | 5.857 (+0.02%) | 0.715 (-0.000) |
| + RB x pressure/box | 0.5 | 4.207 (+0.02%) | 5.859 (+0.01%) | 0.715 (-0.000) |
| + RB x pressure/box | 1.0 | 4.234 (+0.02%) | 5.856 (+0.01%) | 0.715 (+0.000) |
| + ALL interactions | 0.5 | 4.209 (+0.05%) | 5.860 (+0.02%) | 0.714 (-0.000) |
| + ALL interactions | 1.0 | 4.237 (+0.09%) | 5.858 (+0.04%) | 0.715 (-0.000) |
| + all that helped (QB x pressure, WR deep x QB aDOT) | 0.5 | 4.207 (+0.02%) | 5.859 (+0.01%) | 0.715 (-0.000) |
| + all that helped (QB x pressure, WR deep x QB aDOT) | 1.0 | 4.233 (+0.01%) | 5.856 (+0.01%) | 0.715 (-0.000) |

(n = 5,887 2025 player-weeks)

## 3. Archetype vs absolute residual (spread), own position, 2023-25

Model_Burke's quantile width is conditioned on the projection only; a variable that predicts |residual| beyond the projection would widen/narrow the right players.

| position | variable | corr with \|res\| | partial corr (after proj) | Q1 mean \|res\| | Q5 mean \|res\| | n |
|---|---|---|---|---|---|---|
| QB | qb_adot_r6 | -0.006 | +0.002 | 5.71 | 5.35 | 2,050 |
| QB | off_ttt_r6 | +0.007 | -0.004 | 5.21 | 5.52 | 2,104 |
| QB | qb_scramble_rate_r6 | +nan | +nan | 5.28 | 5.23 | 2,051 |
| QB | opp_press_r6_z | -0.021 | -0.023 | 5.61 | 5.60 | 1,607 |
| WR | deep_share_r6 | +0.147 | +0.015 | 3.27 | 4.90 | 8,558 |
| WR | sep_prev | -0.018 | -0.001 | 4.83 | 4.61 | 6,226 |
| WR | opp_two_high_r6_z | +0.028 | +0.024 | 4.15 | 4.37 | 5,695 |
| WR | opp_man_r6_z | -0.019 | -0.015 | 4.33 | 4.11 | 5,695 |
| WR | tprr_r6 | +0.282 | -0.014 | 3.18 | 6.46 | 8,558 |
| TE | deep_share_r6 | +0.151 | +0.009 | 2.38 | 4.16 | 4,477 |
| RB | tprr_r6 | +0.099 | -0.003 | 4.31 | 5.39 | 5,137 |
| RB | opp_box_r6_z | -0.021 | -0.021 | 4.27 | 4.10 | 3,561 |
| RB | opp_press_r6_z | -0.020 | -0.009 | 4.24 | 3.99 | 3,561 |
