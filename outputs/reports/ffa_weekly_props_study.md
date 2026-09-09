# Weekly FFA vs DK reception-yds lines — retro-train verdict (2026-09-09)

**Question.** The 📡 Props Watch model's known blind spot is news. Does the weekly
FFAnalytics consensus (`rec_yds`, `rec_yds_sd`, `injury_status`) — a second
projection of the priced stat — add calibrated edge on top of Model_Burke's
`(z, line)` logistic?

**Data.** Home-PC backfill, 49 weekly raw-stat files: 2023 wk9-18 (wk10 missing),
2024 wk1-20, 2025 wk1-20 → `data/ffanalytics/FFAn_weekly/raw_stats_S_wkW.csv`.
Joined by normalised name to the 2023-25 closing-line frame
(`data/props_frames/props_player_reception_yds.parquet`); 98% of priced player-weeks
match. Rows with illegal multi-book-averaged prices (|price| < 100) dropped, as in
`model_burke_props_run2.py`. Script: `scripts/ffa_weekly_props_study.py <pkg>`.

**Not leaked.** FFA is *worse* than the closing line at predicting the outcome
(MAE 19.67 vs 19.05), so these are genuine pre-game scrapes, not post-hoc numbers.

**Signal in isolation.** Tiny and non-monotone. Over-rate by FFA−line gap:

| ffa_gap (yds) | n | over-rate |
|---|---|---|
| < −10 | 22 | .364 |
| −10 … −5 | 64 | .422 |
| −5 … 0 | 502 | .460 |
| 0 … 5 | 2365 | .480 |
| 5 … 10 | 1689 | .510 |
| ≥ 10 | 384 | .471 |

FFA sits 4.3 yds above the line on average (projection-site optimism). The gap
correlates +0.03 with the outcome residual, and −0.08 with Model_Burke's own
correction — it is a *different* opinion, just not a better one. 2024 shows no
direction at all (gap>5: .478, gap<0: .477); the whole bin pattern is 2025.

**Walk-forward (fit on strictly earlier weeks, 2024 wk1 → 2025 wk22, n=4,811).**

| model | logloss | Brier | AUC | @5% n / ROI | @8% n / ROI |
|---|---|---|---|---|---|
| A `z, line` (shipped) | .6934 | .2501 | .499 | 113 / −3.0% | 29 / +23.8% |
| B A + ffa_gap | .6934 | .2501 | .508 | 129 / −6.0% | 30 / +22.8% |
| B2 A + ffa_gap/line | .6933 | .2501 | .517 | 167 / −3.6% | 42 / +6.0% |
| C B + injury(Q) | .6940 | .2504 | .505 | 139 / −3.3% | 38 / +17.1% |
| F ffa_gap only | .6933 | .2501 | .512 | 115 / −6.7% | 30 / +25.8% |
| market fair | .6931 | .2500 | .516 | | |

Calibration is unchanged to four decimals. The 19 bets that only the FFA-augmented
model flags go 8-11 (−16.7%); the 110 both flag are the same bets. Per season the
FFA models are worse in 2024 and worse in 2025 at 5%, equal at 8%.

**Decision.** Weekly FFA stays **display-only** in the 📡 tab (`ffa_yds`, `ffa_sd`,
`inj` columns). No model change. Full-sample coefficient on ffa_gap is 0.0115/yd —
a 10-yd disagreement moves P(over) ~3 pts, well inside noise.

**Only thing worth a future look.** Deep FFA *unders* (gap < −5, n=86, over-rate .41)
would clear −115 vig if it held, but 2024 shows nothing and the strategy is
overs-only by prior validation. Revisit only if the 2026 ledger shows the same
pattern in-season.
