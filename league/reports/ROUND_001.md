# NFL Model League — Round 1 report (2026-09-11)

**Result: no promotion.** Four candidate models were built by independent agents against a frozen
2016-24 frame and a deterministic judge. All four beat the incumbent's visible composite, none by
the 0.5% promotion margin once the incumbent was represented the way production actually runs it.
Both private (2025) submissions cleared the private margin. Every candidate's gain decomposes into
two things the incumbent already does: shading the point toward the conditional median (MAE is
minimised by the median, and fantasy points are right-skewed) and putting the 80% interval on 0.80
exactly. No candidate found new information about players.

## Final board (visible = 2023-24 played rows, n = 10,161; private = 2025 played rows, n = 5,152)

| model | visible MAE | RMSE | rho | bias | cov80 | pinball | composite | vs incumbent | private composite | private vs incumbent |
|---|---|---|---|---|---|---|---|---|---|---|
| ffa (normaliser) | 4.443 | 6.128 | .670 | +0.50 | .824 | 1.559 | 0.9938 | | 0.9933 | |
| **model_burke_prod (incumbent)** | 4.411 | 6.150 | .672 | +0.85 | .811 | 1.562 | **1.0004** | 1.000 | **1.0004** | 1.000 |
| exp_001_quantiles | 4.400 | 6.141 | .672 | +0.83 | .795 | 1.541 | 1.0050 | 1.0046 | 1.0053 (slot 2) | 1.0049 |
| exp_003_stacker | 4.396 | 6.193 | .671 | +1.09 | .798 | 1.548 | 1.0038 | 1.0034 | (1.0049, lead-only) | |
| exp_002_components | 4.406 | 6.166 | .673 | +0.98 | .803 | 1.550 | 1.0033 | 1.0029 | not submitted | |
| exp_004_contrarian | 4.406 | 6.172 | .671 | +0.94 | .805 | 1.549 | 1.0023 | 1.0019 | 1.0065 (slot 1) | 1.0061 |

Gates: visible ≥ 1.005 × incumbent (none pass; exp_001 misses by 0.0004), private ≥ 1.002 × incumbent
(both submissions pass), no season worse than incumbent × 1.01 (all pass), seed 29 confirms (exp_001
1.0039 vs 1.0050 at seed 17 under the earlier reference; exp_003 1.0034; exp_002 and exp_004 have no
live randomness). Reference baselines: prev_game 6.040, trail4 5.293, dk 4.715, blend 4.539,
minimalist 4.444, kitchen_sink 4.455, model_burke (package quantiles) 4.411 / composite 0.9935.

## Where the candidates' edge comes from (visible, vs the final incumbent)

Composite terms, candidate minus incumbent:

| term (weight) | exp_001 | exp_003 | exp_002 | exp_004 |
|---|---|---|---|---|
| calibration (.05): incumbent cov80 .811 → candidates .795-.805 | +0.0015 | +0.0022 | +0.0018 | +0.0014 |
| MAE (.45): 4.411 → 4.396-4.406 | +0.0011 | +0.0015 | +0.0005 | +0.0005 |
| pinball (.10) | +0.0014 | +0.0009 | +0.0008 | +0.0008 |
| RMSE (.20) | +0.0003 | −0.0014 | −0.0005 | −0.0007 |
| Spearman (.15) | +0.0001 | −0.0001 | +0.0002 | −0.0002 |
| stability (.05) | +0.0002 | +0.0002 | +0.0002 | +0.0001 |
| **total** | **+0.0046** | **+0.0034** | **+0.0029** | **+0.0019** |

Subtracting a single scalar from the incumbent's predictions reproduces most of the MAE gains: on
the visible folds a −0.9 shade takes the (2016+ pool) incumbent from 4.411 to 4.407; on 2025 a −0.3
shade takes it from 4.4005 to 4.3916, against candidate MAEs of 4.383-4.390. Bias-matched, exp_002
is worse than the incumbent (red-team check, 4.4201 vs 4.4186 on the earlier reference). Spearman
is unchanged to three decimals for every candidate: no candidate ranks players better. Two
residuals survive scalar matching: exp_004's projection-conditioned local median (per position,
1,000 nearest FFA neighbours), 0.2% MAE beyond the best scalar shade on 2025 at a 0.3% RMSE cost;
and exp_001's market-conditioned boosted median, 0.010 MAE (0.2%) at matched bias on the visible
folds but weakly significant (25/36 weeks, p = 0.06, concentrated in weeks 1-2 and rows without a
DK line) and partly information the frame-incumbent is denied by design (no market feature). The
one robust candidate-specific result of the round is exp_001's pinball: better than the
incumbent's in 30 of 36 weeks (p < 1e-4), unchanged at matched coverage.

## Candidate summaries (each agent's own notes say the same thing)

- **exp_001_quantiles** (boosted quantile regression on active rows, ridge mean). Proposal said the
  mean stays as `pred`; the final model ships `pred = 0.5·mean + 0.5·p50`, which is where the MAE
  gain lives (with the proposal's own rule it ties the incumbent, 1.0003 vs 1.0004). Pre-registered
  failure condition (pinball ≥2% better) fired at 1.4%. Coverage .795. Best composite; private
  1.0053. Red-team: REJECT for promotion (clean code, no leakage; drift + failure condition +
  below the visible gate), with a recommendation to re-enter the balanced quantile layer and the
  market-conditioned median as a new, pre-registered exp_id.
- **exp_002_components** (component targets fed into a two-stage ridge). Pre-registered failure
  condition fired (learned shrinkage collapsed to 0-0.4); the gain came from an unproposed median
  offset tuned against the judge's literal weights plus a coverage multiplier. Red-team: REJECT for
  promotion, clean code, honest notes.
- **exp_003_stacker** (L1-objective LightGBM stacker learning the FFA/DK blend). Stacking
  falsified (DK weight ~0, dropping lags improves MAE); the L1 loss learns the tier × position
  median shift. Best visible MAE (4.396), worst RMSE (6.193).
- **exp_004_contrarian** (FFA + 0.5 × local median residual, two constants, no learner). Designed
  as the null model and it matches every learner on MAE. Red-team: APPROVE WITH NOTES (edge = the
  calibration term). Private 1.0065, the best private score of the round.

Red-team reviews (read-only agents, exp_001/exp_002/exp_004) found no I/O, no leakage and no
transductive statistics; the seed gate is vacuous for exp_002/exp_004 (no live randomness) and live
but immaterial for exp_001 (seed 17 vs 29 agree to 4 dp).

## What the round actually established

1. **The incumbent's representation moved the bar by 0.011 composite, more than any candidate's
   edge.** Four re-registrations were needed: all-row training → played rows (the generator's
   history holds only players with a box score); package mean → the shipped point projection
   (median bias offset); the DK-implied PPR proxy dropped from the training features (the
   generator's history has no market column; immaterial, 4.412 vs 4.411); and the residual pool for
   the local quantiles narrowed to the generator's 2023+ window (the 2016+ pool over-covered at
   0.83; production reports 0.79-0.81). Composite went 0.9962 → 0.9913 → 0.9956 → 1.0004.
   `LEAGUE_RULES.md` now fixes this wiring.
2. **Loss geometry, not information.** Every gain is median shading plus coverage tuning. The
   incumbent's own scorecard says the same: its residual GBM adds ~0.2% over the bias offset alone
   (control_k0) in every season.
3. **The frame's DK column is a poor proxy.** `mkt_ppr` alone scores 4.715 vs FFA 4.443 and the
   0.6/0.4 blend 4.539, whereas production's per-stat closing-line blend beats FFA. Candidates that
   tried to use it learned to ignore it. A Round 2 frame needs per-stat lines.

## Production follow-up (not applied; the generator is untouched)

- Nothing to change for coverage: production's local quantiles already sit at 0.79-0.81 on 2025.
- A projection-conditioned median offset (exp_004's rule) instead of the package's global median
  offset is worth ~0.2% MAE on played rows at ~0.3% RMSE. Below the 0.5% bar this league set for
  itself and inside the weekly noise (slate MAE sd 0.8); not recommended on this evidence.

## Round 2 rules changes (already in `AGENT_BRIEF.md` / `LEAGUE_RULES.md`)

- Proposals are hashed before the first judge run; unproposed prediction rules are reported as drift.
- Candidates must report MAE at matched bias and state whether they shade to the median.
- Incumbent wiring fixed as above. Private budget: 2 of 2 used this round (exp_004, exp_001).

Registry: `league/registry/experiments.jsonl` (63 rows, every attempt kept), private aggregates in
`league/registry/private_log.jsonl`. Predictions dumps (`league/registry/submissions/`) are not
committed; every row carries the candidate's code hash.
