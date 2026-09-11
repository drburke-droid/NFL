# exp_004_contrarian — notes

## The model (final, 2 judge runs: seed 17 and seed 29, identical because nothing is random)

```
m(pos, FFA) = median(actual - FFA) over the K=1000 ACTIVE training player-weeks of the same
              position nearest in FFA projection
pred  = FFA + 0.5 * m                       (half-way between FFA and its conditional median)
p_q   = FFA + Q_q(actual - FFA) over the same neighbourhood, active rows only, floored at 0 (QB -2)
```

One feature (`baseline_proj`), one grouping variable (`position`), two constants (K, S). Weekly refit
= one sort and ~400 window quantiles. No DK line, no lags, no learned coefficients.

## Visible results (seasons 2023-24, active rows, n = 10,161)

| model | MAE | RMSE | rho | cov80 | pinball | 2023 MAE | 2024 MAE | composite | runtime |
|---|---|---|---|---|---|---|---|---|---|
| ffa (normaliser) | 4.4430 | 6.128 | 0.6704 | 0.762 | 1.5799 | 4.362 | 4.526 | 0.9904 | 5 s |
| minimalist | 4.4436 | 6.125 | 0.6706 | 0.763 | 1.5784 | 4.356 | 4.533 | (0.8909 pre-ref) | 9 s |
| kitchen_sink | 4.4553 | 6.109 | 0.6713 | 0.750 | 1.5710 | 4.368 | 4.545 | (0.8870 pre-ref) | 47 s |
| model_burke_prod (incumbent) | 4.4186 | 6.128 | 0.6727 | 0.765 | 1.5767 | 4.326 | 4.513 | **0.9944** | 186 s |
| **exp_004_contrarian** | **4.4064** | 6.172 | 0.6707 | **0.805** | **1.5494** | **4.316** | **4.498** | **1.0032** | **7 s** |

Composite +0.88% over the incumbent (gate: x1.005 -> 0.9994, passed), +1.3% over FFA. Both seasons
beat the incumbent's MAE. Per position (MAE): QB 5.498 (inc 5.467, ffa 5.495), RB 4.333 (4.363,
4.378), TE 3.484 (3.469, 3.500), WR 4.574 (4.602, 4.628).

### Where the composite gain comes from (term by term vs the incumbent)
- MAE term +0.0012, RMSE term -0.0014, Spearman term -0.0005: the point model is a wash on the
  composite. It wins MAE by targeting the median but pays it back in RMSE, and it cannot match the
  incumbent's small rank gain (0.6727 vs 0.6707) because it has no information beyond FFA.
- Pinball term +0.0018, calibration term +0.0075: the distribution is the whole win. The production
  local-quantile rule draws residual quantiles from ALL training rows, and 34% of those are inactive
  players with actual_ppr = 0; that drags p10/p25/p50 down (ffa's p50 sits 1.2 pts below its pred)
  and under-covers (0.762-0.765). Restricting the neighbourhood to actual_active == 1 gives
  cov80 0.805 and pinball 1.549 with no other change.

## What was tried (harness = judge-identical weekly walk-forward on the visible frame)

Sub-hypothesis (a), recalibration of the FFA line:
- Pure conditional-median target (S=1): MAE 4.412 (-0.7% vs FFA) but RMSE 6.27 (+2.3%); composite
  proxy below the S=0.5 version. Flat optimum S in 0.4-0.6 (MAE 4.410 / 4.406 / 4.404; composite
  0.9015 / 0.9014 / 0.9012 without the pinball term).
- Local MEAN correction (kernel mean regression, the "FFA over-rates tiers" story taken literally):
  MAE 4.538, i.e. WORSE than FFA. On active rows FFA under-predicts the mean (+0.5 bias, a selection
  effect of scoring players who played), so a mean recalibration pushes projections UP while the
  MAE-optimal move is DOWN. The tier over-rating of WR/TE top deciles is real (actual/FFA 0.92-0.95)
  but small next to the skew effect.
- Multiplicative (local median of actual vs local median of FFA) instead of additive: 4.425, worse.
- K = 300 / 600 / 1000 / 2000 / 4000: 4.4205 / 4.4139 / 4.4121 / 4.4121 / 4.4160 at S=1; K=1000 kept.
- Training window all / last 6 / last 4 / last 2 seasons: within 0.002 MAE; all seasons kept.
- Quantile scale 0.9 / 1.05 / 1.1 / 1.2: cov80 0.742 / 0.835 / 0.867 / 0.898, pinball worse each
  way. Per-position scales (QB 1.1, TE 1.05, RB/WR 0.95): pinball flat (1.5495-1.5510), and the
  out-of-sample prior-season coverage by position is not stable (QB 0.814 on 2022, 0.755 on 2023),
  so position-specific widening was rejected as noise-fitting.
- Using inactive rows in the neighbourhood: MAE 4.560, pinball 1.593. Active-only is essential.

Sub-hypothesis (b), DK blend: on the 5,365 visible rows with a DK line, DK-implied PPR has MAE 5.117
vs FFA 4.601 on the same rows; the best blend weight is ~0.05-0.1 (4.596 vs 4.601). On top of the
recalibrated point, w=0.05 changes MAE by -0.0008 and w=0.2 by +0.010. Falsified as expected; DK
is not in the model.

Sub-hypothesis (c), robustness of the incumbent's correction (from the registered submissions):
- Incumbent minus FFA by season x position: 2023 QB -0.024, RB -0.034, TE -0.027, WR -0.046;
  2024 QB -0.033, RB +0.006, TE -0.036, WR -0.005. In 2024 the residual correction did nothing at
  RB/WR; it earns its keep at QB and TE in both seasons (and at QB-low / TE-high tiers).
- This candidate minus FFA: QB +0.003/+0.007 (null), RB -0.056/-0.033, TE -0.026/-0.006,
  WR -0.063/-0.045. The two corrections are only 0.43 correlated. Adding 0.5 x local median shift ON
  TOP of the incumbent mean gives MAE 4.4045 (incumbent 4.4186) — i.e. the incumbent is missing the
  skew term, and the incumbent's QB/TE information is missing from this candidate.
- Suggestion for the lead: the cheapest upgrade to production is (1) active-only local quantiles
  and (2) pred = 0.5 * (Model_Burke mean + local conditional median). Gating the residual GBM off
  at RB/WR is defensible on 2024 but the evidence is one season.

## Honest interpretation

- What is robust: the active-only quantiles. It is a mechanical fix (the current rule mixes a point
  mass at zero into a distribution scored only on players who played) and its effect is the same
  size in both seasons (cov80 0.80 both years). I would expect it to survive 2025.
- What is moderately robust: the median shift. It is a property of the score (MAE on a right-skewed
  target) not of the season; the local median residual is negative in every visible season
  (2016-24 range -0.28 to -0.73 at the position level). S=0.5 is a compromise dictated by the
  0.45/0.20 MAE/RMSE weights; the composite is flat over 0.4-0.6. MAE gain vs FFA was -0.046 (2023)
  and -0.027 (2024); vs the incumbent -0.010 / -0.015. Season-level variation in the FFA bias
  (2021-22 ~0, 2023-24 +0.4/+0.6) means the size of the gain will move; the sign should not.
- What would not survive: any claim that this model is "better" than the incumbent at the point
  level. On the composite's point terms (MAE + RMSE + rho) it is level with the incumbent, slightly
  behind on rank and RMSE. It ties the incumbent with 1 feature instead of 104 and 7 s instead of
  186 s, which is the contrarian result: the residual model's mean-level information is worth about
  as much as knowing that the score is MAE.
- Risk in the private run: 2025 is one season; the gain over the incumbent is 0.9% composite, of
  which 0.75 points is the calibration term. If 2025 residual spread is unusual, cov80 could drift
  off 0.80 (it moved 0.06 between seasons for QB in the visible data) and the calibration term
  would shrink for everyone, but the incumbent's contaminated quantiles start 0.035 further away.

## Runtime
7 s for all 36 weekly fits (seed 17), 7 s at seed 29. Per weekly fit < 0.3 s.
