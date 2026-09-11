# exp_003_stacker — notes

**Hypothesis (fixed in proposal.yaml):** a heavily regularised stacker that learns the FFA/DK blend weight and a
small lag correction as a function of context (position, projection tier, DK line availability, lag volatility,
week), trained on the FFA residual with a shrinkage k chosen by inner walk-forward, lowers visible MAE by >= 0.5%
vs FFA and matches or beats the incumbent composite.

## Final model (model.py, CFG defaults)
- Stage 1: LightGBM, **objective = L1**, 6 leaves, 200 trees, lr 0.05, min_child_samples 400, reg_lambda 5, on the
  residual `actual_ppr - baseline_proj`, trained on **active rows only** (actual_active == 1).
  Features: baseline_proj, mkt_has_line, DK-minus-FFA delta, lag l1/r3/r6 minus FFA, lag std3, games played,
  week, ffa_injury_q, ctx_implied, ctx_spread, lag targets/carries r3, ffa_rec, ffa_rush_yds, position one-hots.
- Stage 2: `pred = FFA + k * correction`, k in {0, 0.1, ..., 1.2} and a per-position median offset flag (the
  incumbent's trick) chosen on an inner walk-forward window = the last 34 weeks of the training frame (model fit
  on everything before them), by the MAE+RMSE part of the league composite relative to FFA on that window.
  k lands at 0.7-0.9 every week; the offset flag is never selected (the L1 model already absorbs it).
- Quantiles: the local residual-quantile rule from run_baselines.py (same position, 300 nearest rows in
  baseline_proj, active rows only), centred on the local median and re-centred on `pred`, with the band width
  multiplied by a factor chosen on the same inner window so [p10, p90] covers 80% (conformal-style). The
  multiplier is ~1.0 every week once the centre is coherent.
- Weekly refit; the full 36-week visible walk-forward takes 55 s (~1.5 s per week).

## Visible results (judge, seed 17) vs registry baselines
| model | MAE | RMSE | Spearman | cov80 | pinball | 2023 MAE | 2024 MAE | composite |
|---|---|---|---|---|---|---|---|---|
| **exp_003_stacker** | **4.396** | 6.193 | 0.6713 | **0.798** | **1.548** | **4.306** | **4.489** | **1.0040** |
| model_burke_prod (incumbent, latest row 19:48Z) | 4.411 | 6.150 | 0.6717 | 0.830 | 1.565 | 4.315 | 4.509 | 0.9956 |
| ffa (latest row) | 4.443 | 6.128 | 0.6704 | 0.824 | 1.559 | 4.362 | 4.526 | 0.9938 |
| minimalist (latest row) | 4.444 | 6.125 | 0.6706 | 0.823 | 1.558 | 4.356 | 4.533 | 0.9943 |

- Composite ratio vs incumbent: 1.0040 / 0.9956 = **1.0084** (gate is 1.005). Both seasons beat the incumbent's
  season MAE (4.306 vs 4.315, 4.489 vs 4.509). Coverage 100%, quantiles monotone.
- Seed 29 (judge, registered as `exp_003_stacker_seed29`): MAE 4.397, 2023 4.305 / 2024 4.491, composite 1.0034 —
  the LightGBM subsampling noise is ~0.001 MAE / ~0.0006 composite.
- Note the incumbent has been re-registered several times during the round (MAE 4.419 -> 4.490 -> 4.411);
  the comparison above uses the latest row at the time of writing.

## What was tried (dev harness = same walk-forward as the judge, no registry writes; 20 variants)
| variant | MAE | RMSE | cov80 | judge-equivalent composite |
|---|---|---|---|---|
| ridge with interactions + auto offset (mean-optimal) | 4.421 | 6.161 | 0.808 | ~0.9995 |
| LightGBM L1, k by MAE, FFA-centred interval scale | 4.397 | 6.212 | 0.814 | ~1.000 |
| LightGBM Huber + monotone constraints | 4.404 | 6.225 | 0.814 | ~0.998 |
| combo: lgb median + ridge mean, (k1, k2) by MAE | 4.407 | 6.227 | 0.830 | ~0.995 |
| combo, (k1, k2) by composite criterion | 4.402 | 6.192 | 0.794 | ~1.002 |
| lgb L1, k by composite criterion (coherent calibration) | 4.398 | 6.194 | 0.796 | ~1.003 |
| + min_child_samples 400 (**final**) | 4.396 | 6.193 | 0.798 | 1.004 |
| final trained on all rows (not active-only) | 4.417 | 6.192 | 0.799 | ~1.000 |
| final without DK features | 4.398 | 6.201 | 0.798 | ~1.003 |
| final without any lag feature | 4.395 | 6.188 | 0.796 | ~1.004 |
| bare: baseline_proj, position, week, injury only | 4.403 | 6.201 | 0.796 | ~1.001 |
| recency weighting (half-life 3 seasons) / last 5 seasons only | 4.396 | 6.184-6.187 | 0.796 | ~1.004 |
| 4 leaves / 400 trees at lr 0.03 / 17-week inner window | 4.396-4.399 | 6.18-6.21 | 0.79-0.80 | ~1.002-1.004 |

## Honest interpretation
1. **The stacking part of the hypothesis is falsified.** The learned weight on the DK line is ~0 (gain
   importance 0.1%; removing DK features changes MAE by +0.001) and the lag terms are ~0 too (removing all of
   them *improves* MAE by 0.001). The exploration showed why: on the visible window the DK closing line is worse
   than FFA at every blend weight (MAE 4.60 at w=0 rising monotonically to 5.12 at w=1 on lined rows), and lag
   deltas have correlation ~0 with the FFA residual. Context-dependent blend weights have nothing to blend.
2. **What actually moves MAE is the loss function.** FFA is a mean forecast; fantasy points are right-skewed
   (zero floor, lumpy TDs), so the MAE-optimal point is the conditional median, which sits 0.5-1.0 pts below
   FFA for projections above 3 and slightly above it below 3. An L1-objective GBM on baseline_proj + position
   (+ ffa_rec / ffa_rush / implied total as tier refinements) learns exactly that shift; k = 0.7-0.9 is the
   shrinkage the inner window asks for. The gain shows up in every tier, both lined and unlined rows, and every
   week bucket (largest early in the season: 4.47 -> 4.38 in weeks 1-4).
3. **The cost is RMSE and bias.** RMSE 6.193 vs 6.150 (incumbent) / 6.128 (FFA); mean bias +1.09 (we
   under-predict on average). Spearman is unchanged (0.6713 vs 0.6717) — the ranking of players is not
   improved. A user who wants expected points (lineup EV, trade value) should not prefer this point forecast;
   a user scored on MAE should. The composite weights MAE 0.45 vs RMSE 0.20, so the trade wins the composite.
4. **Calibration was the second lever.** Choosing the interval width around the model's own inner-window
   predictions (not around FFA) puts cov80 at 0.796-0.798 vs 0.83 for the incumbent, worth ~0.007 composite,
   and the median-centred local quantiles give the best pinball in the registry (1.548).
5. The per-position median offset (incumbent trick) is never selected once the L1 model is present; on its own
   (ridge + offset) it gives MAE 4.421 — roughly the incumbent's MAE.

## Would it survive an unseen season?
Probably yes for the MAE/pinball part, because the mechanism is structural (median < mean for skewed
fantasy outcomes) rather than a fitted feature effect: the same shift shows in every training season and in
both visible seasons (2023 -0.020, 2024 -0.020 vs the incumbent). The 2025 risk is on the RMSE side: if the
private year has a different skew (fewer blowups) the median shift over-corrects and the RMSE penalty grows;
and the DK line has ~60% coverage in 2025 vs 24-57% here — irrelevant to this model since it ignores it.
I would expect a private composite gain of roughly half the visible one (+0.4% vs the incumbent's 1.002 gate
is plausible but not certain), and I would not credit any of it to "learned blend weights".
