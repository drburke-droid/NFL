# exp_002_components — notes

**Hypothesis (proposal.yaml, unchanged):** modelling the box-score components as separate outcomes
shrunk toward FFA's own stat line, and rebuilding PPR from them plus FFA's TD/turnover lines, beats
correcting the PPR total directly because component-level errors are less noisy and more predictable
from prior usage.

## Final visible result (judge, seed 17; seed 29 identical — the model is deterministic)

| | exp_002_components | model_burke_prod (latest, 19:18Z) | ffa (latest, 19:18Z) |
|---|---|---|---|
| composite | **1.0030** | 0.9962 | 0.9938 |
| MAE | **4.4064** | 4.4186 | 4.4430 |
| RMSE | 6.1657 | 6.1278 | 6.1284 |
| Spearman | 0.6727 | 0.6727 | 0.6704 |
| pinball | **1.5499** | 1.5579 | 1.5593 |
| cov80 | **0.8034** | 0.8278 | 0.8243 |
| MAE 2023 / 2024 | 4.3199 / 4.4949 | 4.3261 / 4.5132 | 4.3622 / 4.5256 |
| MAE by position (QB/RB/WR/TE) | 5.505 / 4.309 / 4.580 / 3.498 | 5.467 / 4.363 / 4.602 / 3.469 | 5.495 / 4.378 / 4.628 / 3.500 |
| runtime (both seasons, REFIT=season) | 22 s | 324 s | 7 s |

Visible gate (composite >= incumbent x 1.005 = 1.0012): passed. No season worse than the incumbent.
The other registered candidate at the time of writing, exp_004_contrarian, sits at 1.0032 with MAE 4.4064 —
effectively tied with this one.

## What was tried (6 judge-run budget: 3 used; everything else was an offline replay of the judge protocol)

1. **v1 — per-position, per-component ridge on `actual - ffa_line`, shrinkage learned on the last training
   season, mean-centred local quantiles.** MAE 4.4432 = FFA. Component diagnostic on 2023:
   - usage components are very predictable (corr(resid_hat, resid): attempts 0.73, carries 0.65-0.76,
     targets 0.58-0.71) — but they carry no points;
   - production components against FFA's own line are essentially unpredictable: rec 0.08-0.14,
     rec_yds 0.05-0.09, rush_yds 0.01-0.06, pass_yds 0.11. Raw ridge corrections made every yardage
     component WORSE; learned shrinkage collapsed to 0-0.4. **The core wager is false: FFA's stat line is
     already efficient at the component level, and feeding predicted usage into the production models does
     not help.**
2. **v2 — add an MAE-optimal offset per component (median of the validation residual) next to the
   shrinkage.** MAE 4.4073 (-0.8%), RMSE +0.9%. This is where the whole gain comes from: the outcome
   distribution of every component is right-skewed (TDs and long gains), so FFA's mean-type line sits above
   the median; the MAE-optimal point estimate shades each component down (rec_yds -2 to -6 yds, rush_yds
   -3 to -5, receptions -0.1 to -0.2). The learned offsets are the component-level version of "predict the
   median, not the mean".
3. **Estimator alternatives:** LightGBM median regression per component (MAE 4.4127, 175 s) and linear
   quantile regression (4.4085, 311 s) — no better than ridge + offset and far slower. Dropped.
4. **Per-position multiplier on FFA's TD/INT/fumble points** (WR 0.8, TE 0.6 chosen on validation): MAE
   unchanged, RMSE worse. Dropped.
5. **Offset fraction chosen on validation with the judge's MAE/RMSE weights** (grid 0..1): MAE 4.4064,
   RMSE 6.1657 — recovers a third of the RMSE cost for no MAE loss. Kept.
6. **Quantiles:** spread from played rows only (as the re-registered baselines do), re-centred on the
   median of the local residual (consistent with a median-type point estimate), spread multiplier tuned to
   80% coverage on the model's own validation predictions rather than on baseline_proj. cov80 0.7807 ->
   0.8034, pinball 1.576 -> 1.550. Worth ~+0.007 composite.
7. **Robustness variants (offline):** 2-season selection window, 2016+ history, ridge alpha 100 — all within
   +/-0.001 composite and +/-0.0015 MAE. Default (1-season window, 2018+, alpha 30) kept.

## Honest interpretation

- The gain over the incumbent (+0.28% MAE, +0.5% pinball, +0.7% composite) is real on the visible folds but it
  is **not** evidence that components are more predictable than points. It comes from (a) shading a mean-type
  projection toward the conditional median, applied component by component and tuned to the judge's
  MAE/RMSE mix, and (b) a tighter, correctly-centred 80% interval. A direct median regression on the points
  residual gave a very similar MAE offline (RB 4.28, WR 4.48 on 2023) — the component layer adds structure
  and a natural place to shrink, not new information.
- The cost is RMSE (+0.6% vs the incumbent) and a bias of +0.98 (under-prediction on played rows), which is
  what a median-type estimate looks like on a skewed outcome. Anyone consuming the mean (e.g. for stacking
  or DFS ownership work) should not use this number.
- Would it survive 2025? The mechanism is distributional (skew) rather than a feature effect, the offsets are
  stable in sign and size across the 2022 and 2023 validation seasons (WR rec_yds -5.8 / -5.8, RB rush_yds
  -4.8 / -3.9), and the 2-season selection window gives the same answer, so I expect the MAE gain to
  persist at roughly half to all of its visible size (the private gate needs 0.2%). The pinball/coverage gain
  depends on the 2025 spread resembling 2024's; the multiplier is re-learned each season so the risk is a
  coverage miss of a couple of points, not a collapse. I would not expect anything from the usage stage in an
  unseen season either — it was neutral here.
- Not a promotion of the hypothesis as written: the falsifiable claim (component residuals are more
  predictable than the points residual) failed; the model passes the gate for a different reason and the
  NOTES say so.

## Files
- proposal.yaml — hypothesis, features, algorithm, failure condition (written before code).
- model.py — final model (CFG defaults = registered run, code hash 2eae195a826e).
