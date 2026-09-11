# Candidate agent brief (read fully before writing code)

You are one competitor in the NFL Model League. You produce ONE candidate model in your own
directory and nothing else. Read `league/LEAGUE_RULES.md` first.

## Your working contract
1. Work only inside `league/candidates/<your exp_id>/`. Do not edit any other file in the repo.
2. Start by writing `proposal.yaml`: hypothesis (one falsifiable sentence), targets, features you
   will use (frame column names), algorithm, expected_direction, failure_condition. Write it
   BEFORE code and do not revise the hypothesis afterwards.
3. Implement `model.py` per `league/candidates/model_template.py`: `REFIT` and
   `fit_predict(train, test, seed)`. No I/O of any kind in model.py. Keep a weekly fit under 30 s.
4. Score yourself on the visible folds with, from the repo root:
   `python league/judge/score.py --candidate league/candidates/<exp_id> --seed 17`
   It prints MAE/RMSE/Spearman/coverage/pinball, per-season MAE and the composite, and registers
   the result. You may iterate on the visible folds (registry keeps every attempt) but the
   proposal's hypothesis stays fixed; a new hypothesis is a new exp_id.
5. Finish with `NOTES.md`: what you tried, the final visible metrics vs the baselines in
   `league/registry/experiments.jsonl`, honest interpretation, runtime.
6. Report back (in your final message): exp_id, hypothesis, final composite vs the incumbent's
   composite (label `model_burke_prod` in the registry) and vs `ffa`, per-season MAE, runtime, and
   whether you believe the gain would survive an unseen season.

## Scoring population
The judge scores only players who actually played (actual_active == 1 in the outcomes). `train`
carries actual_active so you can filter or weight; predicting zeros for likely-inactive bench
players earns nothing. Every `test` row must still get a prediction.

## What you must never do
- Read any file with 2025 data or any actual outcome outside the `train` frame the judge passes.
  In particular, never open `data/`, `db/`, `outputs/` or `league/data/holdout_*`.
- Change judge code, scoring config, baselines, or another candidate's directory.
- Claim a gain you did not measure with the judge command above.

## Known facts, so you do not repeat them
- Baseline reference (2025 season, full pool): FFA consensus MAE 4.17; a hindsight oracle that
  knows each player's true season average scores 4.04. The competitive range is ~1-3%.
- Tested in Sept 2026 and found NULL on top of the FFA + DK baseline (all within ±0.1% MAE):
  snap share, QB1 absence, rest/home, weather, pass funnel, plays, DvP, TD/efficiency luck,
  route share (l1/r3/r6), TPRR/YPRR, usage surprise, red-zone routes, opponent man/pressure,
  man-coverage edge, team dropbacks, vacated routes, QB time-to-throw × pressure, QB aDOT ×
  two-high, WR deep share × anything, NGS separation × man, RB × box, and a 10-season training
  history. Residual GBMs at full strength were WORSE than the baseline; only shrunk corrections
  and the FFA/DK blend weights moved MAE. A plain ridge on those features also failed.
- What has NOT been tested: distribution shape (quantile regression, mixture with a point mass
  at zero for inactive risk, position/tier-specific spread), component targets (targets, carries,
  receptions) fed back into points, stacking FFA/DK/lags with a low-capacity calibrator, and
  nonlinear blend weights that depend on line availability or projection tier.
- The incumbent's distribution is conformal-calibrated at 80% coverage 0.80; beating it on
  pinball/calibration is a legitimate way to win the composite even at equal MAE.
- Round 1 lesson (Sept 2026): on the played-only scoring population the MAE-optimal point sits
  ~0.3-0.9 PPR BELOW the conditional mean (right-skewed outcomes). Every Round 1 candidate's MAE
  gain came from shading toward the active-row median (a scalar offset reproduces ~96% of it), and
  every calibration gain from tuning cov80 onto the judge's 0.80 target. Both are levers the incumbent
  already pulls (median bias offset, conformal quantiles). If your model uses them, say so in the
  proposal and report MAE at matched bias; a gain that vanishes at matched bias is not new information.
- Your proposal is hashed by the lead before your first judge run. A prediction rule that is not in
  the proposal (e.g. blending p50 into `pred`) is drift and is reported as such.
