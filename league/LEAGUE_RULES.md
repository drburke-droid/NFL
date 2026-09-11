# NFL Model League — rules (Version 0, Round 1)

Purpose: find out, with a deterministic judge and unseen weeks, whether any modelling hypothesis
beats the incumbent weekly projection (Model_Burke: FFA + DraftKings blend with a calibrated
residual correction and conformal quantiles). Nothing here touches the production generator.
Promotion is a report, never an automatic swap.

## Data
- `league/data/visible_frame.parquet`: 2016-2024 player-weeks (QB/RB/WR/TE with an FFA line),
  104 point-in-time features, actual outcomes. Built by `league/data/build_frame.py`.
- `league/data/holdout_features.parquet`: 2025 features, no outcomes.
- 2025 outcomes live OUTSIDE the repository (private judge only). Candidates must not read any
  2025 outcome from anywhere else in the repo either (the data folder contains 2025 box scores;
  reading them is cheating and the red-team check greps for it). The judge passes all data in memory.
- Known compromise: `mkt_*` is the DK closing line (~2 h pre-kick), not a Wednesday line.

## Protocol
- Visible evaluation: seasons 2023-2024, weeks 1-18, expanding walk-forward (train = every
  visible row before the test week). Scored rows = players who actually played (actual_active == 1):
  known inactives are handled live by the production generator, so predicting zeros for bench
  players earns nothing here. Every test row must still receive a prediction. REFIT = "week" or "season" declared by the candidate.
- Private evaluation: 2025, same protocol, train includes 2025 weeks before the test week.
- Scoring config v1 (frozen): composite = 0.45·(FFA MAE ÷ MAE) + 0.20·(FFA RMSE ÷ RMSE)
  + 0.15·(ρ ÷ FFA ρ) + 0.10·(incumbent pinball ÷ pinball) + 0.05·calibration(80% coverage)
  + 0.05·min-season MAE ratio. Higher is better. The incumbent's composite is the bar.
- Seeds: seed 17 for scoring; a promotion candidate is re-run with seed 29.

## Candidate contract
`league/candidates/<exp_id>/` with `proposal.yaml` (hypothesis, features, algorithm, expected
direction, failure condition — written BEFORE code), `model.py` implementing `fit_predict`
(see `model_template.py`), and `NOTES.md` (what happened). One directory per candidate; a
candidate agent edits nothing outside its directory.

## Gates for promotion (mechanical)
visible composite ≥ incumbent × 1.005; private composite ≥ incumbent × 1.002; no evaluation
season with MAE > incumbent × 1.01; coverage ≥ 99.9%; seed 29 confirms; red-team approval.
Private submissions: at most 2 per round, authorised by the lead only.

## Baselines (must be beaten, all registered)
prev_game, trail4, ffa (normaliser), dk, blend, minimalist (ridge, 11 features),
kitchen_sink (LightGBM, all features), model_burke (package quantiles), model_burke_prod (incumbent: package mean + production local quantiles).

## Incumbent representation (fixed after Round 1 re-registrations)
The package is fitted the way the generator fits it: played rows only (box score present), FFA
baseline, no market feature (production blends DK per stat into the live slate baseline, which the
frame cannot reproduce), point projection = `Model_Burke` (residual correction + median bias
offset), quantiles = the generator's local projection-conditioned residual quantiles on played rows
from the generator's history window (2023+, or the single prior season for the 2023 fold; the 2016+
pool over-covered at 0.83). Registrations before 2026-09-11T20:30Z used other wirings (all-row
training, the mean, the DK proxy as a feature, the 2016+ pool) and are superseded; the judge reads
the LATEST `model_burke_prod` row.

## Honesty notes
- The lead (Claude) has used 2025 in earlier studies; the holdout is pristine for candidate
  agents, not for the lead. Candidate agents are told nothing about 2025 results.
- Prior evidence: ~45 features tested on this baseline in September 2026 were all within
  ±0.1% MAE. Candidates are told this so they do not re-run known nulls.
