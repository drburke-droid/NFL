# NFL DFS Projection System spec (2026-09-09) vs Model_Burke — assessment

Date: 2026-09-10. Spec: `Downloads/NFL_DFS_PROJECTION_SYSTEM_SPEC.md` (v0.1).
Studies run today: `scripts/spec_features_study.py`, `scripts/spec_mechanistic_test.py`
(reports in this folder), plus an injury-designation calibration (inline, numbers below).

## Verdict

1. **The spec's own MVP success criterion — beat trailing averages, FFA alone, market alone,
   and a simple FFA + market blend on an unseen season — is already what Model_Burke does.**
   Model_Burke *is* the FFA + market blend plus a calibrated distribution. Section 28 of the
   spec says plainly: if the mechanistic system cannot beat that blend, the complexity is not
   justified. Every mechanistic feature we have been able to build so far fails that test.
2. **Most of the data the spec buys from Fantasy Points is already free in nflverse.** The
   participation feed (`nflreadpy.load_participation`) carries, per play, the 11 offensive
   players on the field, the targeted receiver's route, coverage type (Cover 0-6, 2-man),
   man/zone, pressure, time to throw, defenders in box, personnel and formation. Complete for
   2023-25, partial (pass plays only) 2016-22. FTN charting (2022+) adds motion, play action,
   RPO, screens, box counts. I built `data/participation/routes_{2016..2025}.parquet`
   (per player-game routes, route share, TPRR, YPRR, red-zone routes, man/pressure splits) and
   `team_def_*.parquet` (opponent man/pressure/two-high rates). Targets reconstructed from the
   feed match nflverse box scores at r = 0.9998.
   **Caveat (checked 2026-09-10): the 2023+ feed is FTN charting released in bulk after the
   post-season; pre-2023 is NGS. It is a backtesting dataset, not an in-season one.** The
   participation loader caps at 2025 and no 2026 file exists. Any live weekly use of routes,
   coverage or pressure needs a paid in-season source (Fantasy Points Data, FTN Data, PFF).
3. **Tested on the walk-forward harness, the spec's Tier-A opportunity features add nothing on
   top of the FFA weekly baseline.** Same result as the 2026-09-09 feature search.
4. **Do not subscribe for projection accuracy.** Details and the one exception below.
5. **One measured upgrade shipped:** the generator's P(plays) for a Questionable player DK has
   not priced while teammates are priced is now 0.20 (was 0.40).

## What the spec asks for vs what exists

| Spec phase | Status here |
|---|---|
| 1 Foundation: IDs, nflverse DB, scoring, point-in-time, FFA history, odds collector, baselines | Done in substance. gsis_id anchor; odds snapshots with `snapshot_time`; FFA weekly 2023-25; DK props 2023-25; every generated CSV kept. Missing: formal `available_at` join for backtests (we use closing lines; noted as a limit in the README). |
| 2 Opportunity: routes, TPRR, carry share, teammate on/off, role state, usage surprise | Data now built free (above). Features tested today: null. Teammate absence shares already measured (context-effects study) and drive the live redistribution. |
| 3 Game environment: possessions, plays, pass rate, game script, red-zone access | plays_r6, pass funnel, Vegas all tested null on FFA. FFA + DK already price the environment. |
| 4 Matchup: coverage, route x coverage, pressure x route, run scheme, OL units | Coverage/pressure/man-edge tested today: null. Run scheme/gap and OL lineup units are the only pieces we cannot build free. |
| 5 Market / consensus calibration | Done: DK stat lines blended before scoring (yards 0.9 / rec 0.3 / TD 0.6 by study), FFA as baseline, market-only projection column. |
| 6 Joint simulation: correlations, percentiles, threshold probabilities | **Not done.** Quantiles are per-player marginals (conformal, 80% coverage 0.80). No QB-WR1 or game-stack correlation. This is the README's stated known limit and the one SaberSim cares about most. |
| 7 Experimental edge | Not started; nothing above justifies it yet. |

## Tests run today (all walk-forward, scored on 2025, FFA weekly baseline)

**A. Direct residual GBM, base lags + context, then each spec family added** (n = 5,887
player-weeks). FFA alone: MAE 4.190, RMSE 5.872, weekly Spearman 0.714.

| family added | ΔMAE (k=1) | ΔRMSE | ΔSpearman |
|---|---|---|---|
| route share l1/r3/r6 | +0.00% | +0.00% | 0.000 |
| TPRR / YPRR r6 (shrunk) | +0.00% | +0.00% | 0.000 |
| usage surprise (route surprise, two-week flag) | −0.12% | −0.06% | 0.000 |
| red-zone route share | +0.00% | +0.00% | 0.000 |
| opponent man / pressure rate | +0.00% | +0.00% | 0.000 |
| player man-vs-zone TPRR gap × opp man rate | −0.04% | +0.03% | 0.000 |
| team dropbacks r6 | +0.00% | +0.00% | 0.000 |
| vacated routes (absent teammates' r3 route share) | +0.01% | −0.01% | 0.000 |
| all thirteen | +0.02% | +0.05% | −0.001 |

Univariate correlations with the FFA residual are all |r| < 0.07. Two mean gradients exist
(route surprise Q1 +0.04 → Q5 +0.66 pts; YPRR r6 Q1 +0.71 → Q5 −0.17, i.e. FFA over-rates hot
efficiency) but they are ~0.6 pts on a 4.2-pt MAE and neither GBM nor ridge can monetise them.

**B. Low-capacity ridge calibrator (spec sec. 30)** on RB/WR/TE with route surprise, YPRR,
TPRR, vacated routes, route share, position-interacted: RMSE −0.1% at best, MAE worse (a mean
shift hurts MAE on a skewed target), Spearman unchanged.

**C. Mechanistic receiving component (spec sec. 18):** rec yards = team dropbacks r6 × route
share r3 × YPRR (shrunk), 2025 RB/WR/TE, n ≈ 4,000.

| projection | MAE rec yds | Spearman |
|---|---|---|
| FFA rec_yds line | 16.07 | 0.639 |
| mechanistic (16-game efficiency) | 16.91 | 0.609 |
| 50/50 blend | 16.41 | 0.637 |

corr(mechanistic − FFA, actual − FFA) = +0.011. The mechanism knows nothing FFA has not
already priced. Same for receptions (1.29 vs 1.22 MAE).

**D. Injury designation → P(played), 2025 skill players, team priced by DK:**

| Questionable and… | played | n |
|---|---|---|
| DK posted a line for the player | 100% | 134 |
| no DK line, FFA proj ≥ 4 | 20% | 54 |
| no DK line, FFA proj ≥ 8 | 11% | 19 |
| Doubtful (any) | 0% | 29 |

For the no-line group, E[actual PPR] including zeros was 1.67 vs an FFA projection of 7.7
(ratio 0.22). The generator's `--p-play-doubt` default is now 0.20. Practice status
(DNP/Limited/Full) adds no separation once the DK-line split is known. Only 2025 is available
from nflverse injuries at present, so this is one season of evidence.

## Subscriptions

| Service | Price found | Recommendation |
|---|---|---|
| Fantasy Points Data Suite | $200/yr standalone ($160 early-bird), no monthly option seen | **Not now.** Its route/coverage/pressure/time-to-throw dimensions are what we just built free (historical) and tested null; the null result means in-season access to the same dimensions would not move the projection either. What is genuinely FP-only: run scheme and gap, route tree for non-targeted receivers, per-route separation grades, time-to-pressure and stunts, per-game play caller since 2021, lineup-combination tools. None of these address the failure mode every test shows: FFA + DK have already priced the mechanism. Revisit only if a joint-simulation build needs per-player route trees. |
| SūmerSports | $10/wk, $20/mo, $100/yr; no export or API | **Skip.** The spec itself limits it to manual research and QA. With a null result on the structured equivalents there is nothing to QA against. |
| FFA API (Premier) | separate product, price via sales@fantasyfootballanalytics.net | **Not needed.** The weekly R scrape already delivers raw stat lines; Premier would only automate it. Ask for a quote only if the Props automation track needs hands-off weekly pulls. |
| The Odds API | already have it | Keep. Historical props 2023-25 are in the DB; live snapshots run each generation. |
| Open-Meteo | free | Already used (wind term, wind-unders study). |

## What from the spec is worth building (free data, in priority order)

1. **Joint game simulation for correlated outputs** (spec 20, 43). Re-centre the existing
   per-player quantiles on a shared game-script draw (team pass volume, score state) so QB–WR1,
   QB–opposing-QB and RB-TD–QB-TD correlations emerge. Deliverables SaberSim will actually
   evaluate: correlation matrix per game, P(20+/25+/30+) per player, stack percentiles. This is
   the README's stated gap and distribution work, not mean-accuracy work, so the null-feature
   result does not argue against it.
2. **Automatic weekly report with miss classification** (spec 45) for the Week 3 review:
   accuracy by position, model vs FFA vs market vs blend, calibration at 10/15/20/25/30+,
   percentile coverage, and the 20 largest misses tagged injury / role / script / TD variance.
3. **Route-participation data as a lineup and role monitor, not a model feature**: the weekly
   `routes_*.parquet` gives route share and usage surprise per player; use it in the Tuesday
   waiver screen and the DOUBT/OUT handling (who actually ran routes last week), where the
   context-effects shares came from box scores only.
4. **Point-in-time discipline for backtests** (spec 7): the odds DB already stores snapshot
   times; add an `available_at` filter to the market blend backtest so the closing-line result
   is confirmed against Wednesday lines.

## What not to build

The ten-model mechanistic chain as the mean engine. Three separate studies (2026-09-09 feature
search, context effects, today's Tier-A set: ~30 features) all land within ±0.2% of the FFA
baseline, and the component test shows the mechanism itself trails FFA at the stat level.
The spec's guiding rule (sec. 53, question 4) removes every one of them.
