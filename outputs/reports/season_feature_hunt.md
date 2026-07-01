# Season-model feature hunt & residual autopsy (2026-07-01)

Reproducible: `scripts/season_feature_hunt.py` (feature blocks / autopsy / blend sweep over the
`model_season.py` walk-forward) + `scripts/season_bias_correction_test.py` (validation over the
PRODUCTION scorer `projection_overhaul.py`). Goal: novel MAE improvement or usable trends.

## 1. Untested feature blocks — all ~zero (the model is efficient)
Walk-forward 2016–2025 over the actual `MODEL_FEATURES` (baseline MAE **2.5192**, rho .781):

| block added | ΔMAE | verdict |
|---|--:|---|
| + half-season trend (`nflv_half_trend`) | +0.0040 | noise — H2 surge already implied by role features |
| + injury priors (already in production scorer) | −0.0029 | ~0 here; keep in production (validated separately) |
| + career comps (`nflv_comp_features`) | −0.0111 | hurts — display-only, keep out |
| + opportunity (vac/inc usage) | −0.0044 | hurts — vacated-role stays a post-hoc bump |
| + all | −0.0090 | hurts |

## 2. Residual autopsy → REAL bias cluster → **APPLIED**
Segmenting walk-forward residuals (actual − pred) surfaced one consistent cluster — the model
**over-projects "aging/moving stars"**:

| segment | bias (production scorer) | n |
|---|--:|--:|
| age ≥ 30 | **−0.52 PPG** | 729 |
| team change | **−0.45 PPG** | 1131 |
| prior PPG ≥ 14 | **−0.24 PPG** | 681 |

Walk-forward correction (offsets learned only from prior seasons) over the production quantile scorer:
**MAE 2.5056 → 2.4804 (+0.025, ≈1%), positive in 7/7 test years (2019–2025)** — the same consistency
signature walk-year had. **Applied** in `projection_overhaul.py`: fixed offsets −0.52 / −0.45 / −0.24
shift floor/central/ceiling for matching 2026 players. Offsets stack (a 30-yo star changing teams gets
≈ −1.2 PPG). Mechanism: LightGBM under-applies mean reversion at distribution edges; consistent with
the age-cliff and "our totals run high on stars" observations.

**Production bug fixed en route:** the 2026 scorer hardcoded `team_change=0` for every player even
though `nflv_rosters_2026` is loaded in the DB. Now derives the real flag (100 movers flagged) — it's
both a model feature and a correction segment.

## 3. Blend-alpha sweep (model vs FFA, per-position rescale like the board)
On FFA-covered rows (n=1839): model-alone 2.844, FFA-alone 2.763, best blend ≈ **0.3 model / 0.7 FFA**
(2.748). BUT the curve is flat 0.3–0.7 (50/50 = 2.757, only +0.008 from optimum, and WR is best at
50/50). **Board default BLEND_W=0.5 stays** — within noise of optimal; nudging toward FFA helps QB/RB
marginally if desired via the slider.

## Chain rebuilt
`projection_overhaul` → `board_2026` → `docs/data.js` → `keepers_2026.js` all regenerated with the
corrected projections (age-30+/mover/star values shift down; keeper decisions re-ranked).
