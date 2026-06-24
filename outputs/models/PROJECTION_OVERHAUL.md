# Projection System Overhaul — position-by-position, with realistic boom/bust

Rebuilt the projection system per position so every player gets **one** central
projection **plus a learned distribution** (floor/ceiling) and **calibrated,
realistic boom/bust probabilities** — all validated walk-forward (train past
seasons, predict each held-out season, 2016–2025).

## What was wrong before
- **Inconsistent projections** across tabs (board model vs comp-blend).
- **Unrealistic 0% bust** — the comp-cohort bust used "best of next 3 years," which
  elite players essentially never fail, so it returned 0%.

## What the validation showed

**1. Central projection (per-position quantile P50) beats "repeat last year":**

| Pos | Model MAE | Repeat-LY MAE | ρ |
|---|---|---|---|
| QB | 3.88 | 4.22 | 0.69 |
| RB | 3.02 | 3.29 | 0.71 |
| WR | 2.56 | 2.83 | 0.78 |
| TE | 1.92 | 2.08 | 0.72 |

**2. Boom/bust must be ABSOLUTE, not relative.** Predicting whether a player beats
his *own projection* by ±30% is **near-random (AUC ~0.52)** — once you've made a good
projection, the residual is noise. But **absolute outcomes are very predictable**:

| Outcome (single next season) | AUC | Calibration (realized by predicted quintile) |
|---|---|---|
| **Bust** (below startable: QB<14/RB<10/WR<9/TE<7) | **0.82** | 22% → 51% → 86% → 90% → 90% |
| **Boom** (elite: QB≥21/RB≥16/WR≥15/TE≥12) | **0.80** | 3% → 6% → 3% → 2% → 31% |

So "bust%/boom%" now = **P(below-startable season)** and **P(elite season)** — modeled per
position with **Platt-calibrated** classifiers. Realistic and never 0%/100%:
- Josh Allen: 17% bust / 26% boom · Mahomes 22% / 13% · **Alvin Kamara (aging RB) 53% / 6%** ·
  Saquon 33% / 7% · top WRs ~10% bust / ~80% boom.

**3. Career comps add no value to the *numbers*.** Adding comp-cohort features to the
projection/boom/bust models changed AUC by ≤0.003 — the base production/role/age/draft
features already subsume them. So **comps stay a qualitative analog tool** (what arc a player
is tracking); the projection and boom/bust come from the validated calibrated models. *(The
earlier standalone "comp bust AUC 0.76" was real but redundant given the base features.)*

## What ships
- **One projection everywhere**: per-player **central / floor (P15) / ceiling (P85) /
  bust% / boom%**, position-specific, in `proj_overhaul`.
- All tabs (Cheat Sheet, Auction, Targets, Comps) read the **same** central + bust/boom.
- The Comps tab now shows the model's calibrated bust/boom (not the old cohort rates), and
  the same projection as the board — resolving the inconsistency.

*Scripts: `generate_comp_features.py`, `projection_overhaul.py` → `proj_overhaul`.*
