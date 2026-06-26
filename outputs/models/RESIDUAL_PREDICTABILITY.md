# Can any feature predict who beats/misses their projection? (2014-2025)

The verdict (above / to / below expectation) IS the projection's own error. So if any
feature predicts it, the projection is leaving signal on the table. `scripts/backtest_residual_features.py`
— walk-forward projections every year, then walk-forward predict the verdict from EVERY
draft-day feature we have (base + half-trend + opportunity + comps + injury-prior + risk).

## Two findings

### 1. The draftable pool systematically returns ~85% of projection
Across 1,648 relevant player-seasons: **48% BELOW, 29% to expectation, 22% ABOVE**, mean
actual/projection ratio ≈ **0.85** — roughly uniform across every feature tercile. This is a
**selection/regression + injury effect**: you draft the highest-projected players, who as a
group regress, get hurt, and lose games. It is NOT tied to any feature — it's a global level
bias, which is exactly what the risk-adjustment + top compression in the Draft Room already
correct (discount the top toward realistic clearing prices). (A small part is the proj_total =
central×16 games assumption; the relevant pool averages slightly fewer.)

### 2. WHO beats vs misses is NOT predictable
| | value |
|---|---|
| Verdict accuracy (walk-forward) | **0.423** |
| Majority-class baseline ("always BELOW") | 0.471 |
| AUC BELOW / to_exp / ABOVE (one-vs-rest) | 0.53 / 0.56 / 0.54 |

The model is **worse than guessing "BELOW" every time**, and every class AUC sits at ~chance.
Throwing all 40+ features — including everything we engineered this project — at the residual
yields no usable skill. The faint directional hints that do exist (higher floor / lower bust /
durability `gw_prior` → slightly better outcomes; ~0.05 ratio spreads) are already baked into
the risk-adjusted valuation.

## Does the risk lever surface "confident ABOVE"? No — it surfaces reliability.
The risk lever (`K_RISK`) adjusts the **valuation**, not the projection — so it cannot change
the predictability finding (the verdict is measured vs the central projection). But does
selecting low-risk players tilt the verdict mix toward booms? The opposite:

| Risk tier | ABOVE | to_exp | BELOW | mean ratio |
|---|---|---|---|---|
| low-risk | **19%** | **38%** | 43% | 0.88 |
| mid | 24% | 26% | 51% | 0.85 |
| high-risk | **24%** | 25% | 52% | 0.85 |

Low-risk players get MORE "to expectation" and FEWER "ABOVE" — the booms come
disproportionately from HIGH-risk players (both tails are fatter). So the lever is a
**variance dial, not a boom-finder**: turn it up for floor protection (fewer BELOW, more
to_exp, fewer ABOVE), down to chase upside (more ABOVE and more BELOW). It cannot isolate
*confident* ABOVE — that's inherently the volatile group.

And at the draftable top (top-30/yr), the verdict mix is ~identical whether you rank by raw
projection or risk-adjusted (23/36/41 vs 22/37/41) — the elite tier is the same players; the
lever changes the PRICE you pay within it, not who's in it.

## Why this is the right answer
The residual is dominated by **unforeseeable in-season injury and TD variance** — irreducible
luck. This is the consistent thread of the whole project: we already extract the predictable
signal (the projection), and the niche ideas (career comps, half-season trend, opportunity)
add nothing to the *residual*. The lever that remains is not "more edge features" but
**managing variance** — which is precisely why the risk-adjusted, scarcity-aware valuation
(certainty-equivalent + VONA) is where the real, durable improvement lives, not in chasing
another predictor of who will boom or bust.
