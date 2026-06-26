# Risk-adjusted auction value (certainty-equivalent) — Draft Room

Replaces the ad-hoc top-end compression with a derived, data-driven valuation. Two
clearly-separated pieces:

## 1. Shape — principled (certainty-equivalent from measured risk)
Each player's points are discounted by a **risk score** built from measured quantities,
then valued (a steady high-floor producer is worth more than a boom/bust, injury-prone
one at the same projection):

```
risk = 0.45·injury(pos) + 0.40·bust + 0.30·downside        (clipped to 0.9)
raPts = proj_pts · (1 − 0.70·risk)                          (certainty-equivalent points)
```
- **injury(pos)** — normalized from the % of startable-tier seasons missing significant
  time (QB 26% → 0, RB 40% → 1; from `qb_reliability.py`). RBs carry the most injury risk.
- **bust** — the player's calibrated P(below startable next year) (from `projection_overhaul`).
- **downside** — floor gap `(proj_ppg − floor)/proj_ppg` (P15 shortfall; from the quantile model).

Auction value is then VONA (value over the marginal startable player) computed on `raPts`,
so prices reflect risk-adjusted edge, not raw points.

## 2. Level — empirical (market anchor)
A full-rational, budget-constrained, second-price auction *clears the top ~$95* (measured
from the league simulator). Real markets top ~$70–73 because of risk aversion to
concentration + the option value of holding cash — a real effect whose *magnitude* is
behavioral, so it's anchored to observed prices (one knob), not derived. The top of the
curve is compressed to match (~$70 ceiling).

## Effect (2026 board, fresh draft)
| | raw VORP | risk-adjusted |
|---|---|---|
| #1 | Gibbs/Bijan (RB) $95 | **Amon-Ra St. Brown (WR) $71** |
| Gibbs / Bijan (RB, risk .60) | $95 | $69 / $70 |
| Josh Allen (QB, risk .11 — safest) | ~$43 | **$56** |
| Derrick Henry / Jacobs (RB, risk .60) | ~$50 | $44–47 |

Safe high-floor producers (steady WRs, durable QBs) rise; volatile, injury-prone RBs
fall — the order now encodes the reliability/fade-risk data we already measured, and the
top lands at a realistic clearing price. `K_RISK` (0.70) and the compression slope are the
only two calibrations; everything else is from the data. Implemented in `dynamicMarket`
(docs/index.html).
