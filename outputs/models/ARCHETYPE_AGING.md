# Do player archetypes age differently / project differently?

Archetypes from pre-known traits (combine size/speed + prior usage), leakage-free.
`scripts/test_archetypes.py`. Counts (player-seasons 2012-2025) all healthy except
WR_deep_threat (83) and QB_mobile (81) — read those as tentative.

## (1) Do they age differently? — YES for a few (mean YoY ΔPPG by age, established players)

Negative = declining. Highlights:

- **QB: mobile backs decline much faster than pocket passers.** Mobile QBs:
  −0.8 → −1.1 → −1.6 → −2.7 → **−3.7** (31+). Pocket QBs at 31+ only **−1.7**.
  Mechanistic: rushing production is age-fragile; the legs go first.
- **WR: deep threats hold value young, then crater.** ≤24 −0.3, 25-26 **−0.2**
  (basically flat through 26), then 27-28 −2.2, 31+ −3.3. Speed-dependent profile —
  fine until the speed goes. (Thin sample, tentative.)
- **RB: receiving backs cliff hardest at 29-30 (−3.7).** Despite the "PPR backs last
  longer" lore, in this league's scoring they fall off sharply at 29-30; workhorses
  decline more steadily (−2.5 at 27-28, −3.3 at 31+).
- **TE: receiving TEs age gracefully late** (31+ only −1.1 vs the −2 to −3 typical of
  WR/RB) — the position's well-known long shelf life.

## (2) Does projecting BY archetype improve accuracy? — NO (meaningfully)

BASE vs +combine traits vs +archetype dummies & archetype×age interactions, walk-forward:

| Central PPG MAE | BASE | +combine | +arch |
|---|---|---|---|
| RB | 2.937 | 2.947 | 2.949 |
| WR | 2.561 | 2.557 | 2.550 |
| TE | 1.923 | 1.914 | 1.917 |
| QB | 4.001 | 3.982 | 3.976 |

Bust/boom AUC: flat except TE (bust +0.010, boom +0.008) and RB boom (+0.006) — small,
and TE AUC is noisy, so likely not real.

**Why:** the model already ingests the traits that DEFINE the archetypes — weight,
forty, target/air-yards share, carries/receptions, and age. Trees recover the
interactions, so explicit archetype labels are redundant for the projection numbers.
The mobile-QB aging effect is real descriptively but the model's `age` + `prior_carries_pg`
already capture most of it (QB MAE barely moved).

## Decision
Keep the projections as-is — no archetype features (no validated lift; same bar that
kept comps & half-trend out of the numbers). The aging curves are valuable as
QUALITATIVE context, not inputs: notably treat aging **mobile QBs (29+)**, **deep-threat
WRs (28+)**, and **receiving RBs (29+)** as elevated decline risks when two candidates
are otherwise close.
