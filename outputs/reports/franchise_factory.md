# Franchise fantasy factories, 1999–2025

**Question** (2026-09-01): do some franchises/regimes produce a statistically real
surplus (or deficit) of fantasy-relevant skill players — 2000s-Patriots-style — and
others a desert?

**Data**: nflverse season PPR 1999–2024 + local `nflv_season` 2025 (27 seasons).
**Startable** = QB top-12, RB top-24, WR top-36, TE top-12 per season (~84 slots/yr,
the repo's dart-hit convention). Relocations merged (OAK→LV, SD→LAC, STL→LA).
Expected per franchise ≈ 71 startable player-seasons over the era.

## Headline: franchise effects are REAL, but diffuse

| Test | Result |
|---|---|
| Global dispersion (chi², df=31) | **70.6, p=0.0001** vs multinomial null |
| Same vs a *stacking-preserving* null (permute team labels within season, so "one elite QB creates 3–4 same-team slots" is baked into the null) | null 95th pctl = 29.1 → **p < 0.0001** |
| Persistence (count_t → count_t+1) | **ρ = 0.31**, label-shuffle null 95th = 0.06, p < 0.0001 |
| Predictive form | next-yr count = 1.34 + 0.16·last_yr + 0.33·trail-5yr (both p ≤ 0.0003) |

The league is **not** a uniform lottery — but the effect is a broad ±30% tilt spread
across many teams, not a few mythical super-factories. No single franchise clears
32-way BH-FDR at q<0.05 (Cleveland comes closest, q=0.062).

## The factories and the deserts (27-yr totals vs exp 71)

**Surplus**: PHI 94 (1.32×), GB 93, DAL 92, IND 90, NO 88, NE 85.
**Deficit**: CLE 46 (0.65×, q=0.06), NYJ 49 (0.69×), HOU 49/63 exp, CHI 58, WAS 60.

## Era/regime runs (5-yr windows)

Family-wise permutation bar for "hottest run anywhere in 27 years" = 28; no run
reaches it, so no *individual* era is significant after selection correction — they
are the tails of the (globally real) franchise distribution:

- **Hottest**: Saints 2011–15 (25) and the whole Payton–Brees 2008–17 plateau;
  Colts 1999–2010 (Manning, 22–23 per window); Patriots 2008–14 (Brady–Moss/Welker/
  Gronk, 22–23). The "2000 Patriots" pre-Moss era was ordinary — NE's factory years
  were 2007+.
- **Coldest**: NYJ 2018–22 (**3** startable in 5 years), CLE 2008–15 (4–5), NYG
  2017–24 (4–6), the expansion-era Browns, and — surprising — DAL 2000–04 (6,
  post-Aikman/Irvin, pre-Romo).
- HC regimes 2011–25: none FDR-significant (min q=0.51). Sean Payton 1.45× over 10
  yrs (raw p=.036) is the best documented; Dan Campbell DET and Nick Sirianni PHI
  both sit at 1.60× (raw p=.055) among active coaches.

## Lens B: draft-and-develop (classes 1999–2020, round-adjusted)

Crediting startable seasons to the *drafting* franchise, vs expectation given the
rounds of their picks: **LAC 1.96×** (raw p=.009) and **DAL 1.94×** at the top;
**CLE 0.44×** (raw p=.009) at the bottom — Cleveland is last in BOTH lenses, the one
coherent "desert" story in the data. Neither extreme survives FDR (q≈0.15) —
suggestive, not proven.

## What it means for 2026

Persistence says ~⅓ of a franchise's trailing-5-yr surplus carries forward. Trailing
2021–25 counts (exp 13.1): **DET 21, PHI 21, DAL 20, TB 19, BUF/CIN 18** at the top;
**NYG 6, CAR 6, NYJ 7, HOU 8, NO/IND/TEN 9** at the bottom. Mild, environment-level
signal — mostly proxying QB/offense continuity, which FFA projections already price.
Use as a tiebreaker lens on darts (a $1 WR in DET/PHI-class infrastructure vs NYG-
class), not as a standalone edge.

**Caveats**: startable counts are offense-quality proxies, not causal "development
skill"; persistence is substantially the same players staying put; 1999 data floor
means pre-1999 dynasties are out of scope.

Study: `scripts/franchise_factory_study.py` (+ `global tests` inline in the report
numbers); panel CSV regenerable from the script.
