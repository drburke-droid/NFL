# Monte Carlo League Simulation v3 — 200 runs (2016-2025 × 200)

Same realistic league as v2 (H2H, FAAB waivers, props-driven lineups, equal in-season info). Each run re-randomizes the **draft bids** (lognormal σ=0.15) and the **schedule**, so results carry uncertainty. Only draft valuation differs between us and the bots.

## Our team — finish distribution
| Place | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % of seasons | 20 | 20 | 23 | 13 | 8 | 5 | 4 | 3 | 2 | 1 | 1 | 0 |

- **Avg finish 3.43**  (95% CI across runs 2.20–4.80)
- **Title rate 20.1%/yr**  → ≈ **2.0 titles per decade**
- **Playoffs (top-6) 89%** of seasons | Top-3 63%
- **P(≥1 title in a 10-yr span) = 91%** | P(≥2) = 70%
- Median titles per decade: 2  (range 0–5)

## Style leaderboard — avg finish ± 95% CI (lower = better)
- **US**                 3.43  [2.20, 4.80]
- rookie_lover           6.14  [4.10, 7.90]
- extreme_balanced       6.25  [4.30, 8.20]
- value_par              6.31  [4.00, 8.40]
- rb_heavy               6.58  [4.40, 8.70]
- mild_stars_scrubs      6.72  [4.50, 8.80]
- balanced               6.76  [5.00, 8.80]
- te_premium             6.87  [4.60, 8.80]
- extreme_stars_scrubs   7.09  [5.30, 8.81]
- wr_heavy               7.16  [5.20, 9.21]
- hero_rb                7.34  [5.30, 9.20]
- zero_rb                7.35  [5.40, 9.30]

## Read
- Over 200 randomized drafts/schedules, our draft-day process is the **best team in the league** (3.43 avg vs a field average of ~6.5), makes the playoffs ~89% of years, and wins ~2.0 titles/decade — but is **not a lock** (it finishes outside the top-3 in 37% of seasons).
- The spread (CI, finish histogram) is the honest picture: a real edge that compounds over many seasons, not a guarantee in any single one. Draft-bid noise + schedule luck are the dominant variance sources here.