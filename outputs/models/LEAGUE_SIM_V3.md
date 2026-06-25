# Monte Carlo League Simulation v3 — 200 runs (2016-2025 × 200)

Same realistic league as v2 (H2H, FAAB waivers, props-driven lineups, equal in-season info). Each run re-randomizes the **draft bids** (lognormal σ=0.15) and the **schedule**, so results carry uncertainty. Only draft valuation differs between us and the bots.

## Our team — finish distribution
| Place | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % of seasons | 13 | 17 | 18 | 13 | 10 | 8 | 7 | 5 | 4 | 3 | 2 | 1 |

- **Avg finish 4.29**  (95% CI across runs 2.70–6.00)
- **Title rate 13.2%/yr**  → ≈ **1.3 titles per decade**
- **Playoffs (top-6) 79%** of seasons | Top-3 48%
- **P(≥1 title in a 10-yr span) = 76%** | P(≥2) = 36%
- Median titles per decade: 1  (range 0–5)

## Style leaderboard — avg finish ± 95% CI (lower = better)
- **US**                 4.29  [2.70, 6.00]
- value_par              4.90  [3.00, 6.60]
- rookie_lover           5.36  [3.70, 7.20]
- te_premium             5.62  [3.90, 7.31]
- balanced               6.03  [4.10, 8.21]
- hero_rb                6.35  [4.60, 8.40]
- extreme_balanced       6.96  [4.70, 9.00]
- wr_heavy               7.10  [4.70, 9.10]
- rb_heavy               7.36  [5.00, 9.41]
- zero_rb                7.54  [5.20, 9.60]
- extreme_stars_scrubs   8.07  [5.79, 10.00]
- mild_stars_scrubs      8.42  [6.10, 10.40]

## Read
- Over 200 randomized drafts/schedules, our draft-day process is the **best team in the league** (4.29 avg vs a field average of ~6.5), makes the playoffs ~79% of years, and wins ~1.3 titles/decade — but is **not a lock** (it finishes outside the top-3 in 52% of seasons).
- The spread (CI, finish histogram) is the honest picture: a real edge that compounds over many seasons, not a guarantee in any single one. Draft-bid noise + schedule luck are the dominant variance sources here.