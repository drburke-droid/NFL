# Monte Carlo League Simulation v3 — 200 runs (2016-2025 × 200)

Same realistic league as v2 (H2H, FAAB waivers, props-driven lineups, equal in-season info). Each run re-randomizes the **draft bids** (lognormal σ=0.15) and the **schedule**, so results carry uncertainty. Only draft valuation differs between us and the bots.

## Our team — finish distribution
| Place | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| % of seasons | 20 | 19 | 20 | 14 | 8 | 5 | 5 | 4 | 2 | 2 | 1 | 0 |

- **Avg finish 3.60**  (95% CI across runs 2.30–5.20)
- **Title rate 20.1%/yr**  → ≈ **2.0 titles per decade**
- **Playoffs (top-6) 86%** of seasons | Top-3 60%
- **P(≥1 title in a 10-yr span) = 92%** | P(≥2) = 63%
- Median titles per decade: 2  (range 0–6)

## Style leaderboard — avg finish ± 95% CI (lower = better)
- **US**                 3.60  [2.30, 5.20]
- rookie_lover           6.12  [4.00, 8.10]
- value_par              6.15  [3.79, 8.20]
- extreme_balanced       6.22  [4.20, 8.40]
- rb_heavy               6.49  [4.30, 8.40]
- mild_stars_scrubs      6.80  [4.70, 8.80]
- te_premium             6.82  [4.80, 9.10]
- balanced               6.92  [5.00, 9.40]
- extreme_stars_scrubs   7.01  [4.80, 9.01]
- wr_heavy               7.17  [4.99, 9.10]
- hero_rb                7.27  [5.50, 9.10]
- zero_rb                7.43  [5.20, 9.30]

## Read
- Over 200 randomized drafts/schedules, our draft-day process is the **best team in the league** (3.60 avg vs a field average of ~6.5), makes the playoffs ~86% of years, and wins ~2.0 titles/decade — but is **not a lock** (it finishes outside the top-3 in 40% of seasons).
- The spread (CI, finish histogram) is the honest picture: a real edge that compounds over many seasons, not a guarantee in any single one. Draft-bid noise + schedule luck are the dominant variance sources here.