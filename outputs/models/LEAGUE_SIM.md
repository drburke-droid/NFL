# 10-Year Keeper-Auction League Simulation (2016-2025)

12 owners, $200 auction, weekly optimal lineups. Bots draft from real ADP (2021-25) / prior-year production; OUR team (us) uses walk-forward projections + VORP + value-leap/fade tilt. Keepers: ≤3, finish-based inflation, 3-yr cap.

## Finish by year (1 = champion)

| Owner (style) | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Avg | Titles | Top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **US** | 4 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 2 | **1.6** | 6 | 9 |
| balanced | 1 | 9 | 10 | 2 | 5 | 3 | 4 | 6 | 9 | 3 | **5.2** | 1 | 4 |
| extreme_balanced | 9 | 1 | 5 | 4 | 8 | 5 | 8 | 2 | 5 | 6 | **5.3** | 1 | 2 |
| rookie_lover | 5 | 7 | 9 | 3 | 4 | 1 | 10 | 4 | 7 | 7 | **5.7** | 1 | 2 |
| value_par | 2 | 4 | 6 | 10 | 7 | 8 | 9 | 7 | 6 | 5 | **6.4** | 0 | 1 |
| rb_heavy | 3 | 10 | 8 | 5 | 9 | 11 | 6 | 8 | 8 | 4 | **7.2** | 0 | 1 |
| wr_heavy | 11 | 8 | 7 | 11 | 3 | 4 | 3 | 12 | 2 | 11 | **7.2** | 0 | 3 |
| mild_stars_scrubs | 6 | 11 | 2 | 6 | 11 | 6 | 7 | 11 | 3 | 10 | **7.3** | 0 | 2 |
| te_premium | 8 | 5 | 11 | 7 | 10 | 9 | 2 | 9 | 4 | 9 | **7.4** | 0 | 1 |
| zero_rb | 12 | 3 | 3 | 8 | 2 | 7 | 5 | 10 | 12 | 12 | **7.4** | 0 | 3 |
| hero_rb | 10 | 6 | 12 | 9 | 6 | 10 | 11 | 3 | 11 | 1 | **7.9** | 1 | 2 |
| extreme_stars_scrubs | 7 | 12 | 4 | 12 | 12 | 12 | 12 | 5 | 10 | 8 | **9.4** | 0 | 0 |

## Our team
- Average finish: **1.6** of 12
- Titles: **6** | Top-3: **9/10** | Playoffs(top-6): **10/10**
- Finishes by year: [4, 2, 1, 1, 1, 2, 1, 1, 1, 2]

## Style leaderboard (avg finish, lower better)
- **US**                 1.60
- balanced               5.20
- extreme_balanced       5.30
- rookie_lover           5.70
- value_par              6.40
- rb_heavy               7.20
- wr_heavy               7.20
- mild_stars_scrubs      7.30
- te_premium             7.40
- zero_rb                7.40
- hero_rb                7.90
- extreme_stars_scrubs   9.40

## Takeaways
- A disciplined **projection + optimal-VORP** process beats a field that drafts only on last-year/ADP rankings — by a wide, consistent margin. Information edge compounds over a full roster.
- Among the naive styles, **balanced / extreme-balanced** build the deepest rosters and finish best; **extreme stars-and-scrubs** is reliably worst (the $1 scrubs tank weekly lineups). Mild positional tilts (RB-heavy, zero-RB, WR-heavy) land mid-pack.

## Assumptions & caveats (this is an idealized upper bound)
- **Bots have no forward projections** — only real ADP (2021-25) or prior-year production, with a fixed style multiplier and a single-pass auction. Real opponents use projections and adapt mid-draft, so our real-world edge is smaller.
- **Deterministic**: no bid noise, no overpays, our team always pays the second-price. One realization, not a distribution.
- Standings = **total points (best-ball)**, no head-to-head schedule luck. Weekly optimal lineup on actual results; K/DST added as season totals (1 slot each).
- Pool = players who actually played that year (no busts-who-vanished drafted); no in-season waivers/trades/injury management.
- Keepers use this-year production as the next-year value proxy with the finish-based inflation and 3-year cap, as specified.
- **Robust takeaway = the style ranking and the direction (process > naive field)**; treat the magnitude of our dominance as optimistic.