# Realistic Keeper League Simulation v2 (2016-2025) — H2H, waivers, props-driven lineups

Every team uses the SAME weekly projection (real player props 2023-25, else form×matchup) for lineups & FAAB waivers — equal in-season info. Only the DRAFT differs: bots use ADP/prior-year × style; we use projections + VORP + leap/fade. 14-wk H2H, top-6 playoffs, FAAB $100, K/DST streamed (flat-equal).

## Finish by year (1 = champion)

| Owner (style) | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Avg | Titles | Top-3 | Playoffs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **US** | 4 | 2 | 3 | 3 | 6 | 3 | 1 | 3 | 1 | 8 | **3.4** | 2 | 7 | 9 |
| balanced | 7 | 3 | 12 | 4 | 10 | 2 | 5 | 1 | 7 | 1 | **5.2** | 2 | 4 | 6 |
| te_premium | 9 | 4 | 1 | 1 | 3 | 4 | 6 | 7 | 6 | 11 | **5.2** | 2 | 3 | 7 |
| value_par | 5 | 5 | 9 | 7 | 2 | 8 | 11 | 6 | 2 | 3 | **5.8** | 0 | 3 | 6 |
| mild_stars_scrubs | 6 | 10 | 2 | 10 | 9 | 1 | 4 | 5 | 8 | 4 | **5.9** | 1 | 2 | 6 |
| rb_heavy | 1 | 1 | 11 | 5 | 5 | 10 | 3 | 4 | 11 | 9 | **6.0** | 2 | 3 | 6 |
| extreme_balanced | 3 | 9 | 10 | 9 | 7 | 7 | 2 | 9 | 5 | 5 | **6.6** | 0 | 2 | 4 |
| wr_heavy | 11 | 7 | 8 | 8 | 1 | 11 | 7 | 2 | 12 | 2 | **6.9** | 1 | 3 | 3 |
| rookie_lover | 2 | 12 | 5 | 11 | 4 | 9 | 8 | 8 | 4 | 10 | **7.3** | 0 | 1 | 4 |
| extreme_stars_scrubs | 10 | 6 | 7 | 2 | 8 | 6 | 12 | 10 | 10 | 7 | **7.8** | 0 | 1 | 3 |
| hero_rb | 8 | 11 | 4 | 6 | 11 | 12 | 9 | 12 | 9 | 6 | **8.8** | 0 | 0 | 3 |
| zero_rb | 12 | 8 | 6 | 12 | 12 | 5 | 10 | 11 | 3 | 12 | **9.1** | 0 | 1 | 3 |

## Our team
- Avg finish **3.4** | Titles **2** | Top-3 **7/10** | Playoffs **9/10**
- Avg regular-season wins: **9.3** of 14 | finishes [4, 2, 3, 3, 6, 3, 1, 3, 1, 8]

## Style leaderboard (avg finish, lower=better)
- **US**                 3.40
- balanced               5.20
- te_premium             5.20
- value_par              5.80
- mild_stars_scrubs      5.90
- rb_heavy               6.00
- extreme_balanced       6.60
- wr_heavy               6.90
- rookie_lover           7.30
- extreme_stars_scrubs   7.80
- hero_rb                8.80
- zero_rb                9.10

## Notes
- In-season skill is equalized (identical projection + waiver logic for all 12), so finishing differences trace to **draft-day roster construction** — our team's only edge.
- Weekly props (2023-25) give true market projections; pre-2023 uses leakage-free recent-form × game-environment. Lineups set by projection, scored by actual (realistic start/sit error).
- K/DST streamed as an equal flat contribution (near-random, everyone streams). H2H wins, top-6 playoffs, FAAB $100 blind weekly bids.
- Single deterministic realization (fixed schedule, no bid RNG). Verdict: with in-season skill equalized, the draft edge is real but **beatable** — our avg finish ~3.4 vs a field average of ~6, with down years. A Monte Carlo (noisy schedules/bids, N runs) would turn this into a finish distribution / title-odds.