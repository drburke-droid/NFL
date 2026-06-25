# Realistic Keeper League Simulation v2 (2016-2025) — H2H, waivers, props-driven lineups

Every team uses the SAME weekly projection (real player props 2023-25, else form×matchup) for lineups & FAAB waivers — equal in-season info. Only the DRAFT differs: bots use ADP/prior-year × style; we use projections + VORP + leap/fade. 14-wk H2H, top-6 playoffs, FAAB $100, K/DST streamed (flat-equal).

## Finish by year (1 = champion)

| Owner (style) | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Avg | Titles | Top-3 | Playoffs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **US** | 4 | 2 | 3 | 4 | 8 | 3 | 2 | 2 | 1 | 12 | **4.1** | 1 | 6 | 8 |
| rb_heavy | 3 | 1 | 1 | 8 | 3 | 12 | 3 | 10 | 8 | 2 | **5.1** | 2 | 6 | 6 |
| value_par | 5 | 4 | 8 | 6 | 1 | 2 | 10 | 1 | 11 | 8 | **5.6** | 2 | 3 | 6 |
| mild_stars_scrubs | 6 | 9 | 2 | 9 | 2 | 5 | 7 | 9 | 6 | 3 | **5.8** | 0 | 3 | 6 |
| extreme_balanced | 2 | 7 | 9 | 7 | 4 | 7 | 8 | 4 | 3 | 10 | **6.1** | 0 | 2 | 4 |
| rookie_lover | 1 | 12 | 5 | 10 | 7 | 4 | 1 | 6 | 9 | 7 | **6.2** | 2 | 2 | 5 |
| te_premium | 8 | 6 | 6 | 1 | 11 | 11 | 9 | 5 | 4 | 9 | **7.0** | 1 | 1 | 5 |
| balanced | 7 | 3 | 12 | 3 | 12 | 8 | 11 | 7 | 2 | 6 | **7.1** | 0 | 3 | 4 |
| zero_rb | 11 | 8 | 11 | 12 | 6 | 1 | 6 | 3 | 10 | 5 | **7.3** | 1 | 2 | 5 |
| wr_heavy | 12 | 11 | 7 | 5 | 9 | 6 | 5 | 11 | 5 | 4 | **7.5** | 0 | 0 | 5 |
| extreme_stars_scrubs | 10 | 5 | 10 | 2 | 5 | 10 | 12 | 8 | 7 | 11 | **8.0** | 0 | 1 | 3 |
| hero_rb | 9 | 10 | 4 | 11 | 10 | 9 | 4 | 12 | 12 | 1 | **8.2** | 1 | 1 | 3 |

## Our team
- Avg finish **4.1** | Titles **1** | Top-3 **6/10** | Playoffs **8/10**
- Avg regular-season wins: **9.3** of 14 | finishes [4, 2, 3, 4, 8, 3, 2, 2, 1, 12]

## Style leaderboard (avg finish, lower=better)
- **US**                 4.10
- rb_heavy               5.10
- value_par              5.60
- mild_stars_scrubs      5.80
- extreme_balanced       6.10
- rookie_lover           6.20
- te_premium             7.00
- balanced               7.10
- zero_rb                7.30
- wr_heavy               7.50
- extreme_stars_scrubs   8.00
- hero_rb                8.20

## Notes
- In-season skill is equalized (identical projection + waiver logic for all 12), so finishing differences trace to **draft-day roster construction** — our team's only edge.
- Weekly props (2023-25) give true market projections; pre-2023 uses leakage-free recent-form × game-environment. Lineups set by projection, scored by actual (realistic start/sit error).
- K/DST streamed as an equal flat contribution (near-random, everyone streams). H2H wins, top-6 playoffs, FAAB $100 blind weekly bids.
- Single deterministic realization; league_sim_v3.py Monte-Carlos the draft bids + schedule for finish distributions / title odds.