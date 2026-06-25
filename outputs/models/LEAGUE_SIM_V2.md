# Realistic Keeper League Simulation v2 (2016-2025) — H2H, waivers, props-driven lineups

Every team uses the SAME weekly projection (real player props 2023-25, else form×matchup) for lineups & FAAB waivers — equal in-season info. Only the DRAFT differs: bots use ADP/prior-year × style; we use projections + VORP + leap/fade. 14-wk H2H, top-6 playoffs, FAAB $100, K/DST streamed (flat-equal).

## Finish by year (1 = champion)

| Owner (style) | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Avg | Titles | Top-3 | Playoffs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **US** | 3 | 7 | 5 | 1 | 2 | 5 | 6 | 9 | 3 | 2 | **4.3** | 1 | 5 | 8 |
| value_par | 5 | 1 | 4 | 5 | 11 | 6 | 3 | 5 | 7 | 1 | **4.8** | 2 | 3 | 8 |
| rb_heavy | 1 | 2 | 3 | 7 | 5 | 12 | 8 | 8 | 2 | 3 | **5.1** | 1 | 5 | 6 |
| wr_heavy | 11 | 4 | 1 | 8 | 1 | 8 | 10 | 1 | 4 | 6 | **5.4** | 3 | 3 | 6 |
| extreme_balanced | 2 | 8 | 8 | 2 | 9 | 2 | 7 | 3 | 12 | 5 | **5.8** | 0 | 4 | 5 |
| extreme_stars_scrubs | 7 | 9 | 2 | 6 | 3 | 11 | 4 | 10 | 5 | 7 | **6.4** | 0 | 2 | 5 |
| rookie_lover | 9 | 3 | 9 | 4 | 8 | 1 | 12 | 7 | 1 | 12 | **6.6** | 2 | 3 | 4 |
| mild_stars_scrubs | 6 | 10 | 11 | 10 | 7 | 3 | 5 | 2 | 9 | 9 | **7.2** | 0 | 2 | 4 |
| te_premium | 10 | 5 | 6 | 11 | 12 | 10 | 1 | 4 | 6 | 8 | **7.3** | 1 | 1 | 5 |
| balanced | 8 | 11 | 12 | 3 | 6 | 4 | 9 | 6 | 8 | 11 | **7.8** | 0 | 1 | 4 |
| hero_rb | 4 | 12 | 7 | 9 | 4 | 9 | 11 | 11 | 11 | 4 | **8.2** | 0 | 0 | 3 |
| zero_rb | 12 | 6 | 10 | 12 | 10 | 7 | 2 | 12 | 10 | 10 | **9.1** | 0 | 1 | 2 |

## Our team
- Avg finish **4.3** | Titles **1** | Top-3 **5/10** | Playoffs **8/10**
- Avg regular-season wins: **8.8** of 14 | finishes [3, 7, 5, 1, 2, 5, 6, 9, 3, 2]

## Style leaderboard (avg finish, lower=better)
- **US**                 4.30
- value_par              4.80
- rb_heavy               5.10
- wr_heavy               5.40
- extreme_balanced       5.80
- extreme_stars_scrubs   6.40
- rookie_lover           6.60
- mild_stars_scrubs      7.20
- te_premium             7.30
- balanced               7.80
- hero_rb                8.20
- zero_rb                9.10

## Notes
- In-season skill is equalized (identical projection + waiver logic for all 12), so finishing differences trace to **draft-day roster construction** — our team's only edge.
- Weekly props (2023-25) give true market projections; pre-2023 uses leakage-free recent-form × game-environment. Lineups set by projection, scored by actual (realistic start/sit error).
- K/DST streamed as an equal flat contribution (near-random, everyone streams). H2H wins, top-6 playoffs, FAAB $100 blind weekly bids.
- Single deterministic realization; league_sim_v3.py Monte-Carlos the draft bids + schedule for finish distributions / title odds.