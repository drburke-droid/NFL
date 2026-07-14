# Skill trajectories: what is skill, what is noise, what injuries cost

Data: nflv_pbp_skill 2011-2025, 2,255 players, real injury reports 2011.0-2025.0.

## 1. Which advanced metrics are actually skill?

YoY r = correlation season T vs T+1 (same player, qualified both years).
Split-half r = first-half vs second-half of the SAME season.
High both = skill. High split-half only = role/context. Low both = noise.

| Pos | Metric | YoY r | Split-half r | n (YoY) | Verdict |
|---|---|---|---|---|---|
| QB | epa_play | 0.43 | 0.50 | 403 | role/context |
| QB | cpoe | 0.42 | 0.38 | 403 | role/context |
| QB | success | 0.46 | 0.50 | 403 | **skill** |
| QB | adot | 0.48 | 0.51 | 403 | **skill** |
| QB | sack_rate | 0.48 | 0.44 | 403 | **skill** |
| QB | comp_pct | 0.54 | 0.49 | 403 | **skill** |
| QB | td_rate | 0.31 | 0.36 | 403 | role/context |
| QB | int_rate | 0.24 | 0.17 | 403 | noise |
| RB | epa_play | 0.17 | 0.18 | 464 | noise |
| RB | success | 0.26 | 0.27 | 464 | noise |
| RB | ypc | 0.21 | 0.24 | 464 | noise |
| RB | td_rate | 0.15 | 0.19 | 464 | noise |
| WR | epa_play | 0.22 | 0.16 | 974 | noise |
| WR | success | 0.30 | 0.21 | 974 | noise |
| WR | yacoe | 0.25 | 0.18 | 974 | noise |
| WR | adot | 0.69 | 0.62 | 974 | **skill** |
| WR | catch_rate | 0.46 | 0.35 | 974 | **skill** |
| WR | ypt | 0.40 | 0.31 | 974 | part-skill |
| TE | epa_play | 0.28 | 0.16 | 315 | noise |
| TE | success | 0.37 | 0.19 | 315 | part-skill |
| TE | yacoe | 0.31 | 0.24 | 315 | part-skill |
| TE | adot | 0.61 | 0.53 | 315 | **skill** |
| TE | catch_rate | 0.38 | 0.26 | 315 | part-skill |
| TE | ypt | 0.38 | 0.32 | 315 | part-skill |

## 2. Skill ages differently than volume

Mean YoY change in era-normalized PER-PLAY skill (composite z) vs change in
PER-GAME volume (plays/game, z), by age bucket. Negative = decline.

| Pos | Age | n | Δ skill z | Δ volume z | Reading |
|---|---|---|---|---|---|
| QB | ≤23 | 27 | +0.03 | +0.04 | in step |
| QB | 24-25 | 67 | -0.14 | -0.08 | skill fades first |
| QB | 26-27 | 62 | -0.01 | +0.11 | skill fades first |
| QB | 28-29 | 51 | -0.17 | -0.02 | skill fades first |
| QB | 30-32 | 65 | -0.18 | -0.01 | skill fades first |
| QB | 33+ | 90 | -0.21 | -0.15 | skill fades first |
| RB | ≤23 | 97 | -0.13 | -0.11 | in step |
| RB | 24-25 | 192 | -0.19 | -0.04 | skill fades first |
| RB | 26-27 | 137 | -0.25 | -0.26 | in step |
| RB | 28-29 | 72 | -0.23 | -0.22 | in step |
| RB | 30-32 | 35 | -0.39 | -0.32 | skill fades first |
| WR | ≤23 | 139 | -0.03 | +0.05 | skill fades first |
| WR | 24-25 | 295 | -0.09 | +0.04 | skill fades first |
| WR | 26-27 | 267 | -0.19 | -0.08 | skill fades first |
| WR | 28-29 | 171 | -0.09 | -0.21 | role fades first |
| WR | 30-32 | 107 | -0.13 | -0.31 | role fades first |
| WR | 33+ | 31 | -0.40 | -0.58 | role fades first |
| TE | ≤23 | 33 | +0.02 | +0.02 | in step |
| TE | 24-25 | 130 | +0.00 | +0.09 | skill fades first |
| TE | 26-27 | 121 | -0.22 | -0.07 | skill fades first |
| TE | 28-29 | 94 | +0.05 | -0.12 | role fades first |
| TE | 30-32 | 57 | -0.13 | -0.20 | role fades first |

## 3. What injuries cost (real injury reports, not games-played proxy)

Per-play EPA in games played while LISTED (Q/D) or in the 2 games after
returning from a 2+ week absence, vs each player's own clean-week baseline
(same season, play-weighted, min 4 clean games):

| Pos | State | n player-wks | EPA/play vs own baseline | 
|---|---|---|---|
| QB | listed Q/D | 145 | -0.061 |
| QB | return window | 382 | +0.023 |
| RB | listed Q/D | 601 | -0.042 |
| RB | return window | 1449 | +0.026 |
| WR | listed Q/D | 1213 | -0.062 |
| WR | return window | 2062 | +0.028 |
| TE | listed Q/D | 433 | -0.097 |
| TE | return window | 1324 | -0.012 |

## 4. Does skill trend predict the next season?

Partial correlations with NEXT-season outcome, controlling current level
(residualize on current skill_composite + age):

| Pos | Predictor | vs next skill z | vs next PPG | n |
|---|---|---|---|---|
| QB | trend2 | -0.208 | -0.115 | 322 |
| QB | inseason_trend | 0.037 | -0.080 | 424 |
| RB | trend2 | -0.009 | -0.136 | 422 |
| RB | inseason_trend | 0.096 | -0.006 | 657 |
| WR | trend2 | -0.161 | -0.194 | 848 |
| WR | inseason_trend | -0.008 | 0.010 | 1226 |
| TE | trend2 | -0.087 | -0.178 | 351 |
| TE | inseason_trend | -0.006 | 0.031 | 501 |

Mean reversion (all positions): players 1z above era mean keep on average
0.32z next year (n=387); players 1z below keep -0.42z (n=283).

Saved table player_skill_seasons: 4,403 rows.
