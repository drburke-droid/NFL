# Segment heterogeneity study (2026-09-15)

**Verdict: segment-specific corrections are worth at most 0.15% MAE, and none of it comes from team or player identity.**
26 segmentation schemes (team, team x early weeks, opponent, position x tier, star vs rest, RB/WR/QB archetypes, experience,
volatility, home, favourite, total, Q-tag, opponent pressure, player identity, and in-season hot/cold) were tested the same
way: empirical-Bayes shrunk offsets fitted on earlier seasons, applied walk-forward 2018-2025 to the incumbent, with a
permutation test for in-sample structure. Findings:

- **Team, opponent, player: nothing.** Team and player offsets shrink to zero (shrink 0.00-0.10) and score 0.000 out of
  sample; the Bengals-weeks-1-2 example swings from -4.0 to +2.6 by season. In-season "this team/player is running hot vs the
  model" offsets are worse than doing nothing. A team- or player-specific model has no signal to learn.
- **Where real structure lives: the shape of the error, not its centre.** Position x tier, volatility, WR deep-share, home,
  favourite, Q-tag and opponent pressure all show in-sample structure (perm p <= 0.01). Applied as mean offsets they improve
  RMSE (up to -0.03, 8/8 seasons) but WORSEN MAE, because the incumbent already targets the median and these segments differ in
  skew (stars and deep-ball WRs have a lower median but a longer right tail). Applied as median offsets the best three
  (position x volatility, WR deep-share, position x early) stack to -0.007 MAE (0.15%), 7/8 seasons, p = 0.01, with zero change
  in ranking (Spearman) and a slight LOSS in weeks 1-2. Adding tier on top gives it back.
- **Stars do not want a different DK weight.** Fitting the per-stat DraftKings blend weight by star/rest, position, tier or
  early weeks scores 5.036-5.038 vs 5.035 global (2024-25, 11,040 rows). The 0.9 yards / 0.3 receptions weights are the answer
  for everyone.
- **Week 2 specifically:** no segment scheme helps weeks 1-2; the stacked best makes them slightly worse.

What could still be worth doing (not done here): a projection-conditioned quantile/skew layer (the same "shape not centre"
finding as model-league Round 1's exp_004), which would improve RMSE/pinball at equal MAE. It is a distribution change, not
a "fade this team" rule.

Scripts: scripts/segment_hetero_study.py (+ the stacking block appended inline). Frame: league visible frame + 2025 with
actuals, incumbent refit on played rows from 2016.

---

# Detail

Played rows 2017-2025 with an incumbent prediction (n = 43,564); residual = actual - incumbent. Out-of-sample = EB-shrunk segment offsets fitted on earlier seasons, applied walk-forward 2018-2025. Perm p = probability that shuffled labels (within season) explain as much residual variance as the real labels; shrink = how much of a typical segment mean survives shrinkage (0 = nothing but noise).

## Cross-season schemes

**Best cross-season schemes by out-of-sample dMAE (negative = better than the incumbent):** pos x volatility -0.0055 (perm p 0.00); WR deep-share -0.0036 (perm p 0.00); pos x early -0.0025 (perm p 0.01); pos x tier -0.0024 (perm p 0.00); pos x favourite -0.0022 (perm p 0.01)

| scheme | segments | perm p | var explained (obs / null, pts^2) | shrink | OOS dMAE vs inc (median offsets) | seasons better | t p | OOS dRMSE (mean offsets) | biggest surviving median offsets (train = all seasons) |
|---|---|---|---|---|---|---|---|---|---|
| team | 33 | 0.235 | 0.033 / 0.029 | 0.10 | +0.0001 | 1/8 | 0.19 | -0.0046 (6/8) | LA +0.03; IND -0.02; DAL +0.02; DET +0.02 |
| team x early | 99 | 0.065 | 0.107 / 0.090 | 0.07 | +0.0014 | 1/8 | 0.07 | -0.0085 (8/8) | WAS/wk5+ -0.09; SEA/wk3-4 +0.08; IND/wk5+ -0.08; MIN/wk5+ +0.08 |
| team x pos | 132 | 0.975 | 0.095 / 0.123 | 0.00 | +0.0000 | 0/8 | nan | +0.0000 (0/8) | ARI/QB +0.00; ARI/RB +0.00; ARI/TE +0.00; ARI/WR -0.00 |
| opponent | 32 | 0.005 | 0.054 / 0.029 | 0.45 | +0.0004 | 2/8 | 0.48 | -0.0139 (8/8) | BUF -0.16; LV +0.16; MIA +0.14; LAC -0.13 |
| opponent x pos | 128 | 0.000 | 0.189 / 0.119 | 0.36 | -0.0010 | 6/8 | 0.17 | -0.0104 (7/8) | TEN/QB +0.40; CIN/QB +0.40; TB/QB +0.35; LAC/WR -0.34 |
| pos x tier | 20 | 0.000 | 0.123 / 0.020 | 0.82 | -0.0024 | 5/8 | 0.58 | -0.0312 (8/8) | WR/7-12 -0.67; WR/13-24 -0.66; RB/7-12 -0.59; WR/top6 -0.59 |
| pos x tier x early | 60 | 0.000 | 0.182 / 0.056 | 0.41 | -0.0002 | 4/8 | 0.95 | -0.0289 (8/8) | WR/7-12/wk5+ -0.86; TE/top6/wk1-2 -0.72; RB/top6/wk5+ -0.71; WR/top6/wk1-2 -0.70 |
| pos x star x early | 24 | 0.000 | 0.087 / 0.023 | 0.59 | -0.0019 | 5/8 | 0.24 | -0.0279 (8/8) | WR/star/wk5+ -0.66; RB/star/wk5+ -0.64; WR/star/wk1-2 -0.43; TE/star/wk1-2 -0.42 |
| pos x early | 12 | 0.005 | 0.024 / 0.012 | 0.37 | -0.0025 | 7/8 | 0.00 | -0.0207 (8/8) | WR/wk5+ -0.17; TE/wk5+ +0.16; QB/wk5+ +0.13; QB/wk3-4 +0.08 |
| RB archetype | 4 | 0.075 | 0.006 / 0.003 | 0.17 | -0.0003 | 3/8 | 0.34 | -0.0183 (6/8) | mixed -0.05; pass-catch +0.03; na +0.01; ground -0.00 |
| RB archetype x early | 12 | 0.085 | 0.017 / 0.010 | 0.17 | -0.0002 | 5/8 | 0.60 | -0.0210 (8/8) | mixed/wk5+ -0.08; pass-catch/wk5+ +0.04; na/wk1-2 -0.04; pass-catch/wk3-4 +0.03 |
| WR deep-share | 4 | 0.000 | 0.023 / 0.003 | 0.79 | -0.0036 | 7/8 | 0.00 | -0.0215 (7/8) | deep -0.39; short +0.15; na +0.10; mid -0.05 |
| QB scramble | 4 | 0.485 | 0.004 / 0.004 | 0.01 | -0.0002 | 2/8 | 0.26 | -0.0039 (2/8) | na -0.01; scrambler +0.01; pocket +0.00; mid +0.00 |
| pos x experience | 8 | 0.215 | 0.009 / 0.008 | 0.21 | -0.0020 | 6/8 | 0.03 | -0.0128 (7/8) | WR/mid -0.10; QB/mid +0.05; TE/new(<6 gms) +0.04; RB/mid -0.04 |
| pos x volatility | 16 | 0.000 | 0.043 / 0.015 | 0.67 | -0.0055 | 7/8 | 0.02 | -0.0221 (8/8) | WR/volatile -0.48; RB/mid -0.26; WR/mid -0.25; TE/steady +0.24 |
| pos x home | 8 | 0.005 | 0.023 / 0.008 | 0.68 | -0.0016 | 6/8 | 0.04 | -0.0199 (8/8) | QB/home +0.25; WR/away -0.19; TE/home +0.13; WR/home -0.10 |
| pos x favourite | 16 | 0.010 | 0.031 / 0.016 | 0.53 | -0.0022 | 6/8 | 0.02 | -0.0148 (7/8) | RB/fav +0.23; WR/pickem -0.18; QB/pickem +0.18; TE/pickem +0.15 |
| pos x total | 16 | 0.885 | 0.010 / 0.016 | 0.00 | +0.0001 | 0/8 | 0.35 | -0.0014 (1/8) | QB/hi +0.00; QB/lo +0.00; QB/mid +0.00; QB/na +0.00 |
| pos x Q-tag | 8 | 0.010 | 0.015 / 0.008 | 0.38 | -0.0012 | 7/8 | 0.03 | -0.0173 (7/8) | QB/ok +0.14; WR/ok -0.14; TE/ok +0.11; WR/Q -0.07 |
| pos x opp pressure | 16 | 0.015 | 0.032 / 0.016 | 0.53 | -0.0020 | 6/8 | 0.01 | -0.0135 (7/8) | WR/lo-press -0.20; TE/lo-press +0.18; WR/mid -0.16; WR/hi-press -0.11 |
| pos x tier x favourite | 80 | 0.000 | 0.211 / 0.075 | 0.56 | -0.0020 | 5/8 | 0.57 | -0.0288 (8/8) | RB/13-24/dog -0.78; WR/top6/pickem -0.75; RB/top6/pickem -0.73; RB/7-12/dog -0.66 |
| pos x tier x total | 80 | 0.000 | 0.187 / 0.074 | 0.53 | +0.0004 | 3/8 | 0.92 | -0.0252 (8/8) | WR/13-24/mid -0.64; WR/top6/lo -0.59; WR/7-12/hi -0.55; RB/7-12/mid -0.55 |
| player (cross-season) | 1404 | 1.000 | 0.789 / 1.261 | 0.00 | +0.0000 | 0/8 | nan | +0.0000 (0/8) | tom brady -0.00; drew brees -0.00; josh mccown +0.00; carson palmer +0.00 |
| player x early | 3459 | 1.000 | 2.244 / 3.183 | 0.00 | +0.0000 | 0/8 | nan | +0.0000 (0/8) | tom brady/wk1-2 +0.00; tom brady/wk3-4 +0.00; tom brady/wk5+ -0.00; drew brees/wk1-2 -0.00 |
| team x season-phase x pos | 395 | 0.990 | 0.319 / 0.366 | 0.00 | -0.0000 | 2/8 | 0.17 | -0.0005 (2/8) | ARI/wk1-2/QB +0.00; ARI/wk1-2/RB +0.00; ARI/wk1-2/TE -0.00; ARI/wk1-2/WR +0.00 |

## Within-season running offsets (walk-forward by week; offset = EB-shrunk mean residual of the segment in the season's earlier weeks)

| scheme | min prior weeks | OOS dMAE vs inc | seasons better | t p |
|---|---|---|---|---|
| team, this season | 2 | +0.0014 | 1/8 | 0.01 |
| team, this season | 4 | +0.0005 | 1/8 | 0.17 |
| player, this season | 2 | +0.0000 | 0/8 | nan |
| player, this season | 4 | +0.0000 | 0/8 | nan |
| team x pos, this season | 3 | +0.0001 | 1/8 | 0.39 |
| player, prior + this season | 0 | +0.0000 | 0/8 | nan |

## The two examples, for the record (in-sample means of actual - incumbent)

- Bengals weeks 1-2, all seasons: mean residual -0.40 on 159 rows; by season 2017 -4.0 (n=19), 2018 +2.0 (n=19), 2019 +2.6 (n=17), 2020 +1.4 (n=19), 2021 -0.2 (n=15), 2022 +0.0 (n=16), 2023 -3.3 (n=17), 2024 -0.8 (n=19), 2025 -1.2 (n=18)
- Every team's weeks 1-2 mean residual is inside -0.73 .. +1.54; the EB shrinkage on team x early keeps 6% of a typical team mean.
- Stars vs the rest (star = top-6 QB/TE, top-12 RB/WR by FFA that week):

| pos | group | n | inc MAE | FFA MAE | mean residual |
|---|---|---|---|---|---|
| QB | rest | 4531 | 5.337 | 5.357 | +0.61 |
| QB | star | 948 | 6.546 | 6.558 | +0.22 |
| RB | rest | 9594 | 4.173 | 4.212 | +0.77 |
| RB | star | 1896 | 6.940 | 7.000 | +0.07 |
| TE | rest | 8052 | 3.381 | 3.425 | +0.75 |
| TE | star | 948 | 5.989 | 6.110 | -0.03 |
| WR | rest | 15699 | 4.471 | 4.527 | +0.59 |
| WR | star | 1896 | 7.160 | 7.328 | +0.01 |

## DraftKings weight heterogeneity (2023-25 played rows with a DK line; PPR rebuilt from FFA stats with the DK line blended per stat)

| weight scheme | 2024 MAE | 2025 MAE | pooled | fitted weights (train 2023-24, yards/rec) |
|---|---|---|---|---|
| global | 5.111 | 4.966 | 5.035 | all:1.0/0.3 |
| by position | 5.111 | 4.969 | 5.037 | ('QB',):1.0/0.0; ('RB',):1.0/0.3; ('TE',):0.5/0.3; ('WR',):1.0/0.6 |
| by star | 5.109 | 4.969 | 5.036 | ('rest',):1.0/0.6; ('star',):0.7/0.0 |
| by position x star | 5.109 | 4.969 | 5.036 | ('QB', 'rest'):0.9/0.0; ('QB', 'star'):1.0/0.0; ('RB', 'rest'):1.0/0.6; ('RB', 'star'):0.7/0.0; ('TE', 'rest'):0.5/0.3; ('TE', 'star'):0.5/0.3; ('WR', 'rest'):1.0/0.6; ('WR', 'star'):1.0/0.0 |
| by tier | 5.115 | 4.969 | 5.038 | ('13-24',):1.0/0.6; ('25-36',):1.0/0.6; ('37+',):1.0/0.6; ('7-12',):0.5/0.0; ('top6',):0.7/0.0 |
| by early | 5.112 | 4.966 | 5.036 | ('wk1-2',):1.0/0.3; ('wk3+',):1.0/0.3 |
| by position x early | 5.110 | 4.972 | 5.038 | ('QB', 'wk1-2'):1.0/0.3; ('QB', 'wk3+'):0.5/0.0; ('RB', 'wk1-2'):0.9/0.0; ('RB', 'wk3+'):1.0/0.6; ('TE', 'wk1-2'):1.0/0.0; ('TE', 'wk3+'):0.5/0.3; ('WR', 'wk1-2'):1.0/0.6; ('WR', 'wk3+'):1.0/0.6 |

FFA-only MAE on the same rows: 2024 5.167, 2025 5.029. Rows: 11,040.


## Stacking the three best median schemes (pos x volatility + WR deep-share + pos x early), sequential residual fits, walk-forward 2018-2025

| variant | OOS dMAE vs inc | seasons better | t p | dMAE weeks 1-2 only | dMAE Spearman change |
|---|---|---|---|---|---|
| vol only | -0.0055 | 7/8 | 0.02 | +0.0110 | -0.0001 |
| vol + deep | -0.0066 | 7/8 | 0.01 | +0.0097 | -0.0000 |
| vol + deep + early | -0.0070 | 7/8 | 0.01 | +0.0055 | -0.0001 |
| vol + deep + early + tier | -0.0047 | 6/8 | 0.31 | -0.0010 | -0.0009 |

incumbent MAE on these rows: 4.603 (so -0.005 is 0.1%).

