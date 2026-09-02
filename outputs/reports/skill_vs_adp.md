# Does true skill predict beyond preseason rankings?

**Script:** `scripts/skill_vs_adp_study.py` (2026-07-15)

**Setup:** target season S, veterans only (need a prior-season skill estimate).
Predictors at draft time: preseason ranking (FFA points + pos rank, 2012-25;
FantasyPros ECR/ADP, 2021-25) vs `skill_true_causal` / `skill_trend`
(player_skill_true through S-1) and `alpha_skill` (player_skill_alpha).
Outcome: season-S PPG (`season_dataset.next_ppg`).

## Headline

- **Pure Kalman true-skill adds ~nothing beyond ADP for QB/RB/TE.** Partial
  r after controlling ffa_points + log pos rank + age: +0.05 to +0.08, none
  significant; walk-forward MAE deltas are noise (-3.6% to +1.1%).
- **Skill trend adds nothing anywhere** (partial r ≈ 0, all p > 0.2).
- **WR is the exception, and it's the alpha blend, not raw skill:**
  - alpha_skill partial r = **+0.152 (p<0.001)** on the 2012-25 FFA sample,
    **+0.207 (p<0.001)** vs FantasyPros ECR+ADP 2021-25.
  - Walk-forward WR MAE improves **+2.3%** over ranking-only and **+3.2%**
    over ranking+prior-PPG when skill features are added — the only position
    where adding skill helps out-of-sample.
  - Per-season stability: positive in 9/12 seasons; 2021-25 mean partial
    r ≈ +0.21 (2022 +0.42, 2025 +0.40; worst year 2019 -0.28).
- QB alpha is weakly positive on the long sample (+0.110, p=0.036) but dies
  on the sharper FantasyPros sample (+0.018) — treat as noise.

## Spearman sanity check (veterans, FFA sample)

| Pos | n | ffa_pos_rank | skill_causal | alpha |
|---|---|---|---|---|
| QB | 366 | +0.597 | +0.372 | +0.385 |
| RB | 631 | +0.699 | +0.265 | +0.455 |
| WR | 758 | +0.661 | +0.274 | +0.556 |
| TE | 362 | +0.618 | +0.288 | +0.491 |

Rankings alone are far stronger than skill alone — the market already prices
most of what skill measures. The question is only the residual.

## Within-ADP-tier test (hi vs lo skill, same tier)

WR hi-skill halves beat lo-skill on next PPG in every tier (18.5 vs 16.8 in
WR1-6; 15.4 vs 14.8; 14.4 vs 13.4; 11.4 vs 11.3). TE1-6 shows the same
(13.8 vs 11.6). QB reverses at the top (skill picks the wrong elite QBs).
RB tiers are flat — RB outcomes are volume/health, not measured skill.

## Interpretation

The market's preseason ranking already embeds ~everything the pure Elo/Kalman
skill estimate knows for QB/RB/TE. The WR edge comes from the earned-volume
component of alpha (efficiency that hasn't yet been paid volume — the
Chase-at-WR25 mechanism). Use alpha_skill as a WR tiebreaker/boost on the
draft board; do not use it to re-rank QB/RB/TE against ADP.
