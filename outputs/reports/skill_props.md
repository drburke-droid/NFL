# Does true skill add value to weekly player props?

**Script:** `scripts/skill_props_study.py` (2026-07-15)

**Setup:** 25K matched closing props 2024-25 from `outputs/prop_ev_backtest.pkl`
(de-vigged consensus `novig`, actual outcome, ≥3 books). Skill joined causally:
`skill_true_causal` + `alpha_skill` through season S-1, plus within-season
play-weighted EPA/play to date (weeks < w, min 50 plays).

## Headline: NO

- **Logistic won_over ~ novig + skill:** pooled coefficients +0.01 to +0.03
  (a 1σ skill edge shifts P(over) by well under 1pp). No market shows a
  usable coefficient.
- **Residual test** (skill vs standardized actual−line, controlling novig +
  line level): receiving markets — where alpha has real draft-time value —
  are dead flat (r = −0.01 to +0.02). Two apparent pockets both fail the
  season split:
  - rush yds EPA-to-date: 2024 r=+0.068 (p=.006) → 2025 r=+0.004. Dead.
  - pass yds EPA-to-date: r=+0.126 but exists only in 2025 (n=599) — no
    out-of-sample year to confirm; treat as noise.
- **Brier ladder (train 2024 → test 2025):** M2 market logistic 0.24783;
  adding skill features makes it *worse* (0.24832). Nothing beats
  market-only models. Skill features rank 22nd/32nd/37th in the anchored LGBM.
- **ROI:** the skill-augmented models degrade the simple M1-shrink strategy
  (+3.0% → +0.9% at EV>2% best-book; negative at median execution).

## Why this differs from the draft-time result

Preseason ADP is a soft market (yearly, recreational, no closing-line
mechanism) — alpha_skill finds a real WR edge there (see skill_vs_adp.md).
Weekly closing prop lines are sharp: they already embed recent volume,
efficiency, matchup, and injury news by kickoff. Everything the skill
estimates know is priced. Consistent with prop_ev_model.md (closing line
beats a full stats LGBM in all 6 markets).

**Practical rule unchanged:** the only prop margins remain structural —
under-shading + best-of-book line shopping — not player evaluation.
