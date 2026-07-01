# Historical fantasy trends 2011–2024 (actionable sweep)

Reproducible: `scripts/trends_sweep.py` (nfl_data_py weekly `fantasy_points_ppr`, 2011–2024; 2025 weekly
fantasy not yet released). Each trend labelled **EDGE** (potential new signal), **DRAFT-STRATEGY**
(context for your decisions, model already fair), or **BUSTED** (thesis not supported in the data).

## 1. Year-over-year repeatability by position — *keeper safety* — DRAFT-STRATEGY
`corr(PPG_Y, PPG_Y+1)`, ≥8-game seasons. Higher = safer to keep.

| Pos | all | 2011–17 | 2018–24 | top-12 → top-12 repeat |
|---|--:|--:|--:|--:|
| WR | 0.71 | 0.68 | **0.73** | 40% |
| TE | 0.70 | 0.68 | 0.72 | 47% |
| RB | 0.66 | 0.62 | **0.70** | 40% |
| QB | 0.57 | 0.65 | **0.50** | **56%** |

- **WR is the most stable keeper** (highest, still rising). **RB stability is *rising*** (0.62→0.70) — the
  old "RBs are volatile" narrative is weakening as the position concentrates into stable workhorses
  (ties to the RB-archetype finding: fewer fluky pass-specialists).
- **QB PPG is getting *less* predictable** (0.65→0.50) — mobile-QB variance/injury — **yet elite QBs are
  the stickiest tier** (56% top-12 repeat). Translation: an *elite* QB keeper is safe; a *mid* QB is a coin flip.
- Top-12 RB/WR repeat is only 40% → **don't overpay to keep a fringe RB1/WR1 expecting a repeat.**

## 2. Positional scarcity — where the points are — DRAFT-STRATEGY
Min PPG of the tier your league starts (early → recent era):

| Pos | tier | 2011–17 → 2018–24 |
|---|---|---|
| RB | #12 / #24 | 13.3→**14.5** / 10.4→11.2 (**scarcer/richer at top**) |
| WR | #12 / #24 | 15.7→15.2 / 12.5→12.7 (flat, **deep**) |
| TE | #6 / #12 | 12.4→**11.2** / 9.7→**8.9** (**top tier thinning**) |
| QB | #6 / #12 | 18.4→19.3 / 16.1→16.7 (up modestly) |

- **RB top tier is getting richer** → paying up for a bell-cow RB is *justified* by the data (matches the
  "RBs are expensive again" narrative). **WR is deep** → you can wait. **Elite TE is scarcer** → either
  pay for a top-3 TE or punt/stream; the middle is thinner than it was.

## 3. Age curves by position — mean next-year PPG change — DRAFT-STRATEGY (already in projection)
| Pos | 24–26 | 26–28 | 28–30 | 30–33 |
|---|--:|--:|--:|--:|
| RB | −1.0 | −1.3 | **−2.3** | −2.1 |
| WR | −0.4 | −0.9 | −1.4 | −1.8 |
| TE | −0.3 | −0.4 | −0.7 | −0.7 |
| QB | (noisy) | −0.7 | −0.5 | −1.2 |

- **RB falls off a cliff at 28+; TE ages best** (barely declines into the 30s); WR is gradual. **A 30-yo TE
  keeper is far safer than a 28-yo RB.** (Age is already a projection feature — this validates it, no bump.)

## 4. QB rushing edge — rushing share of QB1 fantasy points — DRAFT-STRATEGY (likely priced)
2011–15 **12%** → 2016–20 15% → 2021–24 **19%**. Rushing increasingly defines QB1 scoring → **prioritize
dual-threat QBs**; their rushing floor is stickier than passing TDs. (Rushing yards are already a model input.)

## 5. Rookie WR immediate impact — trend REAL, edge BUSTED (validated)
Mean rookie-WR PPG is flat (~8), but **rookie WRs hitting ≥12 PPG jumped 2.4/yr → 2.6 → 3.8/yr** (2021–24) —
hit rate up ~50%. **Tested** whether the actual rookie model under-projects them (`scripts/rookie_wr_era_test.py`):
it does **not**. Walk-forward residual is flat (2016–20 +0.30 vs 2021–25 +0.20; slope t=+0.1) and the model
actually got *sharper* recently (rank corr 0.48 → **0.69**). A trailing-residual bias-correction **HURTS**
MAE (−0.075). The extra hits are a right-tail/draft-capital phenomenon the model already ranks correctly —
draft capital + landing spot encode the modern rookie WR. No edge, no change.

## 6. "Death of the WR2" (season-long) — BUSTED in this data
Season-long WR fantasy-point concentration is **remarkably flat** across 14 years: top-12 ≈ 19–20%,
WR13–24 ≈ 15%, WR25–48 ≈ 23% — no shift. The JJ deployment thesis is a *target-rate/2025* story that does
**not** show up in season-long PPR distribution 2011–24. TE concentration is likewise flat (top-6 ≈ 20%).
Don't restructure a draft around a WR2 collapse that the long-run scoring data doesn't confirm.

## 7. Weekly consistency (floor) — median week-to-week CV — DRAFT-STRATEGY
QB **0.44** < RB 0.56 < WR 0.60 < TE 0.61. **QBs are the steadiest weekly floor; TE/WR are boomiest** —
relevant for best-ball/weekly-lineup risk, not season-long value.

## Bottom line
Most of these are **draft-strategy intelligence**, not model bumps (the projection already prices age,
usage, rushing). The rookie-WR hit-rate edge (#5) was **tested and busted** — the model already ranks
recent rookie WRs correctly (rho 0.69), a correction hurts. The **cautionary counter-finding** is that
the season-long "WR2 death" (#6) isn't in the data. Net: no new model bumps earned; the value is in the
keeper-safety / scarcity / age playbook above. Consistent with the standing scoreboard — real trends
keep turning out to be already priced; only genuinely external signals (Vegas already in model, walk-year,
vacated-role) have ever earned a bump.
