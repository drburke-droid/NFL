# Value-Leap — finding cheap players who get "drafted much higher" next year

The season-level analog of the weekly explosion model. Among players who were
CHEAP / undraftable-caliber last year (prior PPG below a startable line), predict
who breaks out into a startable asset this season — the late-round darts whose ADP
then jumps the following year.

`scripts/test_value_leap.py` (validation), `scripts/build_value_leap.py` (2026 scores).

## Definition
- Pool: prior_ppg < {RB 7, WR 7, TE 5} with ≥3 games (a cup of coffee, not a fantasy asset).
- Leap: next_ppg ≥ {RB 11, WR 11, TE 8} AND a jump of ≥ +4 PPG.
- QBs excluded — a cheap-QB "leap" is usually just winning a starting job (streaming position).

## Validated (walk-forward 2017-2025)
- Base leap rate **4.3%** (rare, like the explosion tail).
- **AUC 0.702**, AP 0.091.
- **precision@top10% = 11.8% → 2.75× base** (top-5% 2.1×, top-20% 2.2×) — comparable to / a touch stronger than the explosion model.
- Top-decile flagged players average **6.1 next PPG vs 3.9** for the pool.
- Face validity: Baker Mayfield '23 (10.1→16.1), Trevor Lawrence '22, Justin Fields '22, Mike Gesicki '19, Rico Dowdle '24, Myles Gaskin '20.
- **Top drivers** (the synthesis): `vac_pc_targets` (vacated targets), `draft_pick` (capital), `ht_d_tch` / `ht_slope` (late-season surge), `prior_cv`, EPA. The niche signals we built feed this directly.

## Productized
Calibrated probability (Platt) so the surfaced % is realistic. `nflv_value_leap`
scores 295 cheap 2026 skill players; `build_draft_tool` carries `leap_prob` into
data.js; web app has a **🚀 Sleepers** tab ranking them with a "why" (vacated role /
won job / late surge). `refresh_2026` runs half-trend → opportunity → value-leap
before the board.

2026 standout: **Dylan Sampson (CLE RB) 26%**; the rest form a realistic ~6% long
tail (deep breakouts are genuinely rare, so the few standouts are the signal). These
are upside dart-throws, not safe picks.
