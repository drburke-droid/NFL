# Is the late-season role surge EVER a valid signal? (Conditional test)

The trend is noise on average (see HALF_TREND_VALIDATION.md). This asks whether
it's real inside a pre-specified, mechanism-driven subgroup — without overfitting.

**Discipline:** subgroups pre-specified from theory (round-number thresholds, no
search); tested against the walk-forward projection RESIDUAL (signal *beyond* the
model); each claim must replicate in two disjoint eras (2013-19 and 2020-25).
`scripts/test_risers_conditional.py`.

## Result

| Subgroup | n | beats proj by | replicates both eras? |
|---|---|---|---|
| Young pass-catcher, low line, won job (your exact case) | 26 | +0.5 | ❌ (+1.0 recent, −0.3 old) |
| **Early-career (≤2 yr), ended entrenched + role jump** | 133 | **+0.7** [+0.0,+1.5] | ✅ (+0.7 / +0.7) |
| Same surge, veterans (age ≥27) | 124 | **−0.5** | (opposite sign) |

The *narrow* "young WR" slice is too thin and era-unstable to trust alone. The
*broadened* version — an early-career player who **ended 2025 entrenched (H2 snap
≥55%) after an in-season jump (Δsnap ≥+12pp)** — beats its projection ~**+0.7 PPG**,
and the estimate is **identical in both halves of history**. The veteran mirror
flips negative. Pre-specified + replicates + sign-flip = credible despite a modest CI.

**Mechanism:** full-season `prior_ppg` understates a player whose role arrived late;
for the young, that role tends to stick, so the model runs slightly low. The model
has age and snap% but underweights their interaction.

**Caveats:** small effect (~0.7 PPG, lower CI ≈ 0); a tiebreaker, not a reach signal.
Threshold/membership effect, not "more snaps = more points."

## Productization
Display **tag only**, no projection override (overriding on +0.7 is the overfit we're
avoiding). `build_draft_tool.py` sets `won_job=1` for: exp≤2, age≤25, H2 snap ≥55%,
Δsnap ≥+12, H2 PPG > H1 PPG, H2 PPG ≥8 (drops already-priced studs who merely ramped,
e.g. Bowers/Worthy, and garbage-time depth). 2026: **Jalen Coker (CAR)** is the lone
qualifier. Shown as `⬆ WON JOB` on the board and floated to the top of Late Risers.
