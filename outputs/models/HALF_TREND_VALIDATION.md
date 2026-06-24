# Does the late-season role/production TREND predict next year? (Walk-forward)

**Question:** players who "found an increased role late" — does that second-half
surge carry into next season enough to improve our projections?

**Method:** built `nflv_half_trend` (2012-2026) — for each player-season, the H1→H2
change in fantasy PPG, snap %, touches/g, target share, plus end-of-season levels
and a per-week momentum slope. Merged onto `season_dataset` (keyed to the *target*
season, fully pre-season — no leakage) and compared **BASE vs BASE+trend**
season-blocked (train < T, predict T) per position. (`scripts/test_half_trend.py`)

**Result — no predictive value added:**

| Metric | BASE | +trend | Δ |
|---|---|---|---|
| Central PPG MAE (all) | 2.694 | 2.693 | **−0.001** |
| Bust AUC | 0.879 | 0.876 | −0.002 |
| Boom AUC | 0.893 | 0.893 | +0.000 |

**Why.** Standalone correlations with next-season PPG:
- `ht_h2_ppg` (second-half PPG *level*): **+0.73** — but that's just "recent form,"
  already captured by full-season `prior_ppg`.
- `ht_d_ppg` (the half-to-half *change*): **−0.01** · `ht_d_snap` **−0.02** ·
  `ht_d_tgtsh` **−0.00** — i.e. the *trend itself* is noise for next year.

Late-role spikes mostly come from others' injuries, soft late-season schedules, and
small samples, and revert. The persistent ones (a rookie genuinely winning a job)
are already reflected in the full-year line.

**Decision:** do **not** fold the trend into the projection numbers (same bar that
kept career comps as a display-only layer). Surface it instead as the **📈 Late
Risers** scouting tab — the role deltas let a human apply offseason context the model
can't (did the player they replaced leave? is the role sticky?), while the validated
`'26 Proj` column remains the source of truth for value.
