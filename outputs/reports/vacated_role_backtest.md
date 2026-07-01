# Backtest: does the inheritor of a vacated WR role beat projection?

Reproducible: `scripts/vacated_role_backtest.py` (nfl_data_py weekly, PPR, 2018–2024). Ex-ante test
(no look-ahead): after a team loses a WR with ≥12% target share, the presumptive inheritor = the top
RETURNING WR in a WR2/3 band (6–18% target share). Control = role-matched holdovers on teams with no
vacancy (strips out mean-reversion).

## Result
| Group | n | Δ target share | PPG Y→Y+1 | beat last-yr line |
|---|--:|--:|--:|--:|
| Presumptive inheritor (ex-ante) | 76 | +0.5pp | 8.7 → **9.2 (+0.5)** | 51% |
| *actual biggest gainer* (look-ahead) | 57 | +4.2pp | 8.7 → 10.5 (+1.8) | 65% |
| Control (role-matched, no vacancy) | 132 | +0.1pp | 9.3 → **8.8 (−0.5)** | 38% |

- **Mechanism is real:** the actual inheritor jumps +4.2pp share / +1.8 PPG (65% beat their line).
- **Actionable edge ≈ +1.0 PPG** (inheritor +0.5 vs control −0.5), **beat-line 51% vs 38%** — a real
  directional tilt, the strongest of the "intuitive" concepts tested.
- **But fails the projection bar:** a flat +1 uplift gives MAE 3.38 vs 3.33 naive (slightly worse) —
  the edge is a noisy mean shift.
- **Weak link = prediction:** the top returning WR only actually inherits sometimes (+0.5pp ex-ante
  vs +4.2pp for the true gainer); teams often fill the hole with a FA/rookie.

## Smarter predictor — CLEAN vacancies (validated & applied)
`scripts/vacated_role_backtest2.py` adds the draft-capital refinement: only count the inheritor when
the team did NOT draft a WR in rounds 1–2 the next spring (a *clean* vacancy the holdover actually
gets, vs a *contested* one a rookie takes).

| Group | n | Δ target share | PPG Y→Y+1 | beat |
|---|--:|--:|--:|--:|
| **Clean vacancy** (no early WR drafted) | 46 | **+1.6pp** | 8.8 → 10.0 (**+1.2**) | 54% |
| Contested (drafted WR rd1–2) | 30 | −1.1pp | 8.6 → 8.0 (−0.6) | 47% |
| Control | 132 | +0.1pp | 9.3 → 8.8 (−0.5) | 38% |

**Clean edge vs control ≈ +1.73 PPG** (vs +1.0 naive), approaching the look-ahead ceiling; contested
inheritors *decline*, confirming the mechanism. MAE is **neutral** (3.79 vs 3.79) — no accuracy cost.

**APPLIED:** `scripts/vacated_role_2026.py` flags 2026 clean-vacancy WR inheritors (2025 usage from
play-by-play, 2026 teams from the roster release, draft from the draft_picks release) → a shrunk
**+1.0 PPG bump** (~0.6× the +1.7 edge) + an `↑ VACATED %` tag in the board. 9 players (e.g. Rashee
Rice ← M. Brown, Josh Downs ← Pittman, Quentin Johnston ← Keenan Allen).

## Standings of the intuitive concepts tested
- **Walk-year bump:** +1.41, improved MAE −1.7% → APPLIED.
- **Vacated-role inheritor:** +1.0 edge, 51/38 beat-line, but flat uplift ~neutral-to-worse on MAE →
  tiebreaker flag only.
- **Teammate-injury rebound:** +0.5 edge, with-QB rate hurt MAE → not used.
- **Post-contract fade / Chase-Burrow override:** failed → not used / reverted.
