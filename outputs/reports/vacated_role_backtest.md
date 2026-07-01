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

## Recommendation
Use as a **soft draft-target flag / tiebreaker** ("stepping into vacated targets, on a team that lost
≥12% share"), NOT a projection value bump. A smarter inheritor-predictor (incorporating offseason FA
signings, draft capital, and player age) could capture more of the +1.8 PPG look-ahead effect and
might then clear the MAE bar — worth building only if we want to act on this.

## Standings of the intuitive concepts tested
- **Walk-year bump:** +1.41, improved MAE −1.7% → APPLIED.
- **Vacated-role inheritor:** +1.0 edge, 51/38 beat-line, but flat uplift ~neutral-to-worse on MAE →
  tiebreaker flag only.
- **Teammate-injury rebound:** +0.5 edge, with-QB rate hurt MAE → not used.
- **Post-contract fade / Chase-Burrow override:** failed → not used / reverted.
