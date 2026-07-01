# Backtest: do QB-injury-depressed pass-catchers rebound above projection?

Reproducible: `scripts/teammate_injury_backtest.py` (nfl_data_py weekly, PPR, flag years 2018–2023
→ outcome 2019–2024). **Verdict: no actionable edge — do not lift these players' projections.**

## Setup
- **FLAGGED** (107 player-seasons): pass-catcher whose team's primary QB missed ≥3 games, who was a
  real contributor with the QB (in-PPG ≥ 8) and dropped ≥22% without him.
- **CONTROL** (484): pass-catchers (PPG ≥ 8) whose primary QB was stable (played ≥15 games).

## Result
| Group | n | blended Yr | with-QB | next-yr | rebound |
|---|--:|--:|--:|--:|--:|
| Flagged | 107 | 10.9 | 12.7 | **10.0** | −0.9 |
| Control | 484 | 13.2 | — | 11.8 | −1.4 |

- Flagged players **regress like everyone else** — they land near their *depressed* line next year
  (10.0), nowhere near their with-QB rate (12.7). Net edge vs control: **+0.5 PPG** (tiny).
- **Beat-their-line rate:** flagged 42% vs control 33% — a small positive tilt, not a big one.
- **Next-year prediction MAE (flagged):** from the depressed line **2.68**; from the with-QB rate
  **3.48** (worse); from a 50/50 blend **2.98** (still worse).

## Takeaways
- The intuition ("QB heals → player bounces back") is real *directionally* but weak, and the good
  games with the QB are a positively-selected sample — projecting at that rate **overshoots**.
- **Do not apply upward projection adjustments** to screened players (Chase Brown, Colts cluster,
  etc.). The screen (`teammate_injury_undervalued.py`) is informational; treat a flag as at most a
  soft tiebreaker (~42/33 tilt), never a value bump.
- Confirms the earlier revert of the Ja'Marr Chase +Burrow override.
