# Fantasy theory backtests — scoreboard

Reproducible: `scripts/theory_backtests.py` + `theory_backtests2.py` (nfl_data_py weekly 2017–24 +
odds DB `nflv_game_lines`). **Leading-indicator test:** regress next-year PPR PPG on
`[last-yr PPG, age, signal]`; a theory is a SIGNAL if its coef is significant (|t|>2) AND it cuts
walk-forward MAE. **Caveat:** the baseline is naive (last-yr PPG + age), *not* the full projection —
so signals the projection already uses are likely already priced in.

| Theory | test | verdict | already in proj? |
|---|---|---|---|
| **Vegas implied team total** | t=+7.7, ΔMAE +.034 | ✅ SIGNAL | **no — external, 2026 live** |
| Availability / durability (games→games) | t=+8.6, ΔMAE +.081 | ✅ SIGNAL | partly (injury-risk term) |
| Draft capital (overall pick) | t=−4.0, ΔMAE +.030 | ✅ SIGNAL | yes |
| Volume: target share (WR/TE) | t=+5.0, ΔMAE +.027 | ✅ SIGNAL | yes |
| TD regression (TD/opportunity) | t=−2.7, ΔMAE +.003 | 🟡 weak | partly |
| Yards/game | t=+2.8, ΔMAE +.008 | 🟡 weak | yes |
| Scheme: team pass rate | t=+3.2, ΔMAE −.001 | 🟡 weak | yes |
| RB opportunities/game | t=+1.1 | ❌ none | yes |
| QB quality: incoming QB prior ppg | t=−0.9 | ❌ none | — |
| Athletic: RB Speed Score | +1.3 PPG hi vs lo | 🟡 modest | maybe |

**Age cliff (descriptive):** next-yr PPG change by age — RB 27-28 **−2.9**, 31-32 **−5.7**; WR 29-30
−1.6, 31-32 −2.9; TE flat/noisy. **2nd-year WR leap:** rookie 8.4 → yr2 9.1 (+0.7); small on
average but rookie target share predicts which sophomores leap (t=+4.0).

## Validation vs the ACTUAL projection (scripts/theory_validate.py)
The table above uses a naive baseline. The stricter test: does each signal add over a RICH model that
mirrors the projection (multi-yr PPG, target share, yards, opportunity, age, draft, durability, pos)?
Walk-forward MAE: naive[ppg,age] **2.472** → rich proxy **2.413** (so those features ARE used).
Incremental MAE improvement OVER the rich proxy (same sample, n=1815):

| signal added over full projection | ΔMAE | verdict |
|---|--:|---|
| **Vegas Y+1 implied team total** | **+0.063** | ✅ ADDS — genuinely new |
| TD regression (TD/opp) | −0.002 | ~0 already in |
| Scheme: team pass rate | −0.003 | ~0 already in |
| QB quality: incoming QB | −0.001 | ~0 already in |
| *(control) target share* | −0.000 | ~0 already in — validates the test |

The target-share control (a signal known to be in the model) correctly adds 0, confirming the method.
So the ~0s for TD-regression / pass-rate / QB are real: **the projection already accounts for them.**
Only **Vegas team total** survives — external data no player-level model contains.

## Recommendation
- **Apply Vegas team total** as a small environment adjustment (external signal, not in the model,
  and 2026 implied totals exist) — after confirming it improves MAE vs the *actual* projection, not
  just the naive baseline.
- **Durability** signal validates keeping the existing injury-risk term (no new work).
- The rest are either **already in the projection** (target share, draft, age, yards) → no additive
  bump, or **weak/null** (RB opp, QB quality, scheme) → not applied. Matches prior discipline:
  walk-year and vacated-role earned bumps; fade, teammate-injury, Chase-Burrow did not.
