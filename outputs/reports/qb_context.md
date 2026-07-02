# QB-context-independent WR evaluation (2026-07-01)

Reproducible: `scripts/qb_context.py`. Question: were Jefferson's (and Chase's) 2025 numbers depressed
by QB play, and do QB-excuse down years rebound? Method: team QB quality (EPA/att, CPOE, ranks of 32),
role metrics (target/air-yards share), xFP opportunity-vs-conversion, per-target pbp splits, and a
historical rebound test against the ACTUAL model's projections.

## 2025 facts — the eye test is CORRECT
- **MIN QB play was bottom-5**: EPA/att −0.116 (rank 28), CPOE −2.3 (rank 26), 10.9% sack rate.
- **Jefferson's role was fully elite and INTACT**: target share ROSE 28→30→31%, air-yards share 40%,
  8.3 tgt/g. His opportunity (xFP) said ~249 pts; he converted 202 (**gap −48** — conversion failure,
  not role loss). MIN passes TO him: 59% comp, **−0.10 EPA/att** vs +0.12 to everyone else (aDOT 10.2
  vs 7.0 — they were short-balling everyone but him and missing him deep).
- **Chase 2025 was NOT a down year**: 19.6 PPG = **WR3 in per-game scoring** — achieved with CIN QB
  play ranked 27th (Burrow out). Target share 32% (career high), xFP gap +7 (healthy conversion).
- **Tyreek 2024 precedent is a DIFFERENT failure mode**: his ROLE collapsed with the backups
  (10.5→7.2 tgt/g, share 32→22%) — role loss, not conversion loss. He then didn't rebound in 2025.

## The decisive history — QB-excuse down years do NOT rebound
Down-year elite-role WRs (fell ≥3 PPG, share ≥20%, prev ≥10 PPG), residual vs the actual model:

| group | n | resid vs model | raw rebound |
|---|--:|--:|--:|
| QB was BAD (EPA rank 24+) | 13 | **−0.74** | −0.6 PPG |
| **QB was FINE (rank ≤16)** | 16 | **+2.17 (t=+2.6)** | +1.5 PPG |
| JJ-exact profile: role intact + conv gap ≤−1/g | 10 | **−1.60** | −0.8 |
| QB-bad AND QB context IMPROVED next yr | 10 | −0.49 | **+0.0** |

The narrative case (Hopkins '17: 12.3→20.7) is real but exceptional — the base rate says the QB-excuse
group *underperforms* even our projection, **even when the QB gets fixed**. The rebound belongs to the
opposite group: a down year DESPITE fine QB play (luck/TD variance) mean-reverts hard. Consistent with
the lottery anti-signal (xFP "unlucky" 0.7x) and the QB-quality null in theory_backtests.

## 2026 watchlist (mechanical rule; nuance below)
- **REBOUND candidate (QB fine): DeVonta Smith** (15.3→11.9, PHI QB rank 12) — the +2.17 profile.
- Middle: McConkey, Evans, McLaurin.
- **QB-excuse group (no historical rebound): Jefferson, Meyers, Jeudy** — and nominally Chase, but see below.

## Verdicts
- **Jefferson**: the eye test on CAUSE is right (QB-driven, role intact) — but the base rate on
  RECOVERY is against him (−1.60 resid for exactly his profile; 0.0 even when QB improves). Model
  stays 13.4 PPG / $25. At a $58 market price he remains the board's biggest trap. If you want
  exposure to the exception case, it's a FLEX-price bet, not a $58 bet.
- **Chase is the special case that survives**: he never collapsed — 19.6 (WR3) *with* rank-27 QB play
  is evidence of genuine QB-independence, and 2024-with-healthy-Burrow was 23.7. Cap stretch from $55
  to **~$58** is defensible on the healthy-Burrow scenario; FFA blend will lift him further.
- The model needs NO change: it already ranked Chase WR2 and correctly refuses the JJ rebound the
  history doesn't support. DeVonta Smith added as a value target (exp ~$27 vs rebound profile).
