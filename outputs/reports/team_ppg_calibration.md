# Team-PPG grading calibration (2026-07-17)

**Question:** what did teams in our league actually average, and are the draft tool's
team grades in line? (User observed FFA-mode mock winners at ~129 PPG/wk but Model-mode
winners far lower.)

## Ground truth — ESPN league standings (points_for / weeks)

| Season | League avg team PPG | Top team |
|---|---|---|
| 2023 | 118.4 | 131.9 |
| 2024 | 120.3 | 139.1 |
| 2025 | **118.7** | **136.7** |

## The bug, reproduced headless

Optimal drafted team, graded by the tool (starters, PPG/week):

| Mode | Before | After fix |
|---|---|---|
| FFA | 132.2 ✓ | 132.2 (unchanged) |
| Model | **112.4 ✗** (below the league AVERAGE — impossible for an optimal draft) | **137.8 ✓** |

Two causes, both in the weekly-grading layer (`wkPts = effPts/17`):

1. **Missed games counted as zeros.** Model projections embed durability-shrunk games
   (13-15); dividing season totals by 17 charges absence at 0 PPG, but in reality the
   slot scores ~replacement (you stream). Worth ~10 PPG/week of team grade.
2. **Scale compression.** Model per-game projections are deliberately compressed for
   ranking (top-12 QB avg 17.3 projected vs FFA consensus 22.8, actual-2025 realized
   23.7). Fine for ordering players, wrong for absolute team sums. Worth another ~10.

## The fix (grading layer only — Value/Exp $ engines untouched)

`wkPts` now = healthy per-game × projected games + **replacement-level backfill** for
projected missed games, all over 17, then **per-position rescale to the FFA consensus
starter-pool scale** (ex-ante like the model, and empirically aligned with the league's
real scoring): QB ×1.31, RB ×1.12, WR ×1.11, TE ×1.13
(= FFA top-N/17 ÷ model backfilled top-N; N = 12/24/24/12).

Deliberately NOT calibrated to realized 2025 leaders (23.7 QB etc.) — realized top-N
is outcome-selected (winner's curse); no honest ex-ante projection should match it.

Pure-FFA mode is untouched (consensus points are already on the realized scale).
Blend mode flows through the calibrated model path on its blended totals.

## Effects

- Draft Room / Mock Draft team grades, 🏆 rankings, roster-card PPG, and the
  "% of own optimal" denominator now sit on the realized scale in every mode —
  a mock winner should grade ~128-138 like real top teams do.
- Cross-position keeper/lineup tradeoffs (which use wkPts sums) now weigh QBs
  correctly instead of on the compressed scale — a 6-pt-league QB was previously
  undervalued ~24% in weekly terms.
- Auction values, Exp $, My Max, walk-away: unchanged (they run on raPts/effPts).

*Harness: scratchpad mode_gap.js (reproduces before/after); actuals from
outputs/espn_standings.json; calibration inputs from nflv_ffa_league + data.js.*
