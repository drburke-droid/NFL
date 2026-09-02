# Is filling the WHOLE roster with below-position-average value players optimal? (No — anchor it.)

**Question** (2026-09-02): build on the skill-vs-price work — if top performers
regularly go below their position's average auction rate, is a roster made
*entirely* of such players the optimal construction?

**Backtest** (`scripts/roster_strategy_study.py`): 2016–2025, league-scored FFA
projections + FFA AAV as market prices (ex-ante only), 12-team/$200 structure
(12 skill players, $190, starters QB/2RB/2WR/TE/FLEX). Each strategy greedily
maximizes projected lineup points under its price rule (+2-opt swaps); scored on
**realized** points of the ex-post optimal lineup (so bench depth counts).

## Results (mean realized lineup points, 10 seasons)

| Strategy | Rule | Mean pts | Spent | Seasons won |
|---|---|---|---|---|
| value_fill | every player ≤ position-avg price | 1436 | **$100** | 1 |
| value_125 | ≤ 1.25× position avg | 1493 | $119 | 2 |
| optimizer | no constraint | 1495 | $190 | 2 |
| stars_scrubs | ≥$40 or ≤$5 only | 1495 | $190 | 2 |
| **anchor_value** | **value rule + 1–2 anchors ≥$50** | **1588** | $190 | 1 |
| mid_band | $6–25 only | 1560 | $184 | 2 |

Paired vs value_fill: anchor_value **+153 pts/season (p=0.084)**, mid_band +124
(p=0.064), others +58–60 (ns). n=10, so treat p-values as directional.

## Why pure value-fill loses

1. **It can't spend the money.** Position-average prices are ~$6–16 (2025: QB $9,
   RB $16, TE $6, WR $13). Twelve players × that cap ≈ **$100 of $190** — the
   strategy structurally strands half the budget, and auction dollars don't bank.
2. **Hit rate rises with price.** Same force as the dart price-band study
   ($10–14 upside darts hit 51% vs 22% at $1–2): below-average-price players are
   below average partly for real reasons. A full roster of them has a high floor
   and no ceiling — value_fill's best season (1823) is the second-worst "best
   season" of any strategy.
3. The depth advantage doesn't save it: realized scoring used the ex-post best
   lineup from all 12 players — the construction most favorable to a deep value
   roster — and it still finished last.

## What IS optimal

**anchor_value** — spend the surplus on 1–2 true anchors (≥$50), then apply the
value rule everywhere else (2025 example: Chase $62 + Bijan $60 + Pollard $15,
Flowers $10, Nix $9, Jeudy $8, Njoku $6 …). Best mean by ~90 pts over the
unconstrained optimizer. This is precisely the shape of the 2026 playbook
(elite keepers as the anchors + mid-price fill engine + $1 darts), now validated
at the roster-construction level. The mid_band result (2nd) says the fill engine
should lean $6–25, not $1–5 — consistent with paying up for upside profiles.

**Caveats**: 10 seasons; single-roster simulation (no opponent budgets/room
inflation — refusing star prices in a real room also inflates mid prices);
prices are FFA AAV, not your league's; K/DST ignored ($2 reserved). The value
signal itself (alpha-skill WR residual, `skill_vs_adp.md`) remains a *player
selection* edge inside the value tier — this study is about *budget shape*.
