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

---

## v2 recheck (2026-09-02): rerun with the cheap-stars CLASS definition

The overnight `cheap_stars.md` study sharpened the target: not "below position
average" but **50–100% of the positional mean AND projected starter-tier**
(14.2% star rate, ~$10, ~18 exist/season). Rerun with class-based strategies,
same engine, positional means computed the study's way (top-168 pool), every
roster backfilled to 12 with $1 bodies:

| Strategy | Mean pts | vs value_fill | p |
|---|---|---|---|
| **class ×4 by projection + free rest** | **1617** | **+176** | **0.049** |
| mid_band ($6–25) | 1604 | +162 | 0.084 |
| anchor_value | 1572 | +131 | 0.160 |
| class + anchors ≥$50 | 1563 | +122 | 0.322 |
| class ×4 position-diverse + free rest | 1552 | +110 | 0.131 |
| **class_fill (whole roster of class)** | **1506** | +65 | 0.160 |
| optimizer / stars_scrubs | 1495 | +54 | ns |
| pure value_fill | 1441 | — | — |

Three conclusions:

1. **The class upgrades the strategy but doesn't change the verdict**: a whole
   roster of class players (1506, spends only $121 — the class is priced $6–16,
   so 12 buys still strand $69) beats naive value-fill but stays mid-pack.
2. **The optimal shape is exactly the cheap-stars report's advice**: force ~4
   class buys (~$40) and spend the rest on anchors — best mean, and the only
   strategy to clear p<0.05 against pure value-fill on 10 seasons.
3. **The winning seed is QB-heavy, and that's not a bug**: picking the 4 class
   buys by raw projection lands on 3–4 cheap starter-tier QBs (2025: Nix $9,
   Kyler $7, Purdy $6, Goff $6) and beats the position-diverse version by 65
   pts/season — the 6-pt-pass-TD league's compressed QB pricing (class QB star
   rate 18.2%, the study's best) rewards stacking two-plus of them. Consistent
   with the positional-fill finding that cheap QB darts are this league's most
   efficient fill.

**Draft-night rule**: anchors first (keepers + one big bid), then buy 3–4 class
players — starter-tier projection at 50–100% of what the room is paying at the
position, checked against live Exp $ — leaning QB when two of the cheap
starter-tier QBs are still on the board, then $1 darts. Whole-roster value-fill
remains a trap: it can't spend the budget.
