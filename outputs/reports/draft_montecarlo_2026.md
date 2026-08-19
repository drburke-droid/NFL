# Draft-night Monte-Carlo — what is Bijan actually worth to THIS roster?

Ran the draft tool's own Mock Draft engine headlessly (the per-owner bots fitted on
2023-25 bids: positional leans, fandom stretches, price enforcement, endgame
surplus-clearing), 20 full simulated auctions per policy with ±15% deterministic
jitter on every bot max. Setup: league predicted keeps locked, my keeps = Burrow $6 /
Rice $32 / Chase $61, my bot bids the sheet's suggested maxes for everything else.
Metric = my final starters' risk-adjusted points (best 1QB/2RB/2WR/1TE/1FLEX).

| policy | mean | p25 | p75 | min |
|---|---|---|---|---|
| skip (Bijan kept by team 1) | 1136 | 1116 | 1157 | 1110 |
| **Bijan bought $55** | **1198** | 1190 | 1203 | 1181 |
| Bijan bought $65 | 1143 | 1136 | 1160 | 1125 |
| Bijan bought $75 | 1100 | 1087 | 1114 | 1061 |

**Read**: at $55 Bijan dominates every world (the p25 of $55 beats the p75 of skip).
At $65 he is roughly break-even vs letting him go. At $75 the roster is WORSE than
never getting him — the money he drains from RB2/FLEX/TE costs more than he adds.

**Walk-away number: ~$68.** "No matter the cost" is disproven by the room's own
bots; past the high-60s, let A Useless Johnson keep him and pivot (Hall/Swift/
Hampton + Kittle at the freed budget beat the $75-Bijan worlds by ~40 pts).

Reproduce: harness2.js + tail9.js pattern (headless eval of docs/index.html with
data files; jitter seeded per sim). The engine is `mdSim(null)` in docs/index.html.
