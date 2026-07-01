# 2026 Auction Draft Plan — Saja Boys (post-keeper simulation)

Reproducible: `scripts/draft_plan_2026.py` (execs `predict_keepers.py`, locks each team's predicted
top-3 +value keepers, re-ranks the remaining pool on the league's positional bid curves, layers owner
tendencies + budgets). **User override: Brock Bowers KEPT by team 7** (engine said 'redraft' on a $2
technicality). 29 keepers league-wide, **~$2,026 in the room**.

## UPDATE — Bowers kept (2026-07-01)
Kittle becomes the **lone elite TE** (val 36, exp $28) with FOUR TE-hungry owners (State your name
×1.76/$176, 3AM ×1.73, Ugh ×1.45, Maple ×1.38) chasing one target → expect him well past $28.
**TE becomes punt-first**: take Kittle only ≤**$32** (still +4); otherwise Kelce ~$17–19 (val 18) /
Loveland ~$14 / LaPorta ~$12 and shift the saved ~$15 to WR2 (BTJ ≤$24) or a 3rd value RB.
Path C (Chase pivot) gets MORE attractive since the TE punt is likely anyway. Everything else holds —
edge board is now RB-dominated: Bijan +12, Hall/Swift +8, Javonte/Love +7, Jeanty +6.

## My situation
Keeps: **Josh Allen $32** (val 48), **Rashee Rice $32** (val 47), Alec Pierce $1 (val 6) → **$135 left,
13 slots** (need RB×2, WR2, TE, FLEX, K, DST + 6 bench). QB + WR1 are set; **zero RBs kept**.
Reserve ~$12 for the last 9–10 slots → **~$120 core budget for 4–5 players**.

## Market structure (post-keeper pool)
- **Positive edge is concentrated**: Bowers +16, Bijan +12, Breece Hall +8, Swift +8, Kittle +8,
  Javonte +7, Jeremiyah Love +7, Jeanty +6, Saquon +4, TreVeyon +4, Kamara +4, Chase +2.
- **WR mid-tier is a value desert**: every WR after Chase is negative edge (Nabers −12, London −13,
  Adams −10, Sutton −9, McConkey −8...). Consistent with the trend sweep: WR is deep — don't pay.
- **QB pool all negative edge** (Burrow −8, Hurts −10) — irrelevant to me (Allen kept), useful as bait.

## Rival pressure
- **RB threat**: Rectify this ($196 budget, RB×1.32), Micahroni ($154, ×1.31), Spitting away ($184, ×1.25).
- **TE bidding war likely**: State your name (×1.76, $176), 3AM (×1.73), Ugh (×1.45), Maple (×1.38) all payTE.
- **WR**: Paul's Perfect Team (×1.51, $174) pays up — feed him the negative-edge WRs.
- 8 of 11 rivals are eliteQB buyers → nominate Burrow/Hurts/Mahomes early to drain budgets.

## Conditional plan (caps = still +edge at that price)
**PATH A — Bijan anchor** (preferred): Bijan ≤**$65** (exp 59, val 71) → Bowers ≤**$26** (exp 19, val 35;
TE war likely) → RB2 from {Swift ~$22, Javonte ~$23, Breece ~$26} ≤ val → WR2 ~$12–18 (McLaurin ≤$18 /
BTJ if he slides) + FLEX dart (Kamara ≤$16 / Dobbins ~$15). ≈ $60+24+24+15+10.

**PATH B — Balanced (Bijan >$65)**: TWO of {Breece ~$26, Jeanty ≤$32, Saquon ≤$34} + ONE elite TE
(Bowers ≤$26 / Kittle ≤$28, val 36) ≈ $85 → WR2 Brian Thomas Jr. ≤$24 or McLaurin ≤$20 → FLEX
TreVeyon ~$18 / Love ~$20 (+7, rookie).

**PATH C — Elite-WR pivot (RB prices explode)**: Chase ~**$53** (val 55, the ONLY +edge WR) beside Rice
→ punt TE (Kelce ~$17 val 18 / Loveland $14 / LaPorta $12) → THREE value RBs (Swift $22 + Javonte $23 +
Love $20 — all medium dual-threats, the validated durable archetype). ≈ $53+17+65.

## Scenario: Chase + Bijan double-anchor (both are in the pool; Chase is MY unkept player)
At expected prices ($60+$53=$113 of $126 core) the pair is +13 edge — IDENTICAL to Path A, but with
$13 left the RB2/TE become $8/$3 darts = two weekly starting holes, and the tournament/sim evidence
says forced stars-and-scrubs finishes LAST (10.1/12; extreme S&S never top-3). Only the DISCOUNT
branch flips it: Bijan ≤$60 AND Chase ≤$45 → +23 edge, best available basket. Decision rule: land
whichever comes up first at its Path price (Bijan ≤$60 → Path A; Chase ≤$50 → Path C), then bid the
second ONLY at a ~$8-below-market discount (Chase ≤$45 / Bijan ≤$55); otherwise complete the single-
anchor path. Keeping Chase at $61 is strictly worse than re-buying at ~$53 (engine's 'pass' stands).

## Do-not-pay list (our corrected model dings aging/moving stars — validated +0.025 MAE)
Derrick Henry $54 (val 45, age 32) · Josh Jacobs $51 (val 42) · CMC $48 (val 42, age 30) ·
Nabers $49 (val 37) · London $44 (val 31) · Adams $21 (val 11) · Metcalf/Waddle/Evans/Diggs teens
(vals $4) · Justin Jefferson above $25 (our val 25 vs market 31).

## Nomination strategy (I nominate what I DON'T want, at $1)
1. Elite QBs first (Burrow/Hurts/Mahomes — 8 rival eliteQB buyers, all negative edge).
2. Name-brand fades (Henry, CMC, Jacobs, Nabers, London) — soak the RB/WR-hungry budgets.
3. Kickers/DSTs when rivals still hold money late.
Hold my targets (Bowers, value-tier RBs) as long as possible — prices fall as budgets drain.
