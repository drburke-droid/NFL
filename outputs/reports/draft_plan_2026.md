# 2026 Auction Draft Plan — Saja Boys (post-keeper simulation)

Reproducible: `scripts/draft_plan_2026.py` (execs `predict_keepers.py`, locks each team's predicted
top-3 +value keepers, re-ranks the remaining pool on the league's positional bid curves, layers owner
tendencies + budgets). Assumes 28 keepers league-wide ($355 of $2400), **$2,045 in the room**.

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

## Do-not-pay list (our corrected model dings aging/moving stars — validated +0.025 MAE)
Derrick Henry $54 (val 45, age 32) · Josh Jacobs $51 (val 42) · CMC $48 (val 42, age 30) ·
Nabers $49 (val 37) · London $44 (val 31) · Adams $21 (val 11) · Metcalf/Waddle/Evans/Diggs teens
(vals $4) · Justin Jefferson above $25 (our val 25 vs market 31).

## Nomination strategy (I nominate what I DON'T want, at $1)
1. Elite QBs first (Burrow/Hurts/Mahomes — 8 rival eliteQB buyers, all negative edge).
2. Name-brand fades (Henry, CMC, Jacobs, Nabers, London) — soak the RB/WR-hungry budgets.
3. Kickers/DSTs when rivals still hold money late.
Hold my targets (Bowers, value-tier RBs) as long as possible — prices fall as budgets drain.
