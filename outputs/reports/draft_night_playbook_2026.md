# 🏈 Draft Night Playbook 2026 — Saja Boys

Keeps LOCKED: Burrow $6 · Rice $32 · Chase $61 = $99. **$101 for 13 spots.**
(Validated best of all 469 keep sets by bot Monte-Carlo; CMC thrown back.)
All prices = Blend-mode Exp $ with league keeps locked. Playoffs = top 6; QB
spend share is this league's one winning-roster signal — Burrow covers it.

## Script A — the Bijan pursuit
- He sells in the FIRST 30 nominations (every $40+ player 2023-25 did). Expect $61.
- **Bid to $65, hard walk-away $68** (MC: $55 dominates, $65 break-even, $75 worse
  than never buying). My sheet max $89 is the VALUE ceiling, not the plan ceiling —
  past $68 the roster math loses more at RB2/flex than Bijan adds.
- Win at <=$60: add Bucky $16 + Kelce $8 + McLaurin/Odunze ~$9-15 flex.
- Win at $65: RB2 becomes a $2-7 flyer (Gainwell tier) + Kelce/Andrews $5-9.

## Script B — Bijan escapes (or >$68)
The solver's roster: **Bucky Irving $16 (walk-away $17) + Swift $20 + Kelce $8 +
A.J. Brown $41 flex** + $1 K/DST + 7 darts. Alternatives at fair price:
- CMC buy-back exp $56 (you know his health best; max $59)
- Hall exp $26 (max $44 — big edge) · Hampton exp $37 (max $43) · Warren $11
- Flex board: McMillan exp $25/max $31 · Egbuka $25/$28 · AJB $41
- **Bucky is the one must-win nomination** (26% keep risk, FFA firmed him to $15)

## Nomination strategy (from this league's 2023-25 timing)
1. Noms 1-30: throw OTHER teams' stars out (they pay 1.03-1.04x; half the money
   drains). Do NOT nominate your value targets. Bijan comes to you regardless.
2. Noms 61-90: the RB lull (0.75x) — buy Swift/Hall/Hampton here. WR holds 1.02x
   until nom ~90 — never chase WR early; your flex buy waits.
3. Noms 91+: everything at 0.45x — TE and flex value if you stayed patient.
4. $1-3 endgame starts ~nom 91: the dart board empties noms 91-140. Have the
   tier list open.

## TE doctrine
Kelce at $8-12 = buy (walk-away $12). Otherwise Goedert/Likely/Andrews $1-6.
**Never pay the model's Kittle price** — FFA/ESPN have him TE11 ($4-8), his
wks-14-17 slate is the worst in football (SF TE z -1.0), and the sheet max is $4
in blend. If the room lets him go <=$8, fine; at $15+ he's someone else's problem.

## Bench (7 slots, ~$8-10 total) — hybrid tier order
T1 buzzing heirs: **Bigsby $2-4** (worth $5 — the one validated synergy cell),
Corum $3-5, Keaton Mitchell $2. T3 young: Adonai Mitchell, Golden, Sampson,
Legette, Worthy $1-2. Half the bench RB-only. Every hit = a $6 keeper in 2027
(waiver rule). Skip vets — the old-cheap-FLASH market is dead since 2022.

## Advisories to watch (auto-shown in 🎯 Your Plan)
- 📅 bye collisions vs your keeps (Burrow CIN wk6, Rice KC wk5, Chase CIN wk6 —
  avoid stacking more CIN/KC byes at the same position)
- 🥶 Rice's KC WR playoff slate is -0.5; prefer flex targets with neutral+ slates
- Δ$ price-disagreement tags: engine-low players (Jer. Love, KWIII) = let the
  room overpay; engine-high (Nabers-type) = quiet value

## Night-before checklist
1. python scripts/fetch_wiki_buzz.py refresh
2. python scripts/late_breakout_final.py && python scripts/bench_darts_2026.py
3. python scripts/build_draft_tool.py  (+ re-import ESPN salary CSV if fresher)
4. Re-run fetch_espn.py (final rosters/keepers) -> predict_keepers -> keeper_likelihood
5. On the tool: tick your 3 keeps in Mock Keepers, pin Bijan + Bucky, set
   what-if $65 on Bijan and read the plan; Blend mode ON.
