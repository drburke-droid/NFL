# How the league reprices the same player year over year

**Question** (2026-07-07): for players bought at auction (non-keeper) in consecutive drafts,
how does the winning bid move with performance in between? E.g. PlayerX QB: 22 ppg in 2022 →
$28 in 2023, plays 19 ppg → $17 in 2024, plays 20 ppg → $18 in 2025.

**Data**: all 560 winning bids on file (`outputs/espn_drafts.csv`, 2023–25), keepers excluded;
real PPR PPG per season from nflverse (`db/nfl_odds.db`, ≥4 games). 149 same-player
consecutive-draft pairs (81 from 2023→24, 68 from 2024→25).

## Headline: the next price is ~half anchor, ~half repricing

```
next bid = -5.5 + 0.45 × last bid + 0.83 × PPG-between     (R² = 0.47, n = 149)
price-only:  R² = 0.42 (corr 0.65)     PPG-only:  R² = 0.22 (corr 0.47)
```

The room keeps about **45 cents of every last-year dollar** and adds about **$0.83 per PPG**
played in between. Last year's price is the single strongest predictor of this year's price —
the league anchors — but performance moves the number materially.

## Repricing is asymmetric: decline is punished ~2.5× harder than improvement is rewarded

| PPG change between drafts | n | avg $ change | median | avg % (buys ≥$5) |
|---|---|---|---|---|
| improved 2+ PPG | 29 | **+$4.6** | +3 | **+23%** |
| flat (±2) | 68 | −$2.4 | −1 | −6% |
| declined 2–5 | 30 | **−$11.5** | −9 | **−58%** |
| declined 5+ | 11 | −$17.3 | −15 | −57% |

A mild decliner loses more than half his price; a genuine improver gains only a quarter.
(The user's PlayerX — 22→19 ppg, $28→$17 = −39% — sits right on this curve, as does the
year-3 flat rebound 20 ppg → $18.)

## What the next draft pays per production tier

| PPG between drafts | n | next-draft avg | median | (prev draft avg) |
|---|---|---|---|---|
| 18+ | 16 | $20.4 | $18 | ($26.2) |
| 15–18 | 36 | $21.6 | $20 | ($22.3) |
| 12–15 | 47 | $12.7 | $10 | ($19.3) |
| 9–12 | 32 | $8.4 | $6 | ($10.1) |
| <9 | 18 | $3.9 | $2 | ($7.0) |

Note 18+ PPG earns *less* than 15–18: the 18+ repeat-auction group is old stars
(Hill, Adams, Kelce types) — the age discount is visible in your league's bids.

## Almost everything re-auctioned gets cheaper

- Overall $ retention: **79%** ($2,578 → $2,043). 56% of players got cheaper; only 34% pricier.
- **$20+ buys retain just 64%** — 44 of 51 got cheaper the next year.
- By position: QB 74% · RB 78% · WR 82% · TE 75% (avg QB $15→$11, TE $11→$8).

**Why: keeper survivorship.** Anyone who breaks out young gets *kept*, not re-auctioned. The
repeat-auction pool is what the keeper skim leaves behind — aging vets and disappointments —
so "he's available again" is itself bad news, and the pool decays ~20%/yr on average.

## Example chains (PPG before → $ → PPG between → $)

```
WR Justin Jefferson  2023: 21.7 → $75 | 20.2 ppg → $52   (elite, aged one year: −31%)
WR Tyreek Hill       2023: 20.1 → $58 | 23.5 ppg → $57   (improved: price held)
WR Tyreek Hill       2024: 23.5 → $57 | 12.8 ppg → $32   (collapsed: −44%)
WR A.J. Brown        2023: 17.6 → $42 | 17.0 ppg → $55   (flat + market up: +31%)
WR A.J. Brown        2024: 17.0 → $55 | 16.7 ppg → $48   (flat: −13%)
QB Josh Allen        2023: 24.7 → $44 | 23.1 ppg → $29   (QB market compressed −34% despite elite play)
RB Josh Jacobs       2023: 19.3 → $41 | 13.9 ppg → $37 | 17.2 ppg → $56  (down then breakout)
TE Travis Kelce      2023: 18.6 → $51 | 14.6 ppg → $21   (age cliff: −59%)
RB Austin Ekeler     2023: 21.9 → $62 | 13.2 ppg → $6    (the cautionary tale)
Jumps: Drake London $7→$37 · Mike Evans $8→$37 · Nico Collins $2→$28 · K.Walker $13→$33
```

## Implications

1. **The mock-draft bots already capture most of this**: rank-on-the-curve pricing reprices
   players by current value, and the QB compression (Allen $44→$29) is exactly the sticky
   QB ceiling now in `ownerBid`.
2. **The 45% anchor is exploitable**: the room over-pays declining ex-stars (anchor holds
   half their old price for ~2 fewer PPG) and under-pays mild improvers (+23% for +2 PPG).
   Value hides in the "flat 15–18 PPG at $20" tier — the market's most honest price point.
3. If we ever want the tool's Exp $ to be sharper on returning vets, blending
   `0.45 × last-year price + 0.83 × current-proj PPG − 5.5` with the curve price is the
   empirically fitted anchor model.

*Method: scratchpad `yoy_analysis.py`; plain OLS, no outlier trimming; names matched on
normalized (suffix-stripped) name; PPG requires ≥4 games.*
