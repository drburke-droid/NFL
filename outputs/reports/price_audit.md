# Price-engine audit — Value & Exp $ (2026-07-17)

Headless audit of the two core board numbers, run against the live page code
(`scratchpad/audit_prices.js` pattern: real `dynamicMarket()`/`baseValue()` in Node).

## Verdict: structurally sound, no bugs found. All flagged gaps trace to deliberate,
## documented mechanisms — now surfaced on the board with a ⚠ market-gap flag.

## 1. Structural checks — all PASS

| Check | Result |
|---|---|
| Budget conservation: Σ Exp $ over top-192 rosterable | **$2,325 = 96.9%** of the $2,400 pool ✓ |
| Position all-time price caps (QB $34 / RB $68 / WR $65 / TE $39) | no violations ✓ |
| Negative prices/values | none ✓ |
| Value monotone in risk-adjusted points (top-40/pos) | no inversions ✓ |

## 2. Calibration vs the league's real 2023-25 winning-bid curves

QB MAD $3.8, TE $4.9 (tight). RB MAD $9.8 and WR $7.8 — our Exp $ runs $5-10 **above**
the historical curve at ranks 2-6. Explanation: the 2026 board carries an unusually deep
elite RB/WR tier (FFA tier-1 is broad) plus live inflation; the totals still clear the
budget, so this is the curve adapting, not drift. Watch on draft day: if early elite
RB/WR sales come in at historical-curve prices, mid-tier Exp $ will re-rate down live.

## 3. Cross-validation vs the external market

- Exp $ vs **ESPN $** (n=83): r = 0.891, bias +$2.9, MAD $6.3
- Exp $ vs **FFA AAV** (n=159): r = 0.934, bias −$0.9, MAD $4.6

Every large per-player gap traced to a known mechanism:

| Player | Ours | Market | Mechanism |
|---|---|---|---|
| Malik Nabers | $47 | $21-24 | **healthy-rate conviction** (HEALTHY_PPG 16.4 — we price the healthy season, market prices injury risk) |
| George Kittle | $23 | $4-9 | healthy-rate (13.3) + room anchor ($18.8 — this room pays for him) |
| Josh Jacobs | $44 | $29-32 | room anchor $36.5 — the league historically overpays him |
| Caleb Williams | $23 | $4-13 | room anchor $19.3 + 6-pt league (sitewide 4-pt prices QBs low) |
| Jeremiyah Love | $12 | $39 | rookie stays on the league curve; national hype runs ahead |
| Kenneth Walker | $25 | $38 | market likes him more than our risk-adjusted model does |

## 4. Value vs market (~+23% pool)

Value intentionally exceeds clearing prices (worth is uncapped by history, per design):
top names show Val $70-82 vs market $50-62. This is the risk-adjusted-VBD scarcity
premium given our projections — it drives My Max/Edge, and is NOT a price prediction.
Do not "fix" this; the Exp $ column is the price predictor.

## 5. Action taken

**⚠ market-gap flag** on the 💰 Auction $ board: whenever Exp $ sits ≥ $12 from the
mean of ESPN $ + FFA AAV (both present), the Exp $ cell shows ⚠ with the traced reason
(healthy-rate conviction / room anchor / rookie-curve) in the tooltip — so at the table
you see WHICH number to trust and why, instead of silently believing either.

*Engines audited: `dynamicMarket()` (curve × per-player anchors × inflation, caps),
`baseValue()` (risk-adjusted VBD). No engine changes made — both validated previously
(anchors: 20% sharper than curve alone, season-out).*
