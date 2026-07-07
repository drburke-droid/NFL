# Per-player price anchors for Exp $ (2026)

`scripts/build_price_anchor.py` → `docs/price_anchor_2026.js`, consumed by
`dynamicMarket()` in the draft tool. Goal: Exp $ predicts what THIS league will pay
for each player, not just what the positional rank curve says.

## Model

Ridge regression on the league's own winning bids (2024–25 targets, 189 returning-player
buys), features all knowable before the draft:

- **last year's price** (auction bid, or keeper cost if he was kept) — the league keeps
  ~45¢ of every prior dollar (see `yoy_auction_repricing.md`)
- **prior-season PPG + games** (representativeness), **career-best PPG** (3-yr window)
- **age** + an age-28+ decline term
- **position** (QB and TE discounts — one-starter positions)

**Finance calibration**: 2024 was an 11-team league. All seasons' bids are rescaled to
dollars-per-team-at-auction (2023 $196 → ×0.765, 2024 $140 → ×1.074, 2025 $150 = ref)
before fitting — and the same normalization now feeds the bid curve in
`predict_keepers.py`, so the smaller-league year no longer deflates the curve.

## Validation (season-out: train one keeper-era year, test the other)

| blend weight W | 0 (curve only) | 0.4 | 0.6 | **0.8** | 1.0 (model only) |
|---|---|---|---|---|---|
| MAE ($) | 8.63 | 7.38 | 7.03 | **6.89** | 7.02 |

Blending the anchor over the rank curve at **W = 0.8 cuts error 20%**. Shipped as
`PRICE_ANCHOR_W`.

## Anchor domain (who gets one)

Anchors apply only where the model has information: **non-rookies whose 2025 was
representative (≥8 games) and whose 2026 projection isn't far above their 2025 play**
(proj − prior ≤ 3 PPG). Injury returns, breakouts, and rookies are priced by the live
rank curve — the room prices those forward, and a backward-looking anchor would
underprice them (e.g. Malik Nabers). 379 players anchored, 153 left to the curve.

## Integration

`dynamicMarket()` blends per player *before* demand gating and room inflation:
`cv = 0.8 × anchor + 0.2 × curve[slot]`, then the existing bench ramp, keeper-inflation
rescale, and Mock Draft owner ceilings apply unchanged. `expIf()` (keeper cards) uses the
same blend, so cards and pool stay on one engine.

Full-sim check vs history after integration: bands $50+: 7 (real 4–11), $25–49: 19–22
(real 15–24), $10–24: 24–26 (real 26–45), tops QB $28 / RB $64 / WR $62 / TE $26;
leftover $66–83 per room (the real league left $88–117).

Regenerate after each data refresh: `python scripts/build_price_anchor.py`
(after `predict_keepers.py`).
