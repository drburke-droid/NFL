# Game-to-game prop model vs Vegas — are we +EV? (2026-07-04)

Reproducible: `scripts/prop_ev_model.py` (+ backtest pickle `outputs/prop_ev_backtest.pkl`).
Setup: per-market LGBM models (reception yds / receptions / rush yds / pass yds / rush att /
completions) on 57K player-games with rolling stats+xFP+EPA+shares, prior-season archetype, opponent
defense-allowed-to-position, defense-vs-archetype, player-vs-def-style, implied total/spread.
Walk-forward (train < test season; test 2024+2025, 26K matched closing markets, ≥3 books, de-vigged
median consensus, pushes dropped).

## Headline: NO — not reliably +EV vs closing lines
| test | result |
|---|---|
| Model MAE vs the closing LINE's MAE | **line wins all 6 markets** (e.g. rec yds 20.7 vs 19.5) |
| Calibration of our P(over) | poor — deciles 0.11→0.71 all realize ~44–49% |
| Our 6%-edge bets, best-of-20-books price | +ROI on unders (+2.6%), **overs −9.9%** |
| Blind-under benchmark (no model), best price | **+1.4%** — most of the "edge" is NOT ours |
| Our unders at **median** book price | **−1.0%** (blind −1.9%) — line shopping was the profit |
| Season persistence (best price) | 2024 +0.5% vs 2025 +4.2% — unstable |

Three structural facts explain everything:
1. **Overs only hit 47.2%** while de-vigged consensus says 49.7% — books shade lines into public
   over-demand, so unders carry a small structural edge *before* vig.
2. **Best-of-20-books ≈ +3-4% price improvement** vs the median book — the entire realized profit.
   At realistic (median) execution the strategy nets below vig.
3. Our model adds only a **mild ordering** on unders (lean ≥15%: +2.9% best-price vs +1.4% blind) and
   its over signals are actively harmful (normal-tail inflation) — the closing line is sharper than
   any stats-only model here, consistent with the weekly-mean saturation finding (weekly_v2).

## What survives (weak pockets, thin n)
- **Rush attempts unders** (volume market): +1.9% at MEDIAN price, n=1347 — most plausible real pocket.
- Pass yds unders +6.7% at median but n=130 (noise until proven).
- Practical rule if betting anyway: **volume-market UNDERS only, only after shopping ≥5 books** —
  the margin is in the PRICE, not the prediction.

## The real next edge to test (data already in hand)
We hold ~547 snapshots per market: the sharp play is beating the **OPENING** line, not the close.
Test: our model (or even just the eventual closing consensus) vs FIRST-snapshot lines — if model-vs-open
edge predicts line MOVEMENT, betting early is classic +CLV. Unbuilt; flagged as the follow-up.
