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

## Opening-line / CLV study (`scripts/prop_open_clv.py`) — untestable with current capture
Ran the opener test on 17K matched markets. Result: our "openers" are NOT openers — the collector's
FIRST prop snapshot is median **2.1 hrs before kickoff** (p90: 9 hrs), so we only captured the final
sliver of each market. Consequently lines "move" in only 1–13% of props (receptions 1%), model-edge-
vs-opener has ZERO correlation with movement (+0.005), prob-CLV is 0.000, ROI at open ≈ at close
(unders +2.9% best-book / −0.7% median; overs −13%), and even cheating steam-follow loses (−2.6%).
**Verdict: CLV hypothesis untested, not disproven** — real prop openers post Tue–Wed and sharpen over
days; we never saw that window.

## Market-anchored follow-up (`scripts/prop_anchored.py`) — the market prob IS improvable, by ONE number
User idea: use the market prob as baseline, learn improvements on top (the FFA-anchored pattern).
Train 2024 → test 2025 (9,027 props), Brier ladder:
| model | 2025 Brier |
|---|--:|
| M0 raw de-vigged market | 0.24829 |
| **M1 = market − 2.2pp global over-bias shrink** | **0.24756** ✅ best |
| M2 market-only logistic | 0.24769 |
| M3 anchored LGBM + ALL our player features | 0.25332 ❌ worse than raw market |
The market probability is beatable — but the ENTIRE improvement is the constant over-bias correction
(estimated on 2024, held on 2025). Every player-level feature on top is overfit noise (M3 deviates
>5pp from market on 47% of props, all of it wrong). ROI with the improved prob: **+5.3% at best-of-20-
books (n=2,404, 2025)** but **−1.5% at ~median book** — the profit is price shopping, not prediction.
Final law of this market: P*(over) ≈ novig − 2.2pp; edge comes from EXECUTION (books × timing), never
from out-modeling the closing consensus.

## Sharp-book hunt (`scripts/sharp_book_hunt.py`) — Pinnacle-truth vs DK: FAILS at the close
2024 sharpness (Brier of de-vigged P(over), consensus-line props): **Pinnacle 0.2463** best among books
alive in 2025 (also lowest hold 5.4%) — but the whole field sits 0.2460-0.2477: at CLOSING, every book
has converged on the same consensus; there is no persistently dumb book. Hunting DK in 2025 with
Pinnacle's prob as truth (same-line props, n=4,647 overlaps): EV>0% → **−5.0%** (n=597), EV>2% →
**−8.7%** (n=192), EV>4% +1.1% on n=52 (noise). Killer diagnostic: the SAME bets placed at Pinnacle's
own prices lose −17% — when DK and Pinnacle disagree at close in our capture, the disagreement
resolves in DK's favor, i.e. we're selecting Pinnacle's stale outliers, not DK's weak lines. The
classic Pinnacle-origination strategy needs LIVE mid-week odds; at T-2h closing snapshots it's dead.

## ACTION for the 2026 season (the one real path to +EV here)
Change the Odds API fetch cadence to snapshot player props **from Tuesday onward** (e.g. 2-3x/day
Tue–Thu, then closing). By mid-season we'd hold true open→close trajectories to test: (a) does our
model beat Tuesday lines? (b) does model-vs-open edge predict movement (CLV)? Books cap prop stakes
anyway, so this is the only framing where a stats model plausibly wins. Until then: no prop betting
edge exists in our data beyond volume-market unders + aggressive line shopping (≈breakeven).
