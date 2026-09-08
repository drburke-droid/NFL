# Model_Burke vs the props market — 2025 season backtest

**Question** (2026-09-08): the user's weekend package (`model_burke`, walk-forward
calibrated distributions) — how would it have done on player props last season?

**Design**: the model is a *correction* architecture, so the book's own closing line
is the baseline it corrects. Per market: one row per player-week 2023–2025;
`baseline_proj` = consensus closing line (median of each book's last pre-game
snapshot; median 11 books/row, 93–97% name match), target = the actual stat
(`nflv_weekly`), features = the package's own lagged usage builder + closing
spread/total + wind. Betting: 2025 season only, P(over) read off the calibrated
quantiles (piecewise-linear CDF), edge vs the de-vigged two-way price, graded at
median closing prices (pushes void; DNPs excluded = books void them too).

## Can it out-predict the closing line? (2023–25 pooled, walk-forward)

| market | n scored | model MAE | line MAE | weekly win rate vs line | 80% coverage |
|---|---:|---:|---:|---:|---:|
| receptions | 5,601 | **1.460** | 1.486 | **69.2%** of 39 wks | 0.846 |
| reception yds | 6,193 | 19.152 | 19.165 | 58.5% of 41 | 0.811 |
| rush yds | 1,170 | 17.807 | 17.809 | 43.8% of 16 | 0.812 |
| pass yds | 0 | — | — | — | — |

Pass yds never engaged: 1,668 rows total < the package's `min_train=2000`. (QB is
also the position the README already concedes.) Conformal coverage lands near
0.80–0.85 everywhere it runs — the calibration layer does its job on stat lines,
not just fantasy points.

## Did it make money in 2025?

| market | edge ≥ | bets | win% | ROI |
|---|---|---:|---:|---:|
| reception yds | 3% | 484 | 50.5% | −3.6% |
| reception yds | 5% | 185 | 52.2% | **+1.4%** |
| reception yds | 8% | 54 | 55.6% | **+14.2%** |
| reception yds | 12% | 10 | 70.0% | +68.7% |
| receptions | 3% | 2,410 | 47.2% | −2.6% |
| receptions | 5% | 1,941 | 47.1% | −1.4% |
| receptions | 8% | 1,345 | 45.8% | −1.8% |
| rush yds | 8% | 17 | 47.1% | +16.2% |
| **pooled @5%** | | 2,152 | 47.4% | **−1.2%** |

Blind benchmarks at the same prices: all-overs ≈ −10%, all-unders ≈ −2.5% in every
market (books shade the over).

## The verdict, in three parts

1. **Bet-everything loses (−1.2%)** — dominated by receptions, where the model fired
   on 60–70% of all available props. That's not signal; that's the model's median
   bias (−0.54 receptions, MAE-optimal but probability-wrong) reading as a permanent
   "under" edge on a **discrete** stat. Its 1,679 unders returned −2.3% ≈ blind
   unders (−2.7%): no skill, just the skew artifact. The README's own warning —
   "quoting the median where the mean belongs is the most common way to look
   wrong" — applies to P(over) too.
2. **The yardage edge looks real but small**: reception yds shows the signature of
   genuine signal — ROI rises monotonically with stated edge (−3.6% → +1.4% →
   +14.2% → +68.7%), i.e. the model's conviction is informative. But the money
   lives in ~50–190 bets/season, and the 8%+ cells are n=54 and n=10 — one season,
   wide CIs. Treat as promising, not proven.
3. **Receptions is the model's best *prediction* market and worst *betting* market**
   at once — it beats the closing line in 69% of weeks on MAE while losing money.
   Fix before trusting: a discrete-aware P(over) (integer mass around x.5 lines)
   and centring probability on `Model_Burke_mean`, not the median.

## Caveats

- Median closing price across ~11 books — line-shopping best price would add
  roughly 1–2% ROI to everything; conversely betting at open vs close is untested.
- The empirical-spread fallback supplied the quantile shape (no external quantile
  model); a dedicated per-stat spread model could sharpen the tails the P(over)
  depends on.
- One season of betting outcomes; the model-vs-line accuracy table is the more
  stable evidence.

Scripts: `scripts/model_burke_props_build.py` (frames from nfl_odds.db) and
`scripts/model_burke_props_run.py` (pipeline + sim); both take the model_burke
package directory as argv[1]. Found and reported upstream: `weekly_winrate`
KeyErrors on an empty eval frame (patched defensively in the local copy).

---

## v2 (same day): the probability layer, fixed — and what it exposed

Replaced the quantile-interpolated P(over) with a **walk-forward calibrated
logistic**: P(over) ~ f(z, line) with z = (Model_Burke_mean − line)/spread-width,
refit weekly on strictly-prior weeks (`scripts/model_burke_props_run2.py`).

**The fix works as calibration.** 2025 receptions deciles now track reality
(predicted 0.44 → realised 0.42; predicted 0.52 → realised 0.55). The v1
phantom under-edge is gone — and with it, almost all stated edge: honest P(over)
on receptions rarely leaves **[0.44, 0.52]**.

**The betting result got worse, not better** (pooled @5%: 1,838 bets, −3.4%).
The remaining "edges" are rows where the de-vigged price sits far from the
model's calibrated ~0.48 — and on those rows the book was right (model unders
won 44.9% when marginal calibration said ~52%). Being calibrated *marginally*
is not being calibrated *conditional on disagreeing with the close*: where the
closing price and the model differ by 5%+, the price knows something (news,
role, injury detail) the model's box-score features don't. That is
closing-line efficiency, measured cleanly.

**What survives both versions:** high-conviction reception-yards OVERS —
v2 @5%: 24 bets, +13.4%; @8%: 13 bets, +76%; every qualifying bet an Over.
v1's monotone curve, same pocket. ~15–25 bets a season.

### Verdict, revised

- The 69%-weekly MAE win on receptions is real but is *pooled point accuracy*;
  per-row, against the close, the market re-absorbs it. The model's honest
  probability edge vs closing prices is ≈ zero outside the rec-yds-over pocket.
- The playable strategy from this architecture is **narrow**: reception-yards
  overs at ≥5–8% calibrated edge, ~1–2 bets/week.
- The natural next test (untested here): the same sheet against **opening**
  lines — a correction model's information advantage should be largest before
  the market has processed the week, and this backtest only ever fought the
  close, the hardest possible benchmark.
- rush-yds v2 never engaged (calibration needs 400 prior rows; the scored
  frame is too thin after burn-in).

## v3: early-vs-close, with the data we hold (`scripts` scratch: props_early.py)

We do NOT hold true openers — snapshots are 1–2 per event, median first capture
2.1h pre-kickoff. Both Odds API keys are DEACTIVATED (cancelled plans), so
re-querying historical opener snapshots needs a reactivated paid plan. What we
could test: the ~350 events/market whose first snapshot leads by ≥5h (median
5.6h — an evening-before market). Model rebuilt with the EARLY line as baseline
(close-as-baseline would be lookahead), bet at early prices, vs the same-events
close sim:

| market | thr | EARLY bets/roi | CLOSE bets/roi |
|---|---|---|---|
| reception yds | 5% | 18 / **+18.8%** | 35 / −0.6% |
| reception yds | 8% | 4 / +94% | 5 / +56% |
| receptions | 5% | 666 / −3.6% | 676 / −2.1% |

Weak directional support where the model already had edge (rec-yds overs do
better against the softer evening-before price) and no rescue where it didn't
(receptions loses either way — earlier prices don't fix a wrong model). n=18/4
in the interesting cells: suggestive only. A real opener test (T-3 to T-5 days)
remains the right experiment and requires restored API access.
