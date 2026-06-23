# Models Report — Avenue 1 (season) & Avenue 2 (weekly)

Built on the signals that survived the EDA. Both use honest, leakage-aware
walk-forward validation. Scripts: `scripts/model_season.py`, `scripts/model_weekly_v2.py`.

---

## Avenue 1 — Season projection model ✅ real lift

**Setup:** LightGBM (L1) predicting next-season PPG. **Season-blocked walk-forward** —
for each target year T (2016–2025), train only on seasons < T. Positional finish & VBD
derived by projecting games (shrink prior games → position mean) and ranking projected
season totals. 4,320 out-of-sample player-seasons.

**PPG accuracy vs the "repeat last year" baseline:**

| Metric | Model | Baseline | 
|---|---|---|
| MAE | **2.65** | 3.02 |
| R² | 0.62 | 0.49 |
| Spearman | 0.757 | 0.720 |

**By position (MAE, model vs baseline):** QB 3.89 vs 4.59 (**+15%**), RB 2.90 vs 3.28 (+12%), WR 2.50 vs 2.83 (+12%), TE 1.84 vs 2.08 (+12%).

**Finish-rank correlation (projected vs actual):** improved at every position (e.g. QB 0.746 vs 0.710, WR 0.752 vs 0.734).

**Honest caveat:** top-tier *identification* (top-12 QB, top-24 RB, etc.) is roughly tied with "repeat last year" — the obvious studs are easy to call. The model's edge is **broad ranking accuracy and regression/breakout calls**, not picking the chalk.

**Top features:** prior2_ppg, prior_ppg, coefficient of variation, weekly std, draft pick, snap %, weight, EPA, air-yards share, 40-time, rec-TD rate. → multi-signal, consistent with the EDA (history depth + role + draft capital + TD regression).

**Output:** `season_predictions` table (per player-season: pred_ppg, proj finish, VBD, actuals).

---

## Avenue 2 — Weekly model v2 ⚠️ informative null result

**Setup:** LightGBM (L1) predicting weekly PPR, walk-forward by (season, week) over the
odds window 2023–25 (14,148 OOS player-weeks). Compared a rolling-stats **BASE** to a
**FULL** set adding xFP (rolling), luck-correction, opportunity share, directional
implied team total, and QB interactions.

**Result: the new signals did NOT improve weekly point MAE.**

| Set | MAE | Corr |
|---|---|---|
| BASE | 4.371 | 0.670 |
| FULL | 4.373 | 0.671 |

No position improved by more than noise; the shootout (O/U 50+) bucket was unmoved.

**Why — and why this is useful, not a failure:**
- **xFP is collinear with opportunity** (ρ = 0.78 with rec+rush attempts). Once the
  baseline has rolling opportunity + snap %, expected points add almost nothing.
- **Weekly luck (actual − expected) does not persist** (lag-1 ρ = **0.03**). A feature
  with no autocorrelation cannot lower next-week error — by construction.
- Independent confirmation of the EDA: the prop market is sharp and weekly point
  prediction is **saturated by volume**. There is little independent edge in mean points.

**Where the new data DOES pay off (the actionable reframe):**
1. **Season projection** (avenue 1) — aggregating over a season averages out the weekly
   noise, and xFP-adjacent role signals helped materially there.
2. **Regression / sell-high signal** — *because* luck doesn't persist, a player who beat
   his xFP is a fade and one who trailed it is a buy-low. Useful for player-pool decisions
   even though it can't sharpen a point projection.
3. **Ceiling, not mean** — DFS GPPs reward the right tail. The L1 model targets the
   conditional *median* (hence a ~+1 under-prediction of the right-skewed mean); the next
   real weekly edge is **quantile/explosion modeling** (upper-tail), not mean MAE — which
   is exactly what the existing `predict_explosions.py` targets and where xFP variance and
   game-environment features are most likely to help.

**Output:** `weekly_v2_predictions` table + feature importances.

---

## Bottom line
- **Avenue 1 is a clear win** — a validated season PPG→finish/VBD model beating "repeat last year" ~12% (15% at QB).
- **Avenue 2's mean-point prediction is already near its ceiling**; the new pipeline's weekly value is in **decision signals (luck-based regression) and tail/explosion modeling**, not mean accuracy.

**Suggested next steps:** (1) extend the season model into a true forward 2026 projection (needs a 2025→2026 prior-feature frame — small builder change); (2) attack the weekly tail with quantile-loss / explosion features (xFP variance, implied team total, shootout flags); (3) source historical ADP to benchmark the season model against the market.
