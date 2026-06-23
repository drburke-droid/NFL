# FantasyFootballAnalytics consensus — does it add value? Yes.

Merged the uploaded FFA weighted-expert season projections (preseason "wk0", 2012–2025)
into the dataset and tested whether they help the season model.

## Ingest
`scripts/fetch_ffa.py` → `nflv_ffa_proj`: projected points, VOR, floor/ceiling, sd,
rank/pos_rank, tier, ADP, dropoff, uncertainty. 2,727 rows, **95% matched to gsis_id**
(misses are mostly players who didn't play that season — Luck/Bell/Edwards — or nickname
diffs). 12 full seasons; 2014–15 are sparse in the source.

## Result — FFA helps, and our model still adds value on top of it
Walk-forward by season (test 2016–2025), same players, predicting next-season PPG:

| Model | MAE | ρ(PPG) | rank-vs-finish |
|---|---|---|---|
| Our prior-year only | 3.055 | 0.664 | 0.444 |
| **FFA consensus alone** | 2.993 | 0.692 | 0.495 |
| **FFA-anchored (prior + FFA)** | **2.859** | **0.717** | **0.519** |

- **FFA consensus alone beats our prior-year-only model** (MAE 2.99 vs 3.06) — a strong, well-built market signal, as expected.
- **FFA-anchored beats both** — MAE **2.859** (−6.4% vs prior-year, **−4.5% vs FFA-alone**), ρ 0.717, finish-rank 0.519. Our prior-year regression/role signals add real, independent value on top of the consensus.
- Biggest gains at **TE (−8.9% MAE)** and **RB (−4.1%)**; the consensus is hardest to beat at QB.

## What the blend actually uses
Top features in the anchored model: **`ffa_dropoff`, `ffa_sd`, `ffa_uncertainty`** (tier-gap
and projection-confidence signals — not just the point estimate), then **`prior_ppr_std`,
`prior_ppg`, `prior2_ppg`, `weight`, `draft_pick`, `ffa_adp`, `prior_receiving_epa`,
`prior_off_pct`, `prior_cv`**. It's a genuine fusion: FFA's *uncertainty/dispersion* metrics
plus our *prior production / role / draft capital*.

## FFA vs ADP
On the 2021–25 overlap, FFA and ADP are ~equal as raw ranking signals (rank-vs-finish
≈ 0.54 each). But FFA is **strictly better to use**: 14 seasons of history vs ADP's 5, and
it ships projected points + floor/ceiling + uncertainty (richer features), which is exactly
what drove the lift above.

## Verdict
**Adds clear value — this is the best season model to date.** FFA is now the season market
anchor (superseding ADP for veterans), with our prior-year signals layered on top.
Output: `nflv_ffa_proj`, `season_ffa_predictions` (1,839 player-seasons, 2016–2025).

**Next:** fold FFA features into `model_season.py` as the production season model; use FFA
floor/ceiling/uncertainty for the explosion/ceiling work; and (when 2026 FFA + ADP publish)
generate a market-anchored 2026 projection.

*Scripts: `scripts/fetch_ffa.py`, `scripts/test_ffa_value.py`.*
