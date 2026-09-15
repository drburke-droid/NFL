# exp_005_skew — notes (2026-09-15)

**Pre-registered failure condition fired.** The claim was that skew-carrying features (volatility,
WR deep share, early week, context, Q-tag, FFA spread) improve pinball >= 1% over exp_004's
projection-only local quantiles. Measured: 1.5479 vs 1.5494 (0.1%), 21/36 weeks, Wilcoxon p = 0.34.
The falsification variant (projection + position only, same GBM, same calibration) scores 1.5514,
so the features are worth 0.2% (Wilcoxon p = 0.03 but immaterial). Coverage landed at 0.799 (target
0.80) from the per-position tail scaling; MAE 4.408 = exp_004's point rule as declared.

| model | cov80 | pinball | pinball at matched 0.80 | composite |
|---|---|---|---|---|
| incumbent (model_burke_prod) | 0.811 | 1.5624 | 1.5624 | 1.0004 |
| exp_004 local quantiles | 0.805 | 1.5494 | 1.5494 | 1.0023 |
| exp_001 boosted quantiles (34 features) | 0.795 | 1.5408 | 1.5409 | 1.0050 |
| exp_005 skew (this) | 0.799 | 1.5479 | 1.5479 | 1.0030 |
| exp_005 projection-only | 0.807 | 1.5514 | 1.5514 | 1.0014 |

What the round established instead: a boosted quantile layer beats the incumbent's local layer on
pinball in 28/36 weeks (p = 0.001) at matched coverage, and exp_001's fuller feature set (lags,
market presence, FFA stat lines) is the best measured layer (beats this one 29/36 weeks). The gain
is the shape of the distribution (stars / deep-ball WRs / volatile players: lower median, longer
right tail), ~1-1.4% pinball, with the point untouched. Per position x tier the skew features help
most for top-12 TE/WR/RB (-0.008 to -0.016 pinball) and nothing for the deep bench.

Action taken: the boosted quantile layer was ported into scripts/sabersim_weekly.py behind `--spread gbm`
(feature family restricted to what the generator's history carries, per-position tail scaling from this
candidate) with its own 2025 spread check next to the local layer. On the generator's own history
(2022-24, actuals only exist from 2022 in data/sabersim/weekly_skill_2022_2025.parquet) the two layers
TIE on 2025: local pinball 1.443 / coverage 0.801 vs GBM 1.444 / 0.819; the point is unchanged. An
apples-to-apples rerun on the league frame (local layer given the same 2016+ pool and the same tail
calibration) closes the league gap too: GBM 1.5479 vs calibrated local 1.5519, 20/36 weeks, Wilcoxon
p = 0.21. The league advantage of boosted quantiles was mostly the data window (2016+ vs the incumbent's
2023+ pool) and coverage tuning, not the features. Production default stays `local`; `--spread gbm`
remains available. Runtime: 8 s for both visible seasons (season refit).
