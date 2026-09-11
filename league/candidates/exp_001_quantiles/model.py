"""exp_001_quantiles: strong simple mean + boosted quantile regression on the active-only population.

Mean   : baseline_proj + 0.5 * ridge residual (minimalist-style, recent seasons).
Quantiles: one LightGBM quantile regressor per alpha, trained on actual_active == 1 rows, target =
           actual_ppr - mean prediction, conditioned on projection level, position, market presence,
           projection spread, lag volatility, injury flags and game context. Sorted, floored at 0
           (-2 for QBs projected >= 5).
No I/O.
"""
import numpy as np, pandas as pd

REFIT = "week"
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
POS = ("QB", "RB", "WR", "TE")

MEAN_COLS = ["baseline_proj", "mkt_ppr", "mkt_has_line", "ctx_spread", "ctx_total", "ctx_implied",
             "lag_fantasy_points_ppr_l1", "lag_fantasy_points_ppr_r3", "lag_targets_r3", "lag_carries_r3", "ffa_injury_q"]
Q_COLS = ["baseline_proj", "mkt_ppr", "mkt_has_line", "week", "ctx_spread", "ctx_total", "ctx_implied", "ctx_home", "ctx_outdoor",
          "ffa_injury_q", "ffa_injury_o",
          "ffa_pass_yds", "ffa_rush_yds", "ffa_rec", "ffa_rec_yds", "ffa_pass_tds", "ffa_rush_tds", "ffa_rec_tds",
          "ffa_pass_yds_sd", "ffa_rush_yds_sd", "ffa_rec_yds_sd", "ffa_rec_sd",
          "lag_fantasy_points_ppr_l1", "lag_fantasy_points_ppr_r3", "lag_fantasy_points_ppr_r6", "lag_fantasy_points_ppr_std3",
          "lag_targets_r3", "lag_targets_std3", "lag_carries_r3", "lag_carries_std3",
          "lag_target_share_r3", "lag_wopr_r3", "lag_games_played", "lag_fp_trend"]

# knobs (tuned on visible folds; hypothesis fixed)
MEAN_SHRINK = 0.5
MEAN_MIN_SEASON = 2022
MEAN_ACTIVE_ONLY = False      # fit the ridge residual on rows that played
PRED_P50_W = 0.5              # pred = (1-w)*mean + w*p50
Q_MIN_SEASON = 2016
Q_RAW = False                 # quantile target: raw actual_ppr instead of residual on the mean
Q_ALPHA_FIT = {}              # optional {alpha: fitted alpha} tail offsets, e.g. {0.10: 0.085, 0.90: 0.915}
# small, heavily regularised learner: larger trees over-fit the tails (cov80 0.74 at 31 leaves vs 0.80 at 7)
Q_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=200,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def _design(df, cols, med):
    X = df[cols].copy()
    X = X.fillna(med).fillna(0.0)
    for p in POS: X[f"pos_{p}"] = (df.position == p).astype(float).values
    return X


def _mean_model(train, test):
    from sklearn.linear_model import Ridge
    tr = train[train.season >= MEAN_MIN_SEASON]
    if MEAN_ACTIVE_ONLY: tr = tr[tr.actual_active == 1]
    med = tr[MEAN_COLS].median()
    X = tr[MEAN_COLS].fillna(med).fillna(0.0); y = tr.actual_ppr - tr.baseline_proj
    m = Ridge(alpha=10.0).fit(X.values, y.values)
    Xt = test[MEAN_COLS].fillna(med).fillna(0.0)
    return test.baseline_proj.values + MEAN_SHRINK * m.predict(Xt.values), m, med


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    import lightgbm as lgb
    out = pd.DataFrame(index=test.index)
    pred, ridge, med_m = _mean_model(train, test)
    out["pred"] = pred

    # quantile training population: rows that actually played, recent seasons
    tr = train[(train.actual_active == 1) & (train.season >= Q_MIN_SEASON)]
    Xtr_m = tr[MEAN_COLS].fillna(med_m).fillna(0.0)
    mean_tr = tr.baseline_proj.values + MEAN_SHRINK * ridge.predict(Xtr_m.values)
    resid = tr.actual_ppr.values - (0.0 if Q_RAW else mean_tr)
    med_q = tr[Q_COLS].median()
    Xq = _design(tr, Q_COLS, med_q); Xq_t = _design(test, Q_COLS, med_q)

    qpred = np.zeros((len(test), len(QS)))
    for i, a in enumerate(QS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(Q_ALPHA_FIT.get(a, Q_ALPHA_FIT.get(str(a), a))), random_state=seed + i, **Q_PARAMS)
        m.fit(Xq, resid)
        qpred[:, i] = (0.0 if Q_RAW else pred) + m.predict(Xq_t)
    qpred = np.sort(qpred, axis=1)
    if PRED_P50_W > 0: out["pred"] = (1 - PRED_P50_W) * pred + PRED_P50_W * qpred[:, 2]
    floor = np.where((test.position == "QB").values & (pred >= 5), -2.0, 0.0)[:, None]
    qpred = np.maximum(qpred, floor)
    for i, a in enumerate(QS): out[f"p{int(a*100)}"] = qpred[:, i]
    return out
