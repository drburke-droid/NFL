"""exp_005_skew: exp_004's point, plus a skew-aware quantile layer.

Point   : FFA + 0.5 * local median residual (K = 1000 nearest active rows in projection, per
          position) -- exp_004 verbatim, not tuned here.
Quantiles: per-alpha LightGBM quantile regression of (actual - point) on the skew-carrying features
          found in the segment heterogeneity study (projection level x position, recent volatility,
          WR deep share, early week, favourite/home/total, Q-tag, experience, FFA's own spread).
          Tail distances scaled per position so that held-out 80% coverage is 0.80 (calibration
          season = the last training season, then refit on everything). Sorted, floored.
No I/O.  REFIT = season.
"""
import numpy as np, pandas as pd

REFIT = "season"
K = 1000; S = 0.5
QS = (0.10, 0.25, 0.50, 0.75, 0.90); QCOLS = ("p10", "p25", "p50", "p75", "p90")
POS = ("QB", "RB", "WR", "TE")
SKEW_COLS = ["baseline_proj", "lag_fantasy_points_ppr_std3", "lag_fantasy_points_ppr_r3", "lag_fp_trend",
             "rt_deep_share_r6", "rt_route_share_r3", "week", "ctx_spread", "ctx_total", "ctx_home",
             "ffa_injury_q", "lag_games_played", "ffa_pass_yds_sd", "ffa_rush_yds_sd", "ffa_rec_yds_sd", "ffa_rec_sd"]
Q_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=200,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
SCALE_CLIP = (0.8, 1.25)
FEATURES_ON = True          # False = projection + position only (the falsification check in the proposal)


def _point(train_active, test):
    """exp_004: FFA + S * local median residual, per position."""
    ffa_t = test.baseline_proj.fillna(0.0).values; pos_t = test.position.values
    med = np.zeros(len(test))
    for pos in np.unique(pos_t):
        tp = train_active[train_active.position == pos]
        if len(tp) < 50: tp = train_active
        tp = tp.sort_values("baseline_proj")
        b = tp.baseline_proj.fillna(0.0).values; r = (tp.actual_ppr - tp.baseline_proj.fillna(0.0)).values
        k = min(K, len(b)); sel = np.flatnonzero(pos_t == pos)
        i = np.searchsorted(b, ffa_t[sel]); lo = np.clip(i - k // 2, 0, len(b) - k)
        for j, l in enumerate(lo): med[sel[j]] = np.median(r[l:l + k])
    return ffa_t + S * med


def _design(df, med):
    cols = SKEW_COLS if FEATURES_ON else ["baseline_proj"]
    X = df[cols].copy().fillna(med[cols]).fillna(0.0)
    for p in POS: X[f"pos_{p}"] = (df.position == p).astype(float).values
    return X


def _fit_q(X, y, seed):
    import lightgbm as lgb
    return [lgb.LGBMRegressor(objective="quantile", alpha=float(a), random_state=seed + i, **Q_PARAMS).fit(X, y) for i, a in enumerate(QS)]


def _predict_q(models, X):
    return np.sort(np.column_stack([m.predict(X) for m in models]), axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    tr = train[train.actual_active == 1]
    pred = _point(tr, test)
    med = tr[SKEW_COLS].median()
    # --- calibration fit: everything but the last training season -> scale per position on that season ---
    last = int(tr.season.max()); cal = tr[tr.season == last]; fit0 = tr[tr.season < last]
    scale = {p: 1.0 for p in POS}
    if len(fit0) > 2000 and len(cal) > 500:
        pt0 = _point(fit0, cal)
        m0 = _fit_q(_design(fit0, med), (fit0.actual_ppr.values - _point(fit0, fit0)), seed)
        q0 = _predict_q(m0, _design(cal, med))
        for p in POS:
            sel = (cal.position == p).values
            if sel.sum() < 100: continue
            r = cal.actual_ppr.values[sel] - pt0[sel]; lo, hi = q0[sel, 0], q0[sel, 4]; c50 = q0[sel, 2]
            best, bests = 1.0, 9
            for s in np.arange(SCALE_CLIP[0], SCALE_CLIP[1] + 1e-9, 0.05):
                cov = np.mean((r >= c50 + s * (lo - c50)) & (r <= c50 + s * (hi - c50)))
                if abs(cov - 0.80) < bests: best, bests = s, abs(cov - 0.80)
            scale[p] = float(best)
    # --- final fit on every active training row ---
    models = _fit_q(_design(tr, med), tr.actual_ppr.values - _point(tr, tr), seed)
    q = _predict_q(models, _design(test, med))
    c50 = q[:, 2:3]; sc = test.position.map(scale).fillna(1.0).values[:, None]
    q = c50 + sc * (q - c50)
    q = np.sort(q, axis=1) + pred[:, None]
    floor = np.where((test.position == "QB").values & (pred >= 5), -2.0, 0.0)[:, None]
    q = np.maximum(q, floor)
    out = pd.DataFrame(index=test.index); out["pred"] = pred
    for i, c in enumerate(QCOLS): out[c] = q[:, i]
    return out
