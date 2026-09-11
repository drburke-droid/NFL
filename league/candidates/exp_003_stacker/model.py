"""exp_003_stacker: low-capacity stacker on the FFA residual with context-dependent blend weights.

pred = baseline_proj + k1 * f_med(context) + k2 * f_mean(context) [+ per-position median offset]
  f_med  : tiny L1-objective LightGBM (6 leaves, 200 trees) on the FFA residual -> conditional-median correction
  f_mean : ridge with hand-built interactions on the same residual -> conditional-mean correction (combo mode only)
  k1, k2, offset flag : chosen on an inner walk-forward window (the most recent `val_weeks` of the training frame,
                        fit on everything before it) by MAE or by the MAE+RMSE part of the league composite. Never eyeballed.
Quantiles: local residual-quantile rule (same position, ~300 nearest training rows in baseline_proj, active rows only),
  re-centred on the model prediction, with the interval width multiplied by a factor chosen on the same inner window
  so that the [p10, p90] band covers 80% of the inner-window actuals (conformal-style).
Training rows: actual_active == 1 (the judge scores only players who played). Weekly refit. No I/O.
"""
import numpy as np, pandas as pd
REFIT = "week"
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
CFG = dict(stage1="lgb", active_only=True, offset="auto", val_weeks=34, criterion="comp",
           k_grid=[round(x, 2) for x in np.arange(0.0, 1.21, 0.1)], k2_grid=[round(x, 2) for x in np.arange(0.0, 1.01, 0.1)],
           leaves=6, n_est=200, lr=0.05, mcs=400, lgb_obj="l1", ridge_alpha=30.0, recent_seasons=None,
           q_active_only=True, q_k=300, q_center="median", q_scale=True, scale_step=0.02, scale_by_pos=False,
           drop=[], half_life=None)
POS = ("QB", "RB", "WR", "TE")

def _fill(s, v=0.0): return s.fillna(v)

def make_X(df, kind):
    bp = df.baseline_proj.astype(float)
    has = _fill(df.mkt_has_line).astype(float)
    dkd = _fill(df.mkt_ppr - bp) * has
    l1 = df.lag_fantasy_points_ppr_l1; l3 = df.lag_fantasy_points_ppr_r3; l6 = df.lag_fantasy_points_ppr_r6
    lagmiss = l6.isna().astype(float)
    l1d = _fill(l1 - bp); l3d = _fill(l3 - bp); l6d = _fill(l6 - bp)
    std3 = _fill(df.lag_fantasy_points_ppr_std3, 5.0)
    X = pd.DataFrame({"bp": bp, "has": has, "dkd": dkd, "l1d": l1d, "l3d": l3d, "l6d": l6d, "lagmiss": lagmiss, "std3": std3,
                      "games": _fill(df.lag_games_played), "week": df.week.astype(float), "injq": _fill(df.ffa_injury_q),
                      "implied": _fill(df.ctx_implied, 22.0), "spread": _fill(df.ctx_spread), "tgt3": _fill(df.lag_targets_r3),
                      "car3": _fill(df.lag_carries_r3), "ffa_rec": _fill(df.ffa_rec), "ffa_rush": _fill(df.ffa_rush_yds)}, index=df.index)
    for p in POS: X[f"pos_{p}"] = (df.position == p).astype(float)
    if kind == "ridge":
        tier = np.clip(bp, 0, 25) / 25.0; early = (df.week <= 4).astype(float)
        for p in POS:
            X[f"dkd_{p}"] = dkd * X[f"pos_{p}"]; X[f"l6d_{p}"] = l6d * X[f"pos_{p}"]; X[f"bp_{p}"] = bp * X[f"pos_{p}"]
        X["dkd_tier"] = dkd * tier; X["l6d_tier"] = l6d * tier; X["l3d_tier"] = l3d * tier
        X["dkd_std"] = dkd * std3 / 5.0; X["l6d_std"] = l6d * std3 / 5.0; X["l3d_std"] = l3d * std3 / 5.0
        X["l6d_early"] = l6d * early; X["l6d_has"] = l6d * has; X["bp_has"] = bp * has; X["bp_tier"] = bp * tier
        X["bp2"] = (bp / 10.0) ** 2
    return X.drop(columns=[c for c in CFG["drop"] if c in X.columns])

def fit_stage1(X, y, kind, seed, w=None):
    if kind == "lgb":
        import lightgbm as lgb
        obj = CFG["lgb_obj"]; extra = {}
        if obj not in ("l1", "quantile"):   # LightGBM forbids monotone constraints under L1/quantile losses
            extra["monotone_constraints"] = [1 if c in ("dkd", "l1d", "l3d", "l6d") else 0 for c in X.columns]
        if obj == "quantile": extra["alpha"] = 0.5
        m = lgb.LGBMRegressor(objective=obj, n_estimators=CFG["n_est"], learning_rate=CFG["lr"], num_leaves=CFG["leaves"],
                              min_child_samples=CFG["mcs"], subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              random_state=seed, verbose=-1, n_jobs=4, **extra)
        m.fit(X.values, y.values, sample_weight=w); return lambda Z: m.predict(Z.values)
    from sklearn.linear_model import Ridge
    mu = X.mean(); sd = X.std().replace(0, 1.0)
    m = Ridge(alpha=CFG["ridge_alpha"]).fit(((X - mu) / sd).values, y.values, sample_weight=w)
    return lambda Z: m.predict(((Z - mu) / sd).values)

def _train_rows(train):
    tr = train
    if CFG["recent_seasons"]: tr = tr[tr.season >= tr.season.max() - CFG["recent_seasons"] + 1]
    if CFG["active_only"]: tr = tr[tr.actual_active == 1]
    return tr

def _offsets(train):
    """per-position median FFA residual on the most recent full training season (active rows) - the incumbent's offset trick."""
    s = train.season.max(); tr = train[(train.season == s) & (train.actual_active == 1)]
    if len(tr) < 500: tr = train[train.actual_active == 1]
    return (tr.actual_ppr - tr.baseline_proj).groupby(tr.position).median()

def _corr_pred(train, test, seed, kind):
    """stage-1 corrections on the FFA residual: columns [median-model (lgb L1), mean-model (ridge)] as requested by `kind`."""
    tr = _train_rows(train); y = tr.actual_ppr - tr.baseline_proj
    w = (0.5 ** ((tr.season.max() - tr.season) / CFG["half_life"])).values if CFG["half_life"] else None   # recency weighting
    kinds = ("lgb", "ridge") if kind == "combo" else (kind,)
    cols = [fit_stage1(make_X(tr, kd), y, kd, seed, w)(make_X(test, kd)) for kd in kinds]
    if len(cols) == 1: cols.append(np.zeros(len(test)))
    return np.column_stack(cols)

def _point(df, c, k1, k2, use_off, offs):
    p = df.baseline_proj.values + k1 * c[:, 0] + k2 * c[:, 1]
    if use_off: p = p + df.position.map(offs).fillna(0.0).values
    return np.maximum(p, 0.0)

def choose_k(val, c, offs, kind):
    """pick (k1, k2, offset flag) on the inner window by MAE ('mae') or by the MAE+RMSE part of the composite vs FFA ('comp')."""
    y = val.actual_ppr.values; bp = val.baseline_proj.values
    ffa_mae = np.abs(y - bp).mean(); ffa_rmse = np.sqrt(((y - bp) ** 2).mean())
    best = (np.inf, 0.0, 0.0, False)
    k2s = CFG["k2_grid"] if kind == "combo" else [0.0]
    offs_opts = (False, True) if CFG["offset"] == "auto" else (bool(CFG["offset"]),)
    for k1 in CFG["k_grid"]:
        for k2 in k2s:
            for use_off in offs_opts:
                e = y - _point(val, c, k1, k2, use_off, offs); mae = np.abs(e).mean()
                score = mae if CFG["criterion"] == "mae" else -(0.45 * ffa_mae / mae + 0.20 * ffa_rmse / np.sqrt((e ** 2).mean()))
                if score < best[0] - 1e-9: best = (score, k1, k2, use_off)
    return best[1], best[2], best[3]

def local_quantiles(train, test, centre):
    """empirical residual quantiles from the ~k training rows of the same position closest in baseline_proj, re-centred on `centre`."""
    tr = train[train.actual_active == 1] if CFG["q_active_only"] else train
    res = (tr.actual_ppr - tr.baseline_proj)
    out = np.zeros((len(test), 5)); kk = CFG["q_k"]; tb = test.baseline_proj.values
    for pos in test.position.unique():
        tp = tr[tr.position == pos]; r = res[tp.index].values; b = tp.baseline_proj.values
        order = np.argsort(b); b = b[order]; r = r[order]
        for i in np.flatnonzero((test.position == pos).values):
            j = np.searchsorted(b, tb[i]); lo = max(0, j - kk // 2); hi = min(len(b), lo + kk); lo = max(0, hi - kk)
            rr = r[lo:hi]
            ctr = rr.mean() if CFG["q_center"] == "mean" else np.median(rr)
            out[i] = np.quantile(rr, QS) - ctr + centre[i]
    return out

def _finish_q(q, pred, test, scale):
    s = test.position.map(scale).fillna(1.0).values if CFG["scale_by_pos"] else np.full(len(test), scale["all"])
    q = pred[:, None] + (q - pred[:, None]) * s[:, None]
    floor = np.where((test.position == "QB").values & (pred >= 5), -2.0, 0.0)[:, None]
    return np.maximum.accumulate(np.maximum(q, floor), axis=1)

def choose_scale(val, q, p_val):
    """width multiplier(s) for the local quantiles so the inner-window [p10, p90] band covers 80% of actuals."""
    y = val.actual_ppr.values
    groups = {p: (val.position == p).values for p in POS} if CFG["scale_by_pos"] else {"all": np.ones(len(val), bool)}
    out = {}
    for g, sel in groups.items():
        best = (np.inf, 1.0)
        for s in np.arange(0.6, 1.81, CFG["scale_step"]):
            qq = _finish_q(q[sel], p_val[sel], val[sel], {"all": s, **{p: s for p in POS}})
            cov = ((y[sel] >= qq[:, 0]) & (y[sel] <= qq[:, 4])).mean()
            if abs(cov - 0.80) < best[0]: best = (abs(cov - 0.80), s)
        out[g] = best[1]
    return out

def fit_predict(train, test, seed):
    kind = CFG["stage1"]
    # inner walk-forward window: fit on everything before the last val_weeks, evaluate on those weeks (active rows)
    ts = np.sort(train.t.unique()); cut = ts[-CFG["val_weeks"]]
    inner_tr = train[train.t < cut]; val = train[(train.t >= cut) & (train.actual_active == 1)]
    c_val = _corr_pred(inner_tr, val, seed, kind); offs_in = _offsets(inner_tr)
    k1, k2, use_off = choose_k(val, c_val, offs_in, kind)
    p_val = _point(val, c_val, k1, k2, use_off, offs_in)
    scale = choose_scale(val, local_quantiles(inner_tr, val, p_val), p_val) if CFG["q_scale"] else {"all": 1.0, **{p: 1.0 for p in POS}}
    # full fit on the whole training window, applied to the test week
    c = _corr_pred(train, test, seed, kind)
    pred = _point(test, c, k1, k2, use_off, _offsets(train))
    q = _finish_q(local_quantiles(train, test, pred), pred, test, scale)
    out = pd.DataFrame({"pred": pred}, index=test.index)
    for i, qq in enumerate(QS): out[f"p{int(qq*100)}"] = q[:, i]
    out.attrs["k"] = (k1, k2); out.attrs["offset"] = use_off; out.attrs["scale"] = scale
    return out
