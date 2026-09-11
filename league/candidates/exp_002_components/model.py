"""exp_002_components: component targets rebuilt into PPR.

Per position and per box-score component (targets, receptions, receiving yards, carries, rushing
yards, attempts, passing yards) a ridge on the component residual (actual - FFA line) from the
FFA stat line, prior-game usage lags, route participation, opponent context and the DK line,
fitted on played rows. Usage components (targets / carries / attempts, no FFA line) are fitted
first and feed the production components. Each component prediction is
    FFA_line + s * residual_hat + f * b
with the shrinkage s and the MAE-optimal offset b learned per position/component on the last
training season (walk-forward), and the offset fraction f chosen per position on the same season
by the judge's MAE/RMSE weights. PPR is rebuilt from the modelled components plus FFA's TD / INT /
fumble points (baseline_proj minus FFA's component points), untouched: a learned multiplier on
them was tested (CFG td_mult) and switched off. Quantiles: empirical residual spread of the ~300
same-position played training rows nearest in baseline_proj, re-centred on the new prediction
(median-centred), with a per-position spread multiplier fitted for 80% coverage on the model's
own last-season validation predictions. REFIT = "season". No I/O.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

REFIT = "season"
CFG = {"kind": "ridge",      # "ridge" (mean), "lgbq" (LightGBM median), "qr" (linear median)
       "offset": True,       # learn an MAE-optimal offset per component on the validation season
       "offset_frac": "learn",   # fraction of the offsets applied: number, or "learn" (MAE/RMSE trade-off on validation)
       "td_mult": False,     # learn a per-position multiplier on FFA's TD/INT/fumble points
       "q_active": True,     # quantile spread from played rows only
       "q_centre": "median", # re-centre the local residual spread on its mean or its median
       "val_seasons": 1,     # number of most recent training seasons used to select shrinkage / offsets / spread scale
       "alpha": 30.0, "min_season": 2018, "qr_rows": 6000}
OFGRID = (0.0, 0.25, 0.5, 0.75, 1.0)
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
PTS = {"pass_yds": 0.04, "rush_yds": 0.1, "rec": 1.0, "rec_yds": 0.1}
COMP = {"targets": (None, "actual_targets"), "rec": ("ffa_rec", "actual_receptions"), "rec_yds": ("ffa_rec_yds", "actual_rec_yds"),
        "carries": (None, "actual_carries"), "rush_yds": ("ffa_rush_yds", "actual_rush_yds"),
        "attempts": (None, "actual_attempts"), "pass_yds": ("ffa_pass_yds", "actual_pass_yds")}
POS_COMPS = {"QB": ["attempts", "carries", "pass_yds", "rush_yds"], "RB": ["carries", "targets", "rush_yds", "rec", "rec_yds"],
             "WR": ["targets", "rec", "rec_yds"], "TE": ["targets", "rec", "rec_yds"]}
STAGE1 = {"targets", "carries", "attempts"}
FEATS = ["ffa_rec", "ffa_rec_yds", "ffa_rush_yds", "ffa_pass_yds", "ffa_pass_tds", "ffa_rush_tds", "ffa_rec_tds", "ffa_pass_int",
         "ffa_rec_sd", "ffa_rec_yds_sd", "ffa_rush_yds_sd", "ffa_pass_yds_sd", "ffa_injury_q", "ffa_injury_o", "baseline_proj",
         "ctx_spread", "ctx_total", "ctx_implied", "ctx_home",
         "lag_targets_l1", "lag_targets_r3", "lag_targets_r6", "lag_receptions_l1", "lag_receptions_r3", "lag_receptions_r6",
         "lag_receiving_yards_l1", "lag_receiving_yards_r3", "lag_receiving_yards_r6", "lag_carries_l1", "lag_carries_r3", "lag_carries_r6",
         "lag_rushing_yards_l1", "lag_rushing_yards_r3", "lag_rushing_yards_r6", "lag_attempts_l1", "lag_attempts_r3", "lag_attempts_r6",
         "lag_passing_yards_l1", "lag_passing_yards_r3", "lag_passing_yards_r6", "lag_target_share_r3", "lag_target_share_r6",
         "lag_wopr_r3", "lag_wopr_r6", "lag_air_yards_share_r6", "lag_games_played", "lag_fp_trend",
         "rt_route_share_l1", "rt_route_share_r3", "rt_route_share_r6", "rt_route_surprise", "rt_tprr_r6", "rt_yprr_r6", "rt_rz_tgt_rate_r6", "rt_team_db_r6",
         "opp_press_r6_z", "opp_man_r6_z", "opp_two_high_r6_z", "opp_box_r6_z", "mkt_ppr", "mkt_rec_yds_line", "mkt_has_line"]
SGRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
TGRID = np.round(np.arange(0.4, 1.21, 0.05), 2)


def _design(df, med):
    """feature matrix: FFA line NaN->0, market NaN->its FFA analogue, everything else -> train median, plus missing flags."""
    X = df[FEATS].copy()
    for c in ("ffa_rec", "ffa_rec_yds", "ffa_rush_yds", "ffa_pass_yds", "ffa_pass_tds", "ffa_rush_tds", "ffa_rec_tds", "ffa_pass_int"):
        X[c] = X[c].fillna(0.0)
    X["mkt_ppr"] = X.mkt_ppr.fillna(df.baseline_proj)
    X["mkt_rec_yds_line"] = X.mkt_rec_yds_line.fillna(X.ffa_rec_yds)
    X["mkt_has_line"] = X.mkt_has_line.fillna(0.0)
    X["lag_missing"] = df.lag_targets_r3.isna().astype(float)
    X["rt_missing"] = df.rt_route_share_r6.isna().astype(float)
    return X.fillna(med).fillna(0.0)


def _ffa_line(df, comp):
    col = COMP[comp][0]
    return df[col].fillna(0.0).values if col else np.zeros(len(df))


def _estimator(seed, n):
    k = CFG["kind"]
    if k == "ridge": return Ridge(alpha=CFG["alpha"])
    if k == "lgbq":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1)
    if k == "qr":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(quantile=0.5, alpha=1.0 / max(n, 1), solver="highs")
    raise ValueError(k)


class _Comp:
    """stage-1 (usage) then stage-2 (production) component models for one position."""
    def __init__(self, pos, seed): self.pos = pos; self.seed = seed

    def fit(self, tr, sel=None):
        tr = tr[tr.actual_active == 1]
        self.med = tr[FEATS].median(numeric_only=True)
        X = _design(tr, self.med); self.mu = X.mean(); self.sd = X.std().replace(0, 1.0)
        Z = ((X - self.mu) / self.sd).values
        self.models = {}; self.sel = sel or {}
        rs = np.random.RandomState(self.seed)
        sub = rs.choice(len(Z), CFG["qr_rows"], replace=False) if (CFG["kind"] == "qr" and len(Z) > CFG["qr_rows"]) else np.arange(len(Z))
        stage1 = np.zeros((len(tr), 0))
        for comp in POS_COMPS[self.pos]:
            y = tr[COMP[comp][1]].values - _ffa_line(tr, comp)
            Zc = Z if comp in STAGE1 else np.hstack([Z, stage1])
            m = _estimator(self.seed, len(sub)).fit(Zc[sub], y[sub]); self.models[comp] = m
            if comp in STAGE1: stage1 = np.hstack([stage1, m.predict(Z)[:, None]])
        return self

    def raw(self, te):
        """residual_hat per component (unshrunk)."""
        X = _design(te, self.med); Z = ((X - self.mu) / self.sd).values
        out = {}; stage1 = np.zeros((len(te), 0))
        for comp in POS_COMPS[self.pos]:
            Zc = Z if comp in STAGE1 else np.hstack([Z, stage1])
            r = self.models[comp].predict(Zc); out[comp] = r
            if comp in STAGE1: stage1 = np.hstack([stage1, r[:, None]])
        return out

    def predict_components(self, te):
        raw = self.raw(te)
        return {c: np.maximum(_ffa_line(te, c) + self.sel.get(c, (1.0, 0.0))[0] * raw[c] + self.sel.get(c, (1.0, 0.0))[1], 0.0) for c in raw}


def _other_points(te):
    """FFA's non-component points: TDs, INT, fumbles (and rounding)."""
    return te.baseline_proj.values - sum(w * te[COMP[c][0]].fillna(0.0).values for c, w in PTS.items())


def _points(te, comps, td_mult=1.0):
    pts = td_mult * _other_points(te)
    for c, w in PTS.items():
        pts = pts + w * (comps[c] if c in comps else te[COMP[c][0]].fillna(0.0).values)
    return pts


def _select(tr, pos, seed):
    """per component: shrinkage s and offset b (MAE-optimal on the last training season); per position: TD multiplier."""
    last = tr.season.max() - CFG["val_seasons"] + 1; fit = tr[tr.season < last]; val = tr[(tr.season >= last) & (tr.actual_active == 1)]
    if len(val) < 200 or (fit.actual_active == 1).sum() < 500: return {c: (0.5, 0.0) for c in POS_COMPS[pos]}, 1.0, None
    m = _Comp(pos, seed).fit(fit); raw = m.raw(val)
    sel = {}
    for comp in POS_COMPS[pos]:
        line = _ffa_line(val, comp); r = raw[comp]; y = val[COMP[comp][1]].values
        best = (np.inf, 0.0, 0.0)
        for g in SGRID:
            b = float(np.median(y - line - g * r)) if CFG["offset"] else 0.0
            e = np.abs(y - np.maximum(line + g * r + b, 0.0)).mean()
            if e < best[0]: best = (e, float(g), b)
        sel[comp] = (best[1], best[2])
    y = val.actual_ppr.values; base = val.baseline_proj.values
    if CFG["offset"] and CFG["offset_frac"] == "learn":
        # fraction of the offsets that maximises the judge's MAE/RMSE terms (0.45 / 0.20) on the validation season
        best = (-np.inf, 1.0)
        for f in OFGRID:
            m.sel = {c: (g, f * b) for c, (g, b) in sel.items()}; pp = np.maximum(_points(val, m.predict_components(val)), 0.0)
            sc = 0.45 * np.abs(y - base).mean() / np.abs(y - pp).mean() + 0.20 * np.sqrt(np.mean((y - base) ** 2)) / np.sqrt(np.mean((y - pp) ** 2))
            if sc > best[0]: best = (sc, f)
        sel = {c: (g, best[1] * b) for c, (g, b) in sel.items()}
    elif CFG["offset"]:
        sel = {c: (g, float(CFG["offset_frac"]) * b) for c, (g, b) in sel.items()}
    m.sel = sel; comps = m.predict_components(val)
    td = 1.0
    if CFG["td_mult"]:
        errs = [np.abs(y - np.maximum(_points(val, comps, t), 0.0)).mean() for t in TGRID]
        td = float(TGRID[int(np.argmin(errs))])
    vpred = pd.Series(np.maximum(_points(val, comps, td), 0.0), index=val.index)   # validation predictions, for the spread calibration
    return sel, td, vpred


def local_quantiles(train, test, centre, k=300, scale=None):
    """generator rule: residual quantiles of the k same-position training rows nearest in baseline_proj, re-centred."""
    if CFG["q_active"]: train = train[train.actual_active == 1]
    out = np.zeros((len(test), 5)); res = (train.actual_ppr - train.baseline_proj)
    for pos in test.position.unique():
        tp = train[train.position == pos]; r = res[tp.index].values; b = tp.baseline_proj.values
        sel = np.flatnonzero((test.position == pos).values); sc = (scale or {}).get(pos, 1.0)
        order = np.argsort(b); bs = b[order]; rs = r[order]
        for i, bp in zip(sel, test.baseline_proj.values[sel]):
            j = np.searchsorted(bs, bp); lo = max(0, j - k // 2); hi = min(len(bs), lo + k); lo = max(0, hi - k)
            rr = rs[lo:hi]; q = np.quantile(rr, QS); c0 = np.median(rr) if CFG["q_centre"] == "median" else rr.mean()
            out[i] = centre[i] + sc * (q - c0)
    floor = np.where((test.position == "QB").values & (centre >= 5), -2.0, 0.0)[:, None]
    return np.maximum(out, floor)


def _learn_scale(tr, vpred):
    """per-position multiplier on the local residual spread giving 80% coverage on the last training season,
    centred on the model's own validation predictions (baseline_proj where none exist)."""
    last = tr.season.max() - CFG["val_seasons"] + 1; fit = tr[tr.season < last]; val = tr[(tr.season >= last) & (tr.actual_active == 1)]
    scale = {}
    for pos in val.position.unique():
        vp = val[val.position == pos]
        if len(vp) < 100 or (fit.position == pos).sum() < 400: scale[pos] = 1.0; continue
        c = vpred.reindex(vp.index).fillna(vp.baseline_proj).values; y = vp.actual_ppr.values; best = (9.0, 1.0)
        for sc in np.arange(0.7, 1.61, 0.05):
            q = local_quantiles(fit, vp, c, scale={pos: float(sc)}); cov = np.mean((y >= q[:, 0]) & (y <= q[:, 4]))
            if abs(cov - 0.80) < best[0]: best = (abs(cov - 0.80), float(sc))
        scale[pos] = best[1]
    return scale


LAST_FIT = {}   # diagnostics only (selected shrinkage / offsets / TD multipliers of the last fit)


def fit_predict(train, test, seed):
    np.random.seed(seed)
    tr = train[train.season >= CFG["min_season"]]
    pred = np.zeros(len(test)); vpreds = []
    for pos in ("QB", "RB", "WR", "TE"):
        sel = (test.position == pos).values
        if not sel.any(): continue
        trp = tr[tr.position == pos]; tep = test[sel]
        choice, td, vpred = _select(trp, pos, seed); LAST_FIT[pos] = (choice, td)
        if vpred is not None: vpreds.append(vpred)
        m = _Comp(pos, seed).fit(trp, choice)
        pred[sel] = _points(tep, m.predict_components(tep), td)
    pred = np.maximum(pred, 0.0)
    vpred = pd.concat(vpreds) if vpreds else pd.Series(dtype=float)
    q = local_quantiles(train, test, pred, scale=_learn_scale(train, vpred))
    out = pd.DataFrame({"pred": pred}, index=test.index)
    for i, qq in enumerate(QS): out[f"p{int(qq*100)}"] = q[:, i]
    return out
