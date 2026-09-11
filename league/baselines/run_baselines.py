"""Required baselines, scored on the visible folds and registered in league/registry.

  prev_game     last game's PPR (0 if none)                          quantiles: empirical position spread
  trail4        trailing 4-game mean PPR
  ffa           FFA weekly consensus stat line, PPR-scored (baseline_proj)      <- FROZEN normaliser
  dk            DK-implied PPR where a line exists (mkt_ppr), FFA elsewhere   (reference only)
  blend         0.6 * DK + 0.4 * FFA where a line exists, FFA elsewhere
  minimalist    ridge on <= 12 features (FFA, DK, spread/total, 3 lags), weekly refit
  kitchen_sink  LightGBM on every feature, weekly refit (residual on FFA)
  model_burke   the shipped package run on the frame (weekly walk-forward), package quantiles
  model_burke_prod  package mean + the generator's local quantiles (as deployed)   <- INCUMBENT
Quantiles for the non-distributional baselines come from the empirical residual spread of the
same position in the training rows nearest in projection (the generator's local_quantiles rule).
Usage: python league/baselines/run_baselines.py <model_burke pkg dir> [--only name]
"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "league", "judge"))
import score as J
QS = J.QS

def local_quantiles(train, test, centre_col, k=300):
    """empirical residual quantiles from the ~k training rows of the same position closest in baseline_proj, re-centred on `centre_col`."""
    out = np.zeros((len(test), 5)); res = train.actual_ppr - train.baseline_proj
    for pos in test.position.unique():
        tp = train[train.position == pos]; r = res[tp.index].values; b = tp.baseline_proj.values
        sel = test.position == pos
        for i, (bp, c) in enumerate(zip(test.loc[sel, "baseline_proj"].values, test.loc[sel, centre_col].values)):
            idx = np.argsort(np.abs(b - bp))[:k]; rr = r[idx]
            out[np.flatnonzero(sel)[i]] = np.quantile(rr, QS) - rr.mean() + c
    floor = np.where((test.position == "QB").values & (test[centre_col].values >= 5), -2.0, 0.0)[:, None]
    return np.maximum(out, floor)

def with_quantiles(train, test, centre):
    # production faithful: the generator's residual history holds only players with a box score
    # (played); the league frame also carries inactive zeros, which must not shape the spread
    if "actual_active" in train.columns: train = train[train.actual_active == 1]
    q = local_quantiles(train, test, centre)
    df = pd.DataFrame({"pred": test[centre].values}, index=test.index)
    for i, qq in enumerate(QS): df[f"p{int(qq*100)}"] = q[:, i]
    return df

class Simple:
    REFIT = "season"
    def __init__(self, kind): self.kind = kind
    def fit_predict(self, train, test, seed):
        t = test.copy()
        if self.kind == "prev_game": t["c"] = t.lag_fantasy_points_ppr_l1.fillna(0.0)
        elif self.kind == "trail4": t["c"] = t.lag_fantasy_points_ppr_r3.fillna(t.lag_fantasy_points_ppr_l1).fillna(0.0)   # package lags are l1/r3/r6
        elif self.kind == "ffa": t["c"] = t.baseline_proj
        elif self.kind == "dk": t["c"] = t.mkt_ppr.fillna(t.baseline_proj)
        elif self.kind == "blend": t["c"] = np.where(t.mkt_ppr.notna(), 0.6 * t.mkt_ppr.fillna(0) + 0.4 * t.baseline_proj, t.baseline_proj)
        return with_quantiles(train, t, "c")

class Minimalist:
    REFIT = "week"
    COLS = ["baseline_proj", "mkt_ppr", "mkt_has_line", "ctx_spread", "ctx_total", "ctx_implied", "lag_fantasy_points_ppr_l1", "lag_fantasy_points_ppr_r3", "lag_targets_r3", "lag_carries_r3", "ffa_injury_q"]
    def fit_predict(self, train, test, seed):
        from sklearn.linear_model import Ridge
        tr = train[train.season >= 2022]   # market only exists from 2023; keep the fit recent
        X = tr[self.COLS].fillna(tr[self.COLS].median()).fillna(0); y = tr.actual_ppr - tr.baseline_proj
        m = Ridge(alpha=10.0).fit(X.values, y.values)
        Xt = test[self.COLS].fillna(tr[self.COLS].median()).fillna(0)
        t = test.copy(); t["c"] = test.baseline_proj + 0.5 * m.predict(Xt.values)
        return with_quantiles(train, t, "c")

class KitchenSink:
    REFIT = "week"
    def fit_predict(self, train, test, seed):
        import lightgbm as lgb
        cols = [c for c in train.columns if c.startswith(("ffa_", "mkt_", "ctx_", "lag_", "rt_", "opp_", "qb_", "ngs_")) or c == "baseline_proj"]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
        tr = train[train.season >= 2019]
        for pos in ("QB", "RB", "WR", "TE"): tr = tr.assign(**{f"pos_{pos}": (tr.position == pos).astype(float)}); test = test.assign(**{f"pos_{pos}": (test.position == pos).astype(float)})
        cols2 = cols + [f"pos_{p}" for p in ("QB", "RB", "WR", "TE")]
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.7, random_state=seed, verbose=-1)
        m.fit(tr[cols2], tr.actual_ppr - tr.baseline_proj)
        t = test.copy(); t["c"] = test.baseline_proj + 0.5 * m.predict(test[cols2])
        return with_quantiles(train, t, "c")

def model_burke_predictions(pkg, frame, seasons):
    """Production-faithful: the generator's history holds only players with a box score, so the
    package is fitted on played rows (FFA baseline, no market feature); inactive rows (never
    scored) get the FFA line."""
    sys.path.insert(0, pkg)
    from model_burke import pipeline
    d = frame[frame.actual_active == 1].copy() if "actual_active" in frame.columns else frame.copy()
    d = d.rename(columns={"ctx_spread": "spread", "ctx_total": "game_total", "ctx_implied": "implied_team_total"})
    # production-faithful: the generator's history carries NO market column (market_proj = NaN); DK enters
    # only at the stat level of the live slate baseline, which this frame cannot reproduce (it holds a
    # DK-implied PPR proxy, not per-stat lines). With the proxy as a GBM feature the package scores
    # 4.412 vs 4.411 without it (incumbent_diag, 2026-09-11): immaterial, so the faithful form is used.
    d["market_proj"] = np.nan
    d = d.rename(columns={c: c[4:] for c in d.columns if c.startswith("lag_")})     # the package expects its own lag names
    d["wind_kn"] = 0.0; d["player"] = d.nname; d["_rid"] = d.index      # the package re-indexes; carry the row id through
    ev, rep = pipeline.run(d, verbose=False)
    ev = ev[ev.season.isin(seasons)]
    # pred = the package's POINT projection (residual correction + median bias offset, MAE-optimised),
    # which is what the generator ships as Proj; Model_Burke_mean is the un-offset mean
    out = pd.DataFrame({"pred": ev.Model_Burke.values, "p10": ev.mb_p10.values, "p25": ev.mb_p25.values, "p50": ev.mb_p50.values, "p75": ev.mb_p75.values, "p90": ev.mb_p90.values}, index=ev._rid.values)
    rest = frame[frame.season.isin(seasons)].index.difference(out.index)      # inactive rows: FFA line, flat quantiles
    if len(rest):
        fb = pd.DataFrame({"pred": frame.loc[rest, "baseline_proj"].values}, index=rest)
        for q in ("p10", "p25", "p50", "p75", "p90"): fb[q] = fb.pred
        out = pd.concat([out, fb])
    return out

def model_burke_prod_predictions(pkg, frame, seasons, seed):
    """The incumbent as DEPLOYED: package mean + the generator's local projection-conditioned quantiles."""
    mb = model_burke_predictions(pkg, frame, seasons)
    out = []
    for s in seasons:
        # residual pool = the generator's history window (2023+ via --history-start; the single prior
        # season when 2023+ is not yet available). The 2016+ pool over-covers (0.825-0.834 vs 0.80).
        train = frame[(frame.season < s) & (frame.season >= min(2023, s - 1))]; test = frame[frame.season == s].copy()
        test["c"] = mb.pred.reindex(test.index).fillna(test.baseline_proj)
        out.append(with_quantiles(train, test, "c"))
    return pd.concat(out)

def baseline_predictions(name, frame, seasons, seed, pkg=None):
    if name == "model_burke": return model_burke_predictions(pkg or os.environ.get("MODEL_BURKE_PKG"), frame, seasons)
    if name == "model_burke_prod": return model_burke_prod_predictions(pkg or os.environ.get("MODEL_BURKE_PKG"), frame, seasons, seed)
    model = {"prev_game": Simple("prev_game"), "trail4": Simple("trail4"), "ffa": Simple("ffa"), "dk": Simple("dk"), "blend": Simple("blend"),
             "minimalist": Minimalist(), "kitchen_sink": KitchenSink()}[name]
    p, _ = J.walk_forward(model, frame, seasons, seed, log=lambda *a: None)
    return p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkg"); ap.add_argument("--only"); A = ap.parse_args()
    os.environ["MODEL_BURKE_PKG"] = A.pkg
    frame = J.load_frame()
    order = ["ffa", "prev_game", "trail4", "dk", "blend", "minimalist", "kitchen_sink", "model_burke", "model_burke_prod"]   # ffa first: it is the normaliser
    if A.only: order = [A.only]
    for name in order:
        t0 = time.time(); print(f"== {name}")
        p = baseline_predictions(name, frame, J.CONFIG["eval_seasons"], 17, A.pkg)
        os.makedirs(os.path.join(ROOT, "league", "registry", "submissions"), exist_ok=True)
        p.to_parquet(os.path.join(ROOT, "league", "registry", "submissions", f"{name}_visible.parquet"))
        rec = J.score_predictions(p, name, kind="visible", extra={"baseline": True, "runtime_s": round(time.time() - t0)}, frame=frame)
        m = rec["metrics"]; print(f"   n={m['n']} cov={m['coverage']} MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} rho {m['spearman']:.3f} cov80 {m['cov80']} pinball {m['pinball'] and round(m['pinball'],3)} composite {m.get('composite') and round(m['composite'],4)}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__": main()
