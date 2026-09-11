"""Candidate model template. Copy to league/candidates/<exp_id>/model.py and implement fit_predict.

Rules (enforced by the judge):
  * No I/O. No file, database or network reads. The judge passes `train` and `test` in memory.
  * `train`: visible rows with t < test week, including actual_* columns. `test`: the rows to
    predict, WITHOUT actual_* columns. Never look up anything about test rows beyond `test`.
  * Return a DataFrame indexed exactly like `test` with columns pred, p10, p25, p50, p75, p90
    (monotone). Every test row needs a prediction (coverage >= 99.9%).
  * REFIT = "week" (fit before every test week; ~36 fits for the visible window) or "season".
    Keep a weekly fit under ~30 s.
  * Use `seed` for anything random.
Feature families in the frame: ffa_* (consensus stat line; baseline_proj = its PPR score),
mkt_* (DK-implied PPR, 2023+ wk5-22 only), ctx_* (spread/total/implied/home/outdoor), lag_*
(prior-game box-score rolling stats), rt_* (route participation & usage), opp_* (opponent defence
z-scores), qb_* (team QB1 prior-6 style), ngs_* (prior-season separation), position, team, week.
See league/data/FRAME_MANIFEST.txt for the full column list.
"""
import numpy as np, pandas as pd
REFIT = "week"

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Example: FFA passthrough with empirical position spread. Replace with your hypothesis.
    out = pd.DataFrame(index=test.index)
    out["pred"] = test.baseline_proj.values
    res = train.actual_ppr - train.baseline_proj
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        qs = res.groupby(train.position).quantile(q)
        out[f"p{int(q*100)}"] = np.maximum(out.pred + test.position.map(qs).values, 0.0)
    return out
