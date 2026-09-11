"""exp_004_contrarian: FFA + its own conditional median, averaged.

Point:      pred = FFA + S * m(FFA, pos)          S = 0.5
            m = median of (actual - FFA) over the K active training player-weeks of the same
            position closest in FFA projection (a kernel median regression of the FFA residual).
            S = 1 targets the conditional median (best MAE, worse RMSE); S = 0 is FFA itself. The
            composite weights MAE 0.45 / RMSE 0.20, so the point sits half-way (flat optimum 0.4-0.6).
Quantiles:  p_q = FFA + Q_q(residual) over the same neighbourhood, ACTIVE rows only (the production
            local-quantile rule includes inactive zeros, which drags p10/p25/p50 down and under-covers).
            p50 is therefore the local conditional median; pred lies between FFA and p50.
No DK line, no lags, no learned parameters beyond (K, S). Weekly refit is a sort + K-window medians.
"""
import numpy as np, pandas as pd

REFIT = "week"
K = 1000          # neighbourhood size in projection (per position)
S = 0.5           # weight on the local median residual for the point prediction
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
QCOLS = ("p10", "p25", "p50", "p75", "p90")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    tr = train[train.actual_active == 1]
    ffa_t = test.baseline_proj.fillna(0.0).values
    pos_t = test.position.values
    q = np.zeros((len(test), 5))
    for pos in np.unique(pos_t):
        tp = tr[tr.position == pos]
        if len(tp) < 50:                      # unseen / tiny position: fall back to the whole active pool
            tp = tr
        tp = tp.sort_values("baseline_proj")
        b = tp.baseline_proj.fillna(0.0).values
        r = (tp.actual_ppr - tp.baseline_proj.fillna(0.0)).values
        k = min(K, len(b))
        sel = np.flatnonzero(pos_t == pos)
        i = np.searchsorted(b, ffa_t[sel])
        lo = np.clip(i - k // 2, 0, len(b) - k)
        for j, l in enumerate(lo):
            q[sel[j]] = np.quantile(r[l:l + k], QS)
    out = pd.DataFrame(index=test.index)
    out["pred"] = ffa_t + S * q[:, 2]
    floor = np.where((pos_t == "QB") & (out.pred.values >= 5), -2.0, 0.0)
    for c, col in enumerate(QCOLS):
        out[col] = np.maximum(ffa_t + q[:, c], floor)
    return out
