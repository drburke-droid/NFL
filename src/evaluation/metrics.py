"""Evaluation metrics for football forecasting and DFS utility."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    roc_auc_score, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from typing import Optional


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "pearson_r": np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else 0.0,
        "spearman_r": scipy_stats.spearmanr(y_true, y_pred).correlation if len(y_true) > 2 else 0.0,
        "n": int(len(y_true)),
    }


def regression_metrics_by_group(df: pd.DataFrame,
                                actual_col: str,
                                pred_col: str,
                                group_col: str) -> pd.DataFrame:
    """Compute regression metrics per group (e.g., position, tier)."""
    rows = []
    for group_val, gdf in df.groupby(group_col):
        if len(gdf) < 5:
            continue
        m = regression_metrics(gdf[actual_col].values, gdf[pred_col].values)
        m[group_col] = group_val
        rows.append(m)
    return pd.DataFrame(rows)


def top_decile_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Of players predicted in the top 10%, what fraction were actually top 10%?"""
    n = len(y_true)
    if n < 20:
        return 0.0
    top_k = max(1, n // 10)
    pred_top = set(np.argsort(y_pred)[-top_k:])
    actual_top = set(np.argsort(y_true)[-top_k:])
    return len(pred_top & actual_top) / top_k


def calibration_report(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       n_bins: int = 10) -> pd.DataFrame:
    """Calibration table: predicted probability vs actual frequency."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "bin_label": f"{lo:.0%}-{hi:.0%}",
            "n": int(mask.sum()),
            "avg_predicted": y_prob[mask].mean(),
            "avg_actual": y_true[mask].mean(),
            "gap": y_prob[mask].mean() - y_true[mask].mean(),
        })
    return pd.DataFrame(rows)


def model_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a clean comparison table across model families.

    results: {model_name: {metric_name: value}}
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    return df.round(4)


def format_report(title: str, metrics: dict, indent: int = 2) -> str:
    """Format a metrics dict as a readable report block."""
    pad = " " * indent
    lines = [f"{pad}{title}:"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{pad}  {k}: {v:.4f}")
        else:
            lines.append(f"{pad}  {k}: {v}")
    return "\n".join(lines)
