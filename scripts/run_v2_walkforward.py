"""
V2 Walk-Forward Pipeline
Runs all four model families (A-D) + quantile regression + threshold classifiers
in strict walk-forward mode. Produces the ablation comparison report.

Usage:
    python scripts/run_v2_walkforward.py
"""

from __future__ import annotations
import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore")

from src.utils.config import load_config
from src.utils.db import get_conn
from src.utils.logging import setup_logger
from src.features.build_dataset import (
    build_full_dataset, get_feature_families, build_walk_forward_split,
)
from src.modeling.models import PointModel, ResidualModel, QuantileModel, ThresholdClassifier
from src.evaluation.metrics import (
    regression_metrics, regression_metrics_by_group,
    top_decile_hit_rate, calibration_report, model_comparison_table, format_report,
)


def main():
    cfg = load_config()
    log = setup_logger("v2_walkforward", cfg)
    log.info("=" * 80)
    log.info("NFL DFS V2 Walk-Forward Pipeline")
    log.info("=" * 80)

    conn = get_conn(cfg)

    # ── Build full dataset ────────────────────────────────────────────
    log.info("Building full feature dataset...")
    df = build_full_dataset(conn, cfg)
    feature_cols, families = get_feature_families(df)
    log.info(f"Dataset: {len(df)} rows, {len(feature_cols)} features")
    log.info(f"Feature families: {', '.join(f'{k}={len(v)}' for k, v in families.items())}")

    # Map family names to column indices in feature_cols
    def family_indices(family_name: str) -> list[int]:
        return [feature_cols.index(c) for c in families[family_name] if c in feature_cols]

    football_idx = family_indices("football")
    market_idx = family_indices("market")
    full_idx = list(range(len(feature_cols)))
    interaction_idx = family_indices("interaction")
    context_idx = family_indices("context")

    # ── Enumerate all prediction weeks ───────────────────────────────
    min_history = cfg["rolling"]["min_history"]
    start_week = cfg["walk_forward"]["start_week"]
    min_train = cfg["walk_forward"]["min_train_rows"]

    all_sw = sorted(df["season_week"].unique())
    predict_weeks = [(sw // 100, sw % 100) for sw in all_sw if sw % 100 >= start_week]
    log.info(f"Walk-forward: {len(predict_weeks)} prediction weeks")

    # ── Model families to run ────────────────────────────────────────
    model_families = {
        "A_football": lambda cfg: PointModel(cfg, football_idx + context_idx),
        "B_market": lambda cfg: PointModel(cfg, market_idx + context_idx),
        "C_full": lambda cfg: PointModel(cfg, full_idx),
        "D_residual": lambda cfg: ResidualModel(cfg, market_idx, football_idx),
    }

    # Storage for all predictions
    all_preds: dict[str, list[pd.DataFrame]] = {name: [] for name in model_families}
    all_preds["E_quantile"] = []
    all_preds["F_threshold"] = []
    all_preds["baseline_rolling"] = []
    all_preds["baseline_market"] = []

    # ── Walk-forward loop ────────────────────────────────────────────
    n_weeks = len(predict_weeks)
    for i, (season, week) in enumerate(predict_weeks):
        split = build_walk_forward_split(
            df, season, week, feature_cols,
            target_col="ppr", min_history=min_history, min_train_rows=min_train,
        )
        if split is None:
            continue

        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  Week {i+1}/{n_weeks}: {season} Wk{week} | "
                     f"train={len(split['train_y'])}, score={len(split['score_y'])}")

        meta = split["score_meta"].copy()
        actual = split["score_y"]

        # ── Run each model family ────────────────────────────────────
        for name, factory in model_families.items():
            model = factory(cfg)
            model.fit(split["train_X"], split["train_y"])
            preds = model.predict(split["score_X"])

            result = meta.copy()
            result["predicted"] = preds
            result["actual"] = actual
            result["model"] = name
            all_preds[name].append(result)

        # ── Quantile model (full features) ───────────────────────────
        qmodel = QuantileModel(cfg)
        qmodel.fit(split["train_X"], split["train_y"])
        qpreds = qmodel.predict_df(split["score_X"])

        qresult = meta.copy()
        for k, v in qpreds.items():
            qresult[k] = v
        qresult["actual"] = actual
        qresult["model"] = "E_quantile"
        all_preds["E_quantile"].append(qresult)

        # ── Threshold classifier (full features) ─────────────────────
        tmodel = ThresholdClassifier(cfg, thresholds=[15.0, 20.0, 25.0, 30.0])
        tmodel.fit(split["train_X"], split["train_y"])
        tpreds = tmodel.predict_proba(split["score_X"])

        tresult = meta.copy()
        for t, probs in tpreds.items():
            tresult[f"prob_ge_{int(t)}"] = probs
        tresult["actual"] = actual
        tresult["model"] = "F_threshold"
        all_preds["F_threshold"].append(tresult)

        # ── Baselines ────────────────────────────────────────────────
        base = meta.copy()
        base["predicted"] = meta["roll_ppr"].fillna(0).values
        base["actual"] = actual
        base["model"] = "baseline_rolling"
        all_preds["baseline_rolling"].append(base)

        # Market baseline: use prop-implied projection if available
        # Simple: average of available prop lines scaled to PPR
        score_df_row = df[(df["season_week"] == season * 100 + week)
                          & (df["roll_games"] >= min_history)].copy()
        market_base = meta.copy()
        # Use the market model's prediction as the market baseline
        market_model = PointModel(cfg, market_idx + context_idx)
        market_model.fit(split["train_X"], split["train_y"])
        market_base["predicted"] = market_model.predict(split["score_X"])
        market_base["actual"] = actual
        market_base["model"] = "baseline_market"
        all_preds["baseline_market"].append(market_base)

    # ── Aggregate results ────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("RESULTS")
    log.info("=" * 80)

    # Combine point-model predictions
    point_models = ["A_football", "B_market", "C_full", "D_residual",
                    "baseline_rolling", "baseline_market"]

    comparison = {}
    for name in point_models:
        if not all_preds[name]:
            continue
        combined = pd.concat(all_preds[name], ignore_index=True)
        m = regression_metrics(combined["actual"].values, combined["predicted"].values)
        m["top_decile"] = top_decile_hit_rate(combined["actual"].values, combined["predicted"].values)
        comparison[name] = m

        # Per-position breakdown
        pos_metrics = regression_metrics_by_group(combined, "actual", "predicted", "position")
        log.info(f"\n{format_report(name, m)}")
        log.info(f"  By position:")
        for _, row in pos_metrics.iterrows():
            log.info(f"    {row['position']:4s}: MAE={row['mae']:.2f} r={row['pearson_r']:.4f} "
                     f"rho={row['spearman_r']:.4f} n={int(row['n'])}")

    # ── Comparison table ─────────────────────────────────────────────
    comp_df = model_comparison_table(comparison)
    log.info("\n" + "=" * 80)
    log.info("MODEL FAMILY COMPARISON")
    log.info("=" * 80)
    log.info(f"\n{comp_df.to_string()}")

    # ── Ablation answers ─────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("ABLATION ANSWERS")
    log.info("=" * 80)

    a_mae = comparison.get("A_football", {}).get("mae", 999)
    b_mae = comparison.get("B_market", {}).get("mae", 999)
    c_mae = comparison.get("C_full", {}).get("mae", 999)
    d_mae = comparison.get("D_residual", {}).get("mae", 999)
    base_mae = comparison.get("baseline_rolling", {}).get("mae", 999)

    log.info(f"\n  Q1. Market-only power: MAE={b_mae:.3f} "
             f"(vs rolling baseline {base_mae:.3f}, improvement={(base_mae-b_mae)/base_mae:.1%})")
    log.info(f"  Q2. Football beyond market: Full MAE={c_mae:.3f} vs Market MAE={b_mae:.3f} "
             f"(football adds {(b_mae-c_mae)/b_mae:.1%})")
    log.info(f"  Q3. Residual model: MAE={d_mae:.3f} vs Full={c_mae:.3f}")
    log.info(f"  Q4. Football-only: MAE={a_mae:.3f} (no market data at all)")

    # ── Quantile analysis ────────────────────────────────────────────
    if all_preds["E_quantile"]:
        log.info("\n" + "=" * 80)
        log.info("QUANTILE MODEL ANALYSIS")
        log.info("=" * 80)

        qdf = pd.concat(all_preds["E_quantile"], ignore_index=True)

        # Calibration: what % of actuals fall below each predicted quantile?
        for q_label, q_val in [("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
                                ("p75", 0.75), ("p90", 0.90)]:
            if q_label in qdf.columns:
                actual_below = (qdf["actual"] <= qdf[q_label]).mean()
                log.info(f"  {q_label}: {actual_below:.1%} of actuals fall below "
                         f"(should be ~{q_val:.0%})")

        # Median as point prediction
        if "median" in qdf.columns:
            med_m = regression_metrics(qdf["actual"].values, qdf["median"].values)
            log.info(f"\n  Median as point prediction: MAE={med_m['mae']:.3f}, "
                     f"r={med_m['pearson_r']:.4f}")

        # Ceiling usefulness
        if "ceiling" in qdf.columns:
            # How often does actual exceed the p90 ceiling?
            exceed_rate = (qdf["actual"] > qdf["ceiling"]).mean()
            log.info(f"  P90 ceiling exceeded: {exceed_rate:.1%} of the time (should be ~10%)")

    # ── Threshold classifier analysis ────────────────────────────────
    if all_preds["F_threshold"]:
        log.info("\n" + "=" * 80)
        log.info("THRESHOLD PROBABILITY ANALYSIS")
        log.info("=" * 80)

        tdf = pd.concat(all_preds["F_threshold"], ignore_index=True)
        for t in [15, 20, 25, 30]:
            prob_col = f"prob_ge_{t}"
            if prob_col not in tdf.columns:
                continue
            actual_hit = (tdf["actual"] >= t).astype(int)
            if actual_hit.sum() < 10:
                continue

            from sklearn.metrics import roc_auc_score, brier_score_loss
            probs_clean = tdf[prob_col].fillna(0).values
            auc = roc_auc_score(actual_hit, probs_clean)
            brier = brier_score_loss(actual_hit, probs_clean)
            base_rate = actual_hit.mean()

            log.info(f"  P(PPR >= {t}): AUC={auc:.4f}, Brier={brier:.4f}, "
                     f"base_rate={base_rate:.1%}")

            # Calibration
            cal = calibration_report(actual_hit.values, tdf[prob_col].values, n_bins=5)
            for _, row in cal.iterrows():
                log.info(f"    {row['bin_label']:10s}: predicted={row['avg_predicted']:.1%}, "
                         f"actual={row['avg_actual']:.1%}, n={int(row['n'])}")

    # ── Save outputs ─────────────────────────────────────────────────
    output_dir = os.path.join(cfg["paths"]["outputs"], "reports")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comparison table
    comp_df.to_csv(os.path.join(output_dir, f"model_comparison_{ts}.csv"))

    # Save all predictions
    pred_dir = os.path.join(cfg["paths"]["predictions"])
    os.makedirs(pred_dir, exist_ok=True)
    for name in point_models:
        if all_preds[name]:
            combined = pd.concat(all_preds[name], ignore_index=True)
            combined.to_parquet(os.path.join(pred_dir, f"{name}_{ts}.parquet"), index=False)

    if all_preds["E_quantile"]:
        qdf.to_parquet(os.path.join(pred_dir, f"quantile_{ts}.parquet"), index=False)

    if all_preds["F_threshold"]:
        tdf.to_parquet(os.path.join(pred_dir, f"threshold_{ts}.parquet"), index=False)

    # Save config snapshot
    with open(os.path.join(output_dir, f"config_{ts}.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    log.info(f"\nOutputs saved to {output_dir}")
    log.info("Done!")

    conn.close()


if __name__ == "__main__":
    main()
