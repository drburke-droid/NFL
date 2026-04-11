"""Model definitions for the four model families + quantile regression.

Family A: Football-only (rolling stats, role features, team stats — no market data)
Family B: Market-only (props, spread, O/U — no football stats)
Family C: Full (all features)
Family D: Residual (two-stage: market prediction → football residual)

Each family uses LightGBM with identical hyperparameters for fair comparison.
"""

from __future__ import annotations
import numpy as np
import lightgbm as lgb
from typing import Optional


def _lgbm_params(cfg: dict, objective: str = "regression") -> dict:
    """Extract LightGBM params from config."""
    p = cfg["modeling"]["lgbm"]
    return {
        "n_estimators": p["n_estimators"],
        "max_depth": p["max_depth"],
        "learning_rate": p["learning_rate"],
        "subsample": p["subsample"],
        "colsample_bytree": p["colsample_bytree"],
        "min_child_samples": p["min_child_samples"],
        "reg_alpha": p["reg_alpha"],
        "reg_lambda": p["reg_lambda"],
        "random_state": p["random_state"],
        "verbose": -1,
        "objective": objective,
    }


class PointModel:
    """Single LightGBM regressor for mean PPR prediction."""

    def __init__(self, cfg: dict, feature_indices: Optional[list[int]] = None):
        self.cfg = cfg
        self.feature_indices = feature_indices
        self.model: Optional[lgb.LGBMRegressor] = None

    def _select(self, X: np.ndarray) -> np.ndarray:
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PointModel":
        self.model = lgb.LGBMRegressor(**_lgbm_params(self.cfg))
        self.model.fit(self._select(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._select(X))

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class ResidualModel:
    """Two-stage model: market prediction → football residual.

    Stage 1: predict PPR from market features only.
    Stage 2: predict (actual - stage1_pred) from football features.
    Final prediction = stage1 + stage2.
    """

    def __init__(self, cfg: dict,
                 market_indices: list[int],
                 football_indices: list[int]):
        self.cfg = cfg
        self.market_idx = market_indices
        self.football_idx = football_indices
        self.stage1: Optional[lgb.LGBMRegressor] = None
        self.stage2: Optional[lgb.LGBMRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ResidualModel":
        # Stage 1: market-only prediction
        X_market = X[:, self.market_idx]
        self.stage1 = lgb.LGBMRegressor(**_lgbm_params(self.cfg))
        self.stage1.fit(X_market, y)

        # Stage 2: football features predict residual over market
        market_pred = self.stage1.predict(X_market)
        residual = y - market_pred

        X_football = X[:, self.football_idx]
        self.stage2 = lgb.LGBMRegressor(**_lgbm_params(self.cfg))
        self.stage2.fit(X_football, residual)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        market_pred = self.stage1.predict(X[:, self.market_idx])
        residual_pred = self.stage2.predict(X[:, self.football_idx])
        return market_pred + residual_pred

    def predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (market_prediction, residual_prediction) separately."""
        market_pred = self.stage1.predict(X[:, self.market_idx])
        residual_pred = self.stage2.predict(X[:, self.football_idx])
        return market_pred, residual_pred


class QuantileModel:
    """LightGBM quantile regression — produces distribution estimates."""

    def __init__(self, cfg: dict, quantiles: Optional[list[float]] = None):
        self.cfg = cfg
        self.quantiles = quantiles or cfg["modeling"]["quantiles"]
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileModel":
        for q in self.quantiles:
            params = _lgbm_params(self.cfg, objective="quantile")
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        """Return dict mapping quantile → predictions."""
        return {q: model.predict(X) for q, model in self.models.items()}

    def predict_df(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return dict with named columns: p10, p25, p50, p75, p90, ceiling, floor."""
        preds = self.predict(X)
        result = {}
        for q, vals in preds.items():
            label = f"p{int(q * 100)}"
            result[label] = vals

        # Derived
        if 0.5 in preds:
            result["median"] = preds[0.5]
        if 0.1 in preds:
            result["floor"] = preds[0.1]
        if 0.9 in preds:
            result["ceiling"] = preds[0.9]
        if 0.9 in preds and 0.1 in preds:
            result["range"] = preds[0.9] - preds[0.1]

        return result


class ThresholdClassifier:
    """Predict probability of exceeding PPR thresholds.

    Uses LightGBM binary classifiers for each threshold.
    """

    def __init__(self, cfg: dict, thresholds: list[float] | None = None):
        self.cfg = cfg
        self.thresholds = thresholds or [15.0, 20.0, 25.0, 30.0]
        self.models: dict[float, lgb.LGBMClassifier] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdClassifier":
        for t in self.thresholds:
            y_bin = (y >= t).astype(int)
            pos = y_bin.sum()
            neg = len(y_bin) - pos
            if pos < 10:
                continue

            params = _lgbm_params(self.cfg)
            del params["objective"]
            model = lgb.LGBMClassifier(
                **params,
                scale_pos_weight=neg / max(pos, 1),
            )
            model.fit(X, y_bin)
            self.models[t] = model
        return self

    def predict_proba(self, X: np.ndarray) -> dict[float, np.ndarray]:
        """Return dict mapping threshold → P(PPR >= threshold)."""
        return {t: model.predict_proba(X)[:, 1] for t, model in self.models.items()}
