"""Central walk-forward dataset builder.

build_walk_forward_dataset() is the single entry point for all models.
For a given (season, week), it returns a dict with:
  - train_X, train_y: features/target from all prior weeks
  - score_X, score_meta: features + metadata for the target week
  - feature_names: ordered list of feature column names
  - feature_sets: dict mapping family name → column indices

All transforms are fit ONLY on training data. No future leakage.
"""

from __future__ import annotations
import sqlite3
import pandas as pd
import numpy as np
from typing import Any, Optional

from src.features.rolling import (
    player_rolling_features, player_role_features, team_rolling_features,
)
from src.utils.db import load_player_stats, load_game_odds, load_game_event_lookup


def _merge_odds(df: pd.DataFrame, odds: dict, event_lookup: pd.DataFrame) -> pd.DataFrame:
    """Merge spread, O/U, and prop lines onto player rows."""
    # Drop any existing event_id from stats to avoid conflicts
    df = df.drop(columns=["event_id"], errors="ignore")
    df = df.merge(event_lookup[["team", "season", "week", "event_id", "is_home"]],
                  on=["team", "season", "week"], how="left")
    df = df.merge(odds["spreads"], on="event_id", how="left")
    df = df.merge(odds["totals"], on="event_id", how="left")

    if len(odds["props"]) > 0:
        prop_cols = [c for c in odds["props"].columns if c.startswith("prop_")]
        df = df.merge(
            odds["props"][["event_id", "player_name"] + prop_cols],
            left_on=["event_id", "name"], right_on=["event_id", "player_name"],
            how="left",
        )
        df.drop(columns=["player_name"], errors="ignore", inplace=True)

    return df


def _merge_team_rolling(df: pd.DataFrame,
                        team_off: pd.DataFrame,
                        team_def: pd.DataFrame) -> pd.DataFrame:
    """Merge rolling team offense (own team) and defense (opponent)."""
    off_cols = [c for c in team_off.columns if c.startswith("roll_")]
    df = df.merge(team_off[["team", "season", "week"] + off_cols],
                  on=["team", "season", "week"], how="left")

    def_cols = [c for c in team_def.columns if c.startswith("roll_")]
    df = df.merge(team_def[["team", "season", "week"] + def_cols],
                  left_on=["opponent", "season", "week"],
                  right_on=["team", "season", "week"],
                  how="left", suffixes=("", "_def"))
    df.drop(columns=["team_def"], errors="ignore", inplace=True)

    return df


def _build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build interaction features. Uses only columns already in df."""
    # Defense quality proxy from rolling opp stats (continuous, no archetype needed)
    opp_epa = df["roll_opp_pass_epa"].fillna(0)
    opp_ppr = df["roll_opp_ppr_total"].fillna(0)

    # Player profile × opponent weakness
    df["ix_ppr_x_opp_epa"] = df["roll_ppr"].fillna(0) * opp_epa
    df["ix_targets_x_opp_epa"] = df["roll_target_share"].fillna(0) * opp_epa
    df["ix_rush_x_opp_rush"] = df["roll_rushing_yards"].fillna(0) * df["roll_opp_rush_yds"].fillna(0) / 100
    df["ix_rec_x_opp_pass"] = df["roll_receiving_yards"].fillna(0) * df["roll_opp_pass_yds"].fillna(0) / 100

    # Game environment × player profile
    df["ix_ou_x_ppr"] = df["over_under"].fillna(44) * df["roll_ppr"].fillna(0) / 20
    df["ix_ou_x_targets"] = df["over_under"].fillna(44) * df["roll_target_share"].fillna(0)
    df["ix_spread_x_ppr"] = df["abs_spread"].fillna(3) * df["roll_ppr"].fillna(0) / 10

    # Volatility × opportunity
    df["ix_cv_x_opp_epa"] = df["roll_ppr_cv"].fillna(0) * opp_epa
    df["ix_trend_x_opp_epa"] = df["roll_ppr_trend"].fillna(0) * opp_epa
    df["ix_ceiling_x_ou"] = df["roll_ppr_max"].fillna(0) * df["over_under"].fillna(44) / 44

    # Prop × opponent (market expectation meets matchup)
    for prop_col in ["prop_rec_yds", "prop_rush_yds", "prop_pass_yds"]:
        if prop_col in df.columns:
            df[f"ix_{prop_col}_x_opp_epa"] = df[prop_col].fillna(0) * opp_epa

    return df


def _select_feature_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Select and categorize feature columns into families for ablation.

    Returns (all_features, family_map) where family_map keys are:
      - 'football': rolling stats, role features, team stats
      - 'market': props, spread, O/U
      - 'interaction': ix_* features
      - 'context': is_home, position
    """
    roll = sorted([c for c in df.columns if c.startswith("roll_")])
    role = sorted([c for c in df.columns if c.startswith("role_")])
    prop = sorted([c for c in df.columns if c.startswith("prop_")])
    ix = sorted([c for c in df.columns if c.startswith("ix_")])

    football = roll + role
    market = prop + ["abs_spread", "over_under"]
    market = [c for c in market if c in df.columns]
    interaction = ix
    context = [c for c in ["is_home"] if c in df.columns]

    all_features = football + market + interaction + context
    # Deduplicate while preserving order
    seen = set()
    all_features = [f for f in all_features if not (f in seen or seen.add(f))]

    family_map = {
        "football": [f for f in football if f in all_features],
        "market": [f for f in market if f in all_features],
        "interaction": [f for f in interaction if f in all_features],
        "context": [f for f in context if f in all_features],
    }

    return all_features, family_map


def build_full_dataset(conn: sqlite3.Connection,
                       cfg: dict,
                       positions: list[str] | None = None) -> pd.DataFrame:
    """Build complete feature matrix for ALL weeks. Used by the walk-forward
    loop which handles the time splits externally.

    This is pure feature engineering — no model fitting happens here.
    """
    positions = positions or cfg["walk_forward"]["positions"]
    alpha = cfg["rolling"]["alpha"]
    window = cfg["rolling"]["window"]

    # Load raw data
    stats = load_player_stats(conn, positions=positions)
    odds = load_game_odds(conn)
    event_lookup = load_game_event_lookup(conn)

    # Rolling features (all use shift(1) internally — time-safe)
    stats = player_rolling_features(stats, alpha=alpha, window=window)
    stats = player_role_features(stats, alpha=alpha)
    team_off, team_def = team_rolling_features(stats, alpha=alpha)

    # Merge context
    df = stats.copy()
    df = _merge_odds(df, odds, event_lookup)
    df = _merge_team_rolling(df, team_off, team_def)
    df = _build_interaction_features(df)

    # Season-week sort key
    df["season_week"] = df["season"] * 100 + df["week"]

    return df


def get_feature_families(df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Return feature column names and family groupings."""
    return _select_feature_columns(df)


def build_walk_forward_split(df: pd.DataFrame,
                             target_season: int,
                             target_week: int,
                             feature_cols: list[str],
                             target_col: str = "ppr",
                             min_history: int = 3,
                             min_train_rows: int = 200) -> Optional[dict[str, Any]]:
    """Split dataset for a single walk-forward step.

    Returns None if insufficient data. Otherwise returns dict with:
      train_X, train_y, score_X, score_y, score_meta, feature_names
    """
    target_sw = target_season * 100 + target_week

    train = df[(df["season_week"] < target_sw) & (df["roll_games"] >= min_history)].copy()
    score = df[(df["season_week"] == target_sw) & (df["roll_games"] >= min_history)].copy()

    train = train.dropna(subset=[target_col])
    score = score.dropna(subset=[target_col])

    if len(train) < min_train_rows or len(score) == 0:
        return None

    valid_features = [f for f in feature_cols if f in df.columns]

    return {
        "train_X": train[valid_features].fillna(0).values,
        "train_y": train[target_col].values,
        "score_X": score[valid_features].fillna(0).values,
        "score_y": score[target_col].values,
        "score_meta": score[["player_id", "name", "position", "team", "season",
                             "week", "opponent", "ppr", "roll_ppr", "event_id"]].copy(),
        "feature_names": valid_features,
    }
