"""Time-safe rolling feature computation.

Every feature is computed using ONLY data from prior weeks (shift(1) before
any aggregation). No future leakage by construction.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

# ── Player-level rolling stats ────────────────────────────────────────────

PLAYER_ROLL_COLS = [
    "ppr", "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "passing_epa",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_epa", "target_share",
]


def player_rolling_features(stats: pd.DataFrame,
                            alpha: float = 0.65,
                            window: int = 6) -> pd.DataFrame:
    """Compute exponentially weighted rolling averages per player.

    All features use shift(1) so the current game is NEVER included.
    Returns original df with roll_* columns appended.
    """
    stats = stats.sort_values(["player_id", "season", "week"]).copy()

    for col in PLAYER_ROLL_COLS:
        stats[f"roll_{col}"] = stats.groupby("player_id")[col].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )

    # Game count (shifted)
    stats["roll_games"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # Volatility / ceiling / floor
    stats["roll_ppr_std"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).std()
    )
    stats["roll_ppr_max"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).max()
    )
    stats["roll_ppr_min"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).min()
    )
    stats["roll_ppr_cv"] = stats["roll_ppr_std"] / stats["roll_ppr"].replace(0, np.nan)

    # Trend: last 2 vs last 6
    stats["roll_ppr_recent"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean()
    )
    stats["roll_ppr_trend"] = stats["roll_ppr_recent"] - stats["roll_ppr"]

    return stats


# ── Continuous role descriptors (time-safe, per player-week) ─────────────

def player_role_features(stats: pd.DataFrame,
                         alpha: float = 0.65) -> pd.DataFrame:
    """Compute continuous role features that replace hard archetype labels.

    These capture the *how* of a player's usage, not just volume.
    All use shift(1) — strictly prior data.
    """
    stats = stats.copy()

    # QB role features
    qb = stats["position"] == "QB"
    stats.loc[qb, "role_scramble_rate"] = stats.loc[qb].groupby("player_id").apply(
        lambda g: (g["carries"].shift(1) / (g["carries"].shift(1) + g["attempts"].shift(1)).replace(0, np.nan))
                  .ewm(alpha=alpha, min_periods=1, adjust=False).mean()
    ).droplevel(0) if qb.any() else np.nan

    stats.loc[qb, "role_rush_share"] = stats.loc[qb].groupby("player_id").apply(
        lambda g: g["rushing_yards"].shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
                  / (g["rushing_yards"].shift(1) + g["passing_yards"].shift(1)).replace(0, np.nan)
                  .ewm(alpha=alpha, min_periods=1, adjust=False).mean()
    ).droplevel(0) if qb.any() else np.nan

    # RB role features
    rb = stats["position"] == "RB"
    if rb.any():
        stats.loc[rb, "role_target_share"] = stats.loc[rb].groupby("player_id")["target_share"].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )
        stats.loc[rb, "role_rush_share"] = stats.loc[rb].groupby("player_id").apply(
            lambda g: (g["carries"].shift(1) / (g["carries"].shift(1) + g["targets"].shift(1)).replace(0, np.nan))
                      .ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        ).droplevel(0)
        stats.loc[rb, "role_td_rate"] = stats.loc[rb].groupby("player_id").apply(
            lambda g: ((g["rushing_tds"].shift(1) + g["receiving_tds"].shift(1))
                       / (g["carries"].shift(1) + g["targets"].shift(1)).replace(0, np.nan))
                      .ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        ).droplevel(0)

    # WR / TE role features
    for pos_label in ["WR", "TE"]:
        mask = stats["position"] == pos_label
        if not mask.any():
            continue
        stats.loc[mask, "role_target_share"] = stats.loc[mask].groupby("player_id")["target_share"].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )
        stats.loc[mask, "role_air_yards_share"] = stats.loc[mask].groupby("player_id")["air_yards_share"].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )
        stats.loc[mask, "role_ypr"] = stats.loc[mask].groupby("player_id").apply(
            lambda g: (g["receiving_yards"].shift(1) / g["receptions"].shift(1).replace(0, np.nan))
                      .ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        ).droplevel(0)
        stats.loc[mask, "role_td_rate"] = stats.loc[mask].groupby("player_id").apply(
            lambda g: (g["receiving_tds"].shift(1) / g["targets"].shift(1).replace(0, np.nan))
                      .ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        ).droplevel(0)

    return stats


# ── Team-level rolling stats ────────────────────────────────────────────

def team_rolling_features(stats: pd.DataFrame,
                          alpha: float = 0.65) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling team offense and defense stats.

    Returns (team_off, team_def) DataFrames at the team-season-week grain.
    """
    # Team offense aggregation
    team_off = stats.groupby(["team", "season", "week"]).agg(
        team_pass_yds=("passing_yards", "sum"),
        team_rush_yds=("rushing_yards", "sum"),
        team_pass_tds=("passing_tds", "sum"),
        team_rush_tds=("rushing_tds", "sum"),
        team_pass_att=("attempts", "sum"),
        team_rush_att=("carries", "sum"),
        team_ppr_total=("ppr", "sum"),
        team_pass_epa=("passing_epa", "sum"),
        team_rush_epa=("rushing_epa", "sum"),
    ).reset_index()

    team_off["team_total_yds"] = team_off["team_pass_yds"] + team_off["team_rush_yds"]
    team_off["team_pass_rate"] = (
        team_off["team_pass_att"] /
        (team_off["team_pass_att"] + team_off["team_rush_att"]).replace(0, np.nan)
    )

    team_off = team_off.sort_values(["team", "season", "week"])
    off_cols = ["team_pass_yds", "team_rush_yds", "team_total_yds", "team_pass_tds",
                "team_rush_tds", "team_pass_att", "team_rush_att", "team_ppr_total",
                "team_pass_rate", "team_pass_epa", "team_rush_epa"]
    for col in off_cols:
        team_off[f"roll_{col}"] = team_off.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )

    # Team defense: opponent's offensive output
    team_def = stats.groupby(["opponent", "season", "week"]).agg(
        opp_pass_yds=("passing_yards", "sum"),
        opp_rush_yds=("rushing_yards", "sum"),
        opp_pass_tds=("passing_tds", "sum"),
        opp_rush_tds=("rushing_tds", "sum"),
        opp_ppr_total=("ppr", "sum"),
        opp_pass_epa=("passing_epa", "sum"),
    ).reset_index().rename(columns={"opponent": "team"})

    team_def["opp_total_yds"] = team_def["opp_pass_yds"] + team_def["opp_rush_yds"]
    team_def = team_def.sort_values(["team", "season", "week"])
    def_cols = ["opp_pass_yds", "opp_rush_yds", "opp_total_yds",
                "opp_pass_tds", "opp_rush_tds", "opp_ppr_total", "opp_pass_epa"]
    for col in def_cols:
        team_def[f"roll_{col}"] = team_def.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )

    return team_off, team_def
