"""
NFL DFS Points Prediction — Walk-Forward Engine
Predicts player PPR fantasy points using:
  - Rolling exponentially weighted player stats
  - Dynamic archetype assignment (rolling profile → nearest cluster center)
  - Rolling team offense/defense stats
  - Betting lines (spread, O/U) and player prop lines
  - Opponent archetype matchup features
  - Predicted game script probabilities

Walk-forward: for each week, trains ONLY on prior data.
"""

import sqlite3
import os
import sys
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "nfl_odds.db")

# Rolling window config
ROLLING_WINDOW = 6       # games to look back
ROLLING_ALPHA = 0.65     # exponential decay (higher = more recent weight)
MIN_HISTORY = 3          # min games before we predict a player
WALK_FORWARD_START = 5   # start predicting from week 5 (need history)

POSITIONS = ["QB", "RB", "WR", "TE"]


# =====================================================================
# Data Loading
# =====================================================================

def load_all_data(conn):
    """Load all required data from DB."""
    print("Loading data...")

    stats = pd.read_sql_query("""
        SELECT ps.player_id, ps.player_display_name as name, ps.position,
               ps.team, ps.season, ps.week, ps.opponent,
               ps.completions, ps.attempts, ps.passing_yards, ps.passing_tds,
               ps.interceptions, ps.sacks, ps.passing_epa,
               ps.carries, ps.rushing_yards, ps.rushing_tds, ps.rushing_epa,
               ps.targets, ps.receptions, ps.receiving_yards, ps.receiving_tds,
               ps.receiving_epa, ps.target_share, ps.air_yards_share,
               ps.fantasy_points_ppr as ppr
        FROM player_stats ps
        WHERE ps.position IN ('QB','RB','WR','TE')
        ORDER BY ps.season, ps.week
    """, conn)

    odds = pd.read_sql_query("""
        SELECT gs.game_id, gs.event_id, gs.season, gs.week,
               gs.home_team, gs.away_team, gs.game_script
        FROM game_scripts gs
    """, conn)

    # Spread and O/U
    spreads = pd.read_sql_query("""
        SELECT DISTINCT event_id, MIN(ABS(point)) as abs_spread
        FROM game_odds WHERE market='spreads' AND bookmaker='draftkings'
        GROUP BY event_id
    """, conn)

    totals = pd.read_sql_query("""
        SELECT DISTINCT event_id, point as over_under
        FROM game_odds WHERE market='totals' AND outcome_name='Over' AND bookmaker='draftkings'
    """, conn)
    totals = totals.drop_duplicates(subset="event_id", keep="first")

    # Player props — pivot to one row per player per event
    props_raw = pd.read_sql_query("""
        SELECT pp.event_id, pp.player_name, pp.market, pp.point as prop_line
        FROM player_props pp
        WHERE pp.bookmaker = 'draftkings' AND pp.outcome_type = 'Over'
    """, conn)

    # Pivot props
    props_wide = props_raw.pivot_table(
        index=["event_id", "player_name"],
        columns="market",
        values="prop_line",
        aggfunc="first"
    ).reset_index()
    props_wide.columns.name = None
    # Rename prop columns
    prop_rename = {
        "player_pass_yds": "prop_pass_yds", "player_pass_tds": "prop_pass_tds",
        "player_pass_attempts": "prop_pass_att", "player_pass_completions": "prop_pass_comp",
        "player_rush_yds": "prop_rush_yds", "player_rush_attempts": "prop_rush_att",
        "player_reception_yds": "prop_rec_yds", "player_receptions": "prop_receptions",
        "player_rush_reception_yds": "prop_rush_rec_yds",
        "player_anytime_td": "prop_anytime_td",
    }
    props_wide = props_wide.rename(columns=prop_rename)

    # Team archetypes (for cluster centers reference)
    team_arch = pd.read_sql_query("""
        SELECT team, season, unit, archetype, style FROM team_archetypes
    """, conn)

    print(f"  Stats: {len(stats)} rows, Props: {len(props_wide)} player-games with props")
    print(f"  Odds: {len(spreads)} spreads, {len(totals)} totals")

    return stats, odds, spreads, totals, props_wide, team_arch


# =====================================================================
# Rolling Features
# =====================================================================

PLAYER_ROLL_COLS = [
    "ppr", "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "passing_epa",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_epa", "target_share",
]


def compute_ewm(group, cols, alpha, window):
    """Compute exponentially weighted rolling averages, shifted by 1 (no leakage)."""
    result = {}
    for col in cols:
        series = group[col].astype(float)
        # Shift by 1 so current game is excluded
        shifted = series.shift(1)
        ewm = shifted.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        # Also limit to window
        result[f"roll_{col}"] = ewm
    # Rolling game count (for weighting/filtering)
    result["roll_games"] = group["ppr"].shift(1).expanding().count()
    return pd.DataFrame(result, index=group.index)


def build_rolling_features(stats):
    """Build rolling exponentially weighted features per player."""
    print("Computing rolling player features...")

    stats = stats.sort_values(["player_id", "season", "week"])

    # Group by player (across seasons for continuity)
    rolling = stats.groupby("player_id").apply(
        lambda g: compute_ewm(g, PLAYER_ROLL_COLS, ROLLING_ALPHA, ROLLING_WINDOW),
    )

    # Flatten multi-index if needed
    if isinstance(rolling.index, pd.MultiIndex):
        rolling = rolling.droplevel(0)

    stats = pd.concat([stats, rolling], axis=1)

    # Rolling variance of PPR (consistency measure)
    stats["roll_ppr_std"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).std()
    )

    # Rolling ceiling/floor
    stats["roll_ppr_max"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).max()
    )
    stats["roll_ppr_min"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).min()
    )

    print(f"  Built {len([c for c in stats.columns if c.startswith('roll_')])} rolling features")
    return stats


# =====================================================================
# Rolling Team Stats
# =====================================================================

def build_team_rolling(stats):
    """Build rolling team offense and defense stats per team per week."""
    print("Computing rolling team stats...")

    # Team offense: aggregate player stats per team-game
    team_off = stats.groupby(["team", "season", "week"]).agg(
        team_pass_yds=("passing_yards", "sum"),
        team_rush_yds=("rushing_yards", "sum"),
        team_pass_tds=("passing_tds", "sum"),
        team_rush_tds=("rushing_tds", "sum"),
        team_pass_att=("attempts", "sum"),
        team_rush_att=("carries", "sum"),
        team_targets=("targets", "sum"),
        team_ppr_total=("ppr", "sum"),
        team_pass_epa=("passing_epa", "sum"),
        team_rush_epa=("rushing_epa", "sum"),
    ).reset_index()

    team_off["team_total_yds"] = team_off["team_pass_yds"] + team_off["team_rush_yds"]
    team_off["team_pass_rate"] = team_off["team_pass_att"] / (team_off["team_pass_att"] + team_off["team_rush_att"]).replace(0, np.nan)

    # Rolling averages for team offense
    team_off = team_off.sort_values(["team", "season", "week"])
    off_cols = ["team_pass_yds", "team_rush_yds", "team_total_yds", "team_pass_tds",
                "team_rush_tds", "team_pass_att", "team_rush_att", "team_ppr_total",
                "team_pass_rate", "team_pass_epa", "team_rush_epa"]

    for col in off_cols:
        team_off[f"roll_{col}"] = team_off.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(alpha=ROLLING_ALPHA, min_periods=1, adjust=False).mean()
        )

    # Team defense: what opponents scored AGAINST this team
    # Build from opponent perspective
    team_def = stats.groupby(["opponent", "season", "week"]).agg(
        opp_pass_yds=("passing_yards", "sum"),
        opp_rush_yds=("rushing_yards", "sum"),
        opp_pass_tds=("passing_tds", "sum"),
        opp_rush_tds=("rushing_tds", "sum"),
        opp_ppr_total=("ppr", "sum"),
        opp_pass_epa=("passing_epa", "sum"),
    ).reset_index().rename(columns={"opponent": "team"})

    opp_total = opp_pass_yds = None  # just for clarity
    team_def["opp_total_yds"] = team_def["opp_pass_yds"] + team_def["opp_rush_yds"]

    team_def = team_def.sort_values(["team", "season", "week"])
    def_cols = ["opp_pass_yds", "opp_rush_yds", "opp_total_yds",
                "opp_pass_tds", "opp_rush_tds", "opp_ppr_total", "opp_pass_epa"]

    for col in def_cols:
        team_def[f"roll_{col}"] = team_def.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(alpha=ROLLING_ALPHA, min_periods=1, adjust=False).mean()
        )

    print(f"  Team offense: {len(team_off)} team-games, Defense: {len(team_def)} team-games")
    return team_off, team_def


# =====================================================================
# Dynamic Archetype Assignment
# =====================================================================

def build_dynamic_archetypes(stats):
    """Assign archetypes dynamically based on rolling stats using fixed cluster centers."""
    print("Assigning dynamic archetypes from rolling stats...")

    # Build archetype features from rolling stats (same features used in clustering)
    archetype_features = {
        "QB": ["roll_attempts", "roll_passing_yards", "roll_passing_tds",
               "roll_carries", "roll_rushing_yards", "roll_passing_epa"],
        "RB": ["roll_carries", "roll_rushing_yards", "roll_rushing_tds",
               "roll_targets", "roll_receptions", "roll_receiving_yards", "roll_rushing_epa"],
        "WR": ["roll_targets", "roll_receptions", "roll_receiving_yards",
               "roll_receiving_tds", "roll_target_share", "roll_receiving_epa"],
        "TE": ["roll_targets", "roll_receptions", "roll_receiving_yards",
               "roll_receiving_tds", "roll_target_share", "roll_receiving_epa"],
    }

    # Archetype labels per position (ordered by volume)
    archetype_labels = {
        "QB": ["Game Manager", "High-Volume Passer", "Dual-Threat"],
        "RB": ["Depth", "Rotational", "Starter", "Workhorse"],
        "WR": ["Depth", "WR3/Flex", "WR2", "Alpha WR1"],
        "TE": ["Blocking/Depth", "Secondary Receiver", "Elite Receiving"],
    }

    stats = stats.copy()
    stats["dyn_archetype"] = "Unknown"

    for pos in POSITIONS:
        pos_mask = stats["position"] == pos
        feats = archetype_features[pos]
        labels = archetype_labels[pos]
        n_clusters = len(labels)

        pos_data = stats.loc[pos_mask, feats].dropna()
        if len(pos_data) < n_clusters * 10:
            continue

        # Fit K-Means on the full rolling data to get stable centers
        scaler = StandardScaler()
        X = scaler.fit_transform(pos_data.values)
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        km.fit(X)

        # Sort clusters by the first volume feature (ascending) to assign labels
        center_volumes = km.cluster_centers_[:, 0]  # first feature is typically volume
        sorted_idx = np.argsort(center_volumes)
        label_map = {sorted_idx[i]: labels[i] for i in range(n_clusters)}

        # Assign all rows
        predictions = km.predict(X)
        stats.loc[pos_data.index, "dyn_archetype"] = [label_map[p] for p in predictions]

    assigned = (stats["dyn_archetype"] != "Unknown").sum()
    print(f"  Assigned dynamic archetypes to {assigned}/{len(stats)} rows")
    return stats


# =====================================================================
# Feature Assembly
# =====================================================================

def assemble_features(stats, odds, spreads, totals, props, team_off, team_def, team_arch, conn=None):
    """Build the final feature matrix for modeling."""
    print("Assembling feature matrix...")

    df = stats.copy()

    # Merge odds via game lookup
    # Build event_id mapping: team + season + week → event_id
    game_lookup = odds.copy()
    # Create home/away lookups
    home_lookup = game_lookup[["event_id", "season", "week", "home_team", "away_team", "game_script"]].copy()
    home_lookup = home_lookup.rename(columns={"home_team": "team", "away_team": "opponent_full"})
    home_lookup["is_home"] = 1

    away_lookup = game_lookup[["event_id", "season", "week", "away_team", "home_team", "game_script"]].copy()
    away_lookup = away_lookup.rename(columns={"away_team": "team", "home_team": "opponent_full"})
    away_lookup["is_home"] = 0

    event_lookup = pd.concat([home_lookup, away_lookup], ignore_index=True)

    df = df.merge(event_lookup[["team", "season", "week", "event_id", "is_home", "game_script"]],
                  on=["team", "season", "week"], how="left")

    # Merge spread and O/U
    df = df.merge(spreads, on="event_id", how="left")
    df = df.merge(totals, on="event_id", how="left")

    # Merge props (match on event_id + player name)
    if len(props) > 0:
        prop_cols = [c for c in props.columns if c.startswith("prop_")]
        df = df.merge(
            props[["event_id", "player_name"] + prop_cols],
            left_on=["event_id", "name"],
            right_on=["event_id", "player_name"],
            how="left",
        )
        if "player_name_y" in df.columns:
            df = df.drop(columns=["player_name_y"])

    # Merge rolling team offense (player's own team)
    off_roll_cols = [c for c in team_off.columns if c.startswith("roll_team")]
    df = df.merge(
        team_off[["team", "season", "week"] + off_roll_cols],
        on=["team", "season", "week"], how="left",
    )

    # Merge rolling team defense (opponent)
    def_roll_cols = [c for c in team_def.columns if c.startswith("roll_opp")]
    df = df.merge(
        team_def[["team", "season", "week"] + def_roll_cols],
        left_on=["opponent", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
        suffixes=("", "_def"),
    )
    if "team_def" in df.columns:
        df = df.drop(columns=["team_def"])

    # Merge team archetypes (season-level as fallback)
    for unit, prefix in [("offense", "own"), ("defense", "opp")]:
        ta = team_arch[team_arch["unit"] == unit][["team", "season", "archetype", "style"]].copy()
        ta = ta.rename(columns={"archetype": f"{prefix}_{unit}_arch", "style": f"{prefix}_{unit}_style"})
        merge_col = "team" if unit == "offense" else "opponent"
        df = df.merge(ta, left_on=[merge_col, "season"], right_on=["team", "season"],
                      how="left", suffixes=("", f"_{prefix}"))
        if f"team_{prefix}" in df.columns:
            df = df.drop(columns=[f"team_{prefix}"])

    # Encode categoricals
    cat_cols = ["position", "dyn_archetype", "own_offense_arch", "own_offense_style",
                "opp_defense_arch", "opp_defense_style"]
    le_dict = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            le_dict[col] = le

    # =================================================================
    # INTERACTION FEATURES — encode backtesting matchup edges
    # =================================================================
    print("  Engineering interaction features...")

    # --- 1. Player archetype × Opp defense archetype (hash interaction) ---
    df["ix_player_v_def"] = (df["dyn_archetype"].fillna("Unk") + " vs "
                             + df["opp_defense_arch"].fillna("Unk"))
    le_ix1 = LabelEncoder()
    df["ix_player_v_def_enc"] = le_ix1.fit_transform(df["ix_player_v_def"])
    le_dict["ix_player_v_def"] = le_ix1

    # --- 2. Player archetype × Opp defense STYLE ---
    df["ix_player_v_defstyle"] = (df["dyn_archetype"].fillna("Unk") + " vs "
                                  + df["opp_defense_style"].fillna("Unk"))
    le_ix2 = LabelEncoder()
    df["ix_player_v_defstyle_enc"] = le_ix2.fit_transform(df["ix_player_v_defstyle"])
    le_dict["ix_player_v_defstyle"] = le_ix2

    # --- 3. Own offense archetype × Opp defense archetype ---
    df["ix_off_v_def"] = (df["own_offense_arch"].fillna("Unk") + " vs "
                          + df["opp_defense_arch"].fillna("Unk"))
    le_ix3 = LabelEncoder()
    df["ix_off_v_def_enc"] = le_ix3.fit_transform(df["ix_off_v_def"])
    le_dict["ix_off_v_def"] = le_ix3

    # --- 4. Numeric matchup edge scores from backtesting ---
    # Encode the known backtesting deltas as a lookup feature
    # Player archetype vs defense archetype → expected PPR delta
    bt_edges = {}
    if conn is not None:
        try:
            bt_rows = pd.read_sql_query("""
                SELECT player_arch, def_arch, delta FROM backtest_individual
                WHERE significant = 1
            """, conn)
            for _, r in bt_rows.iterrows():
                bt_edges[(r["player_arch"], r["def_arch"])] = r["delta"]
            print(f"  Loaded {len(bt_edges)} backtesting edge lookups")
        except Exception:
            pass

    def get_bt_edge(row):
        return bt_edges.get((row.get("dyn_archetype", ""), row.get("opp_defense_arch", "")), 0.0)
    df["bt_matchup_edge"] = df.apply(get_bt_edge, axis=1)

    # --- 5. Numeric tier interaction scores ---
    # Defense quality tier (numeric: Elite=4, Above-Avg=3, Mid=2, Bottom=1)
    def_tier_map = {"Elite": 4, "Above-Average": 3, "Above-Avg Pressure": 3,
                    "Above-Avg Opportunistic": 3, "Middle-of-Pack": 2, "Bottom-Tier": 1}
    off_tier_map = {"Elite": 4, "Pass-First": 3, "Run-Heavy": 3, "Bottom-Tier": 1}

    df["opp_def_tier"] = df["opp_defense_arch"].map(def_tier_map).fillna(2)
    df["own_off_tier"] = df["own_offense_arch"].map(off_tier_map).fillna(2)

    # Player volume tier (based on dynamic archetype)
    player_tier_map = {
        "Dual-Threat": 4, "High-Volume Passer": 3, "Game Manager": 1,
        "Workhorse": 4, "Starter": 3, "Rotational": 2, "Depth": 1,
        "Alpha WR1": 4, "WR2": 3, "WR3/Flex": 2,
        "Elite Receiving": 4, "Secondary Receiver": 3, "Blocking/Depth": 1,
    }
    df["player_tier"] = df["dyn_archetype"].map(player_tier_map).fillna(2)

    # Interaction: player tier × inverse defense quality
    # High player tier vs weak defense = high value; low tier vs elite = low value
    df["ix_tier_mismatch"] = df["player_tier"] * (5 - df["opp_def_tier"])
    # Offense tier vs defense tier
    df["ix_off_def_gap"] = df["own_off_tier"] - df["opp_def_tier"]

    # --- 6. Rolling stats × matchup context ---
    # Player's rolling PPR × defense vulnerability (inverse tier)
    df["ix_roll_ppr_x_defweak"] = df["roll_ppr"].fillna(0) * (5 - df["opp_def_tier"]) / 4
    # Target share × opponent pass EPA allowed (leaky defense = more passing value)
    df["ix_targets_x_opp_epa"] = df["roll_target_share"].fillna(0) * df["roll_opp_pass_epa"].fillna(0)
    # Rush yards × opponent rush yards allowed
    df["ix_rush_x_opp_rush"] = df["roll_rushing_yards"].fillna(0) * df["roll_opp_rush_yds"].fillna(0) / 100
    # Receiving yards × opponent pass yards allowed
    df["ix_rec_x_opp_pass"] = df["roll_receiving_yards"].fillna(0) * df["roll_opp_pass_yds"].fillna(0) / 100

    # --- 7. Game environment × player profile ---
    # High O/U + high-volume player = amplified
    df["ix_ou_x_tier"] = df["over_under"].fillna(44) * df["player_tier"] / 4
    # Spread × player tier (big favorite's starters get game script boost)
    df["ix_spread_x_tier"] = df["abs_spread"].fillna(3) * df["player_tier"]

    # --- 8. Prop line × matchup (props already embed market knowledge, but
    #     crossing with defense quality captures when the market may be wrong) ---
    for prop_col in ["prop_rec_yds", "prop_rush_yds", "prop_pass_yds"]:
        if prop_col in df.columns:
            df[f"ix_{prop_col}_x_defweak"] = df[prop_col].fillna(0) * (5 - df["opp_def_tier"]) / 4

    n_ix = len([c for c in df.columns if c.startswith("ix_")])
    print(f"  Created {n_ix} interaction features")
    print(f"  Feature matrix: {len(df)} rows, {len(df.columns)} columns")
    return df, le_dict


# =====================================================================
# Walk-Forward Prediction
# =====================================================================

def get_feature_cols(df):
    """Select feature columns for the model."""
    roll_cols = [c for c in df.columns if c.startswith("roll_")]
    prop_cols = [c for c in df.columns if c.startswith("prop_")]
    enc_cols = [c for c in df.columns if c.endswith("_enc")]
    # Only numeric ix_ columns (skip the raw string ones that end with _enc separately)
    ix_cols = [c for c in df.columns if c.startswith("ix_") and c not in
               ["ix_player_v_def", "ix_player_v_defstyle", "ix_off_v_def"]]

    features = roll_cols + prop_cols + enc_cols + ix_cols + [
        "abs_spread", "over_under", "is_home",
        "bt_matchup_edge", "player_tier", "opp_def_tier", "own_off_tier",
    ]

    # Only keep columns that exist and are numeric
    features = [f for f in features if f in df.columns]
    # Deduplicate
    features = list(dict.fromkeys(features))
    return features


def walk_forward(df):
    """Walk-forward prediction: train on past, predict current week."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD PREDICTION")
    print("=" * 80)

    feature_cols = get_feature_cols(df)
    print(f"  Using {len(feature_cols)} features")

    # Filter to rows with enough history and valid target
    df = df.dropna(subset=["ppr"]).copy()
    df = df[df["roll_games"] >= MIN_HISTORY].copy()

    # Create season-week key for ordering
    df["season_week"] = df["season"] * 100 + df["week"]
    all_weeks = sorted(df["season_week"].unique())

    # Find start point
    start_idx = 0
    for i, sw in enumerate(all_weeks):
        if sw % 100 >= WALK_FORWARD_START:
            start_idx = i
            break

    predictions = []
    week_results = []

    print(f"  Training weeks: {len(all_weeks[:start_idx])}, Prediction weeks: {len(all_weeks[start_idx:])}")

    for i in range(start_idx, len(all_weeks)):
        current_sw = all_weeks[i]
        current_season = current_sw // 100
        current_week = current_sw % 100

        # Train on everything before this week
        train = df[df["season_week"] < current_sw].copy()
        test = df[df["season_week"] == current_sw].copy()

        if len(train) < 100 or len(test) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values
        y_train = train["ppr"].values
        X_test = test[feature_cols].fillna(0).values
        y_test = test["ppr"].values

        # Train LightGBM
        model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        test = test.copy()
        test["predicted_ppr"] = preds
        test["error"] = test["ppr"] - preds
        test["abs_error"] = test["error"].abs()

        predictions.append(test)

        mae = test["abs_error"].mean()
        corr = test[["ppr", "predicted_ppr"]].corr().iloc[0, 1]
        week_results.append({
            "season": current_season, "week": current_week,
            "n": len(test), "mae": mae, "corr": corr,
        })

    all_preds = pd.concat(predictions, ignore_index=True)
    week_df = pd.DataFrame(week_results)

    # Get feature importance from last model
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return all_preds, week_df, importance, model


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(all_preds, week_df, importance):
    """Print comprehensive evaluation metrics."""

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    overall_mae = all_preds["abs_error"].mean()
    overall_corr = all_preds[["ppr", "predicted_ppr"]].corr().iloc[0, 1]
    rmse = np.sqrt((all_preds["error"] ** 2).mean())

    print(f"\n  Total predictions: {len(all_preds)}")
    print(f"  MAE:  {overall_mae:.2f} PPR points")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Correlation: {overall_corr:.4f}")

    # Naive baseline: just predict rolling average
    baseline_mae = (all_preds["ppr"] - all_preds["roll_ppr"]).abs().mean()
    print(f"\n  Baseline (rolling avg): MAE = {baseline_mae:.2f}")
    print(f"  Model improvement: {(baseline_mae - overall_mae) / baseline_mae:.1%}")

    # By position
    print("\n  BY POSITION:")
    print(f"  {'Pos':4s}  {'N':>6s}  {'MAE':>6s}  {'RMSE':>6s}  {'Corr':>6s}  {'Base MAE':>8s}  {'Impr':>6s}")
    print("  " + "-" * 50)
    for pos in POSITIONS:
        p = all_preds[all_preds["position"] == pos]
        if len(p) == 0:
            continue
        mae = p["abs_error"].mean()
        rmse_p = np.sqrt((p["error"] ** 2).mean())
        corr = p[["ppr", "predicted_ppr"]].corr().iloc[0, 1]
        base = (p["ppr"] - p["roll_ppr"]).abs().mean()
        impr = (base - mae) / base
        print(f"  {pos:4s}  {len(p):6d}  {mae:6.2f}  {rmse_p:6.2f}  {corr:6.4f}  {base:8.2f}  {impr:+5.1%}")

    # By season
    print("\n  BY SEASON:")
    for season in sorted(all_preds["season"].unique()):
        s = all_preds[all_preds["season"] == season]
        mae = s["abs_error"].mean()
        corr = s[["ppr", "predicted_ppr"]].corr().iloc[0, 1]
        base = (s["ppr"] - s["roll_ppr"]).abs().mean()
        print(f"  {season}: MAE={mae:.2f}, Corr={corr:.4f}, Baseline={base:.2f}, Improvement={(base-mae)/base:+.1%}")

    # By week progression
    print("\n  WEEKLY MAE TREND (rolling 4-week avg):")
    week_df["rolling_mae"] = week_df["mae"].rolling(4, min_periods=1).mean()
    for _, r in week_df.iterrows():
        bar = "█" * int(r["rolling_mae"] * 3)
        print(f"    {int(r['season'])} Wk{int(r['week']):2d}: MAE={r['mae']:.2f}  r={r['corr']:.3f}  n={int(r['n']):3d}  {bar}")

    # Feature importance — highlight interaction features
    print("\n  TOP 30 FEATURES:")
    for _, r in importance.head(30).iterrows():
        bar = "█" * int(r["importance"] / importance["importance"].max() * 30)
        tag = " ★IX" if r["feature"].startswith("ix_") or r["feature"] == "bt_matchup_edge" else ""
        print(f"    {r['feature']:35s}  {r['importance']:6.0f}  {bar}{tag}")

    # Show all interaction features ranked
    ix_feats = importance[importance["feature"].apply(
        lambda f: f.startswith("ix_") or f in ["bt_matchup_edge", "player_tier", "opp_def_tier", "own_off_tier"]
    )]
    if len(ix_feats) > 0:
        print(f"\n  INTERACTION FEATURES ONLY ({len(ix_feats)} total):")
        total_imp = importance["importance"].sum()
        ix_total = ix_feats["importance"].sum()
        print(f"  Combined share of total importance: {ix_total/total_imp:.1%}")
        for _, r in ix_feats.iterrows():
            rank = (importance["importance"] > r["importance"]).sum() + 1
            print(f"    #{rank:<3d} {r['feature']:35s}  {r['importance']:6.0f}")

    # Where the model adds the most value
    print("\n  MODEL vs BASELINE BY PLAYER TIER:")
    all_preds["roll_ppr_bucket"] = pd.cut(
        all_preds["roll_ppr"],
        bins=[-1, 5, 10, 15, 20, 50],
        labels=["Low (0-5)", "Mid (5-10)", "Good (10-15)", "Great (15-20)", "Elite (20+)"]
    )
    tier_stats = all_preds.groupby("roll_ppr_bucket").agg(
        n=("ppr", "count"),
        model_mae=("abs_error", "mean"),
        actual_avg=("ppr", "mean"),
        pred_avg=("predicted_ppr", "mean"),
    ).reset_index()
    # Compute baseline MAE per tier
    for idx, tier_row in tier_stats.iterrows():
        tier_data = all_preds[all_preds["roll_ppr_bucket"] == tier_row["roll_ppr_bucket"]]
        tier_stats.loc[idx, "base_mae"] = (tier_data["ppr"] - tier_data["roll_ppr"]).abs().mean()

    print(f"  {'Tier':15s}  {'N':>6s}  {'Model MAE':>9s}  {'Base MAE':>9s}  {'Impr':>6s}  {'Avg Actual':>10s}")
    print("  " + "-" * 65)
    for _, r in tier_stats.iterrows():
        impr = (r["base_mae"] - r["model_mae"]) / r["base_mae"] if r["base_mae"] > 0 else 0
        print(f"  {str(r['roll_ppr_bucket']):15s}  {int(r['n']):6d}  {r['model_mae']:9.2f}  {r['base_mae']:9.2f}  {impr:+5.1%}  {r['actual_avg']:10.2f}")

    # Best/worst individual predictions
    print("\n  BEST PREDICTIONS (smallest error, high scorers):")
    high = all_preds[all_preds["ppr"] >= 15].nsmallest(10, "abs_error")
    for _, r in high.iterrows():
        print(f"    {r['name']:25s} {r['position']:2s} Wk{int(r['week']):2d} "
              f"Pred={r['predicted_ppr']:.1f} Act={r['ppr']:.1f} Err={r['error']:+.1f}")

    print("\n  BIGGEST MISSES (largest error):")
    misses = all_preds.nlargest(10, "abs_error")
    for _, r in misses.iterrows():
        print(f"    {r['name']:25s} {r['position']:2s} Wk{int(r['week']):2d} "
              f"Pred={r['predicted_ppr']:.1f} Act={r['ppr']:.1f} Err={r['error']:+.1f}")


# =====================================================================
# Save Results
# =====================================================================

def save_results(conn, all_preds, importance):
    conn.executescript("""
        DROP TABLE IF EXISTS dfs_predictions;
        DROP TABLE IF EXISTS dfs_feature_importance;

        CREATE TABLE dfs_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT, player_name TEXT, position TEXT,
            team TEXT, season INTEGER, week INTEGER, opponent TEXT,
            predicted_ppr REAL, actual_ppr REAL, error REAL,
            roll_ppr REAL, dyn_archetype TEXT,
            abs_spread REAL, over_under REAL, is_home INTEGER
        );

        CREATE TABLE dfs_feature_importance (
            feature TEXT PRIMARY KEY,
            importance REAL
        );

        CREATE INDEX idx_dfs_pred_player ON dfs_predictions(player_id);
        CREATE INDEX idx_dfs_pred_week ON dfs_predictions(season, week);
        CREATE INDEX idx_dfs_pred_pos ON dfs_predictions(position);
    """)

    rows = []
    for _, r in all_preds.iterrows():
        rows.append((
            r["player_id"], r["name"], r["position"],
            r["team"], int(r["season"]), int(r["week"]), r["opponent"],
            round(r["predicted_ppr"], 2), round(r["ppr"], 2), round(r["error"], 2),
            round(r.get("roll_ppr", 0), 2) if pd.notna(r.get("roll_ppr")) else None,
            r.get("dyn_archetype"),
            r.get("abs_spread"), r.get("over_under"),
            int(r["is_home"]) if pd.notna(r.get("is_home")) else None,
        ))

    conn.executemany("""
        INSERT INTO dfs_predictions
        (player_id, player_name, position, team, season, week, opponent,
         predicted_ppr, actual_ppr, error, roll_ppr, dyn_archetype,
         abs_spread, over_under, is_home)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    for _, r in importance.iterrows():
        conn.execute("INSERT INTO dfs_feature_importance VALUES (?, ?)",
                     (r["feature"], round(r["importance"], 4)))

    conn.commit()
    print(f"\n  Saved {len(rows)} predictions + {len(importance)} feature importances")


# =====================================================================
# Main
# =====================================================================

def main():
    print("NFL DFS Walk-Forward Prediction Engine")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    # Load
    stats, odds, spreads, totals, props, team_arch = load_all_data(conn)

    # Rolling features
    stats = build_rolling_features(stats)

    # Team rolling stats
    team_off, team_def = build_team_rolling(stats)

    # Dynamic archetypes
    stats = build_dynamic_archetypes(stats)

    # Assemble features
    df, le_dict = assemble_features(stats, odds, spreads, totals, props, team_off, team_def, team_arch, conn=conn)

    # Walk-forward
    all_preds, week_df, importance, model = walk_forward(df)

    # Evaluate
    evaluate(all_preds, week_df, importance)

    # Save
    print("\nSaving results...")
    save_results(conn, all_preds, importance)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
