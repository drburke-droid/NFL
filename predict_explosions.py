"""
NFL DFS Explosion Prediction
Predicts when a player will score 2+ standard deviations above their
rolling average — the "boom" games that win DFS tournaments.

Uses the same walk-forward framework and features as predict_dfs.py,
but with a classification target (explosion yes/no) and metrics tuned
for precision/recall tradeoff.
"""

import sqlite3
import os
import sys
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")

ROLLING_WINDOW = 6
ROLLING_ALPHA = 0.65
MIN_HISTORY = 3
WALK_FORWARD_START = 5
POSITIONS = ["QB", "RB", "WR", "TE"]
EXPLOSION_THRESHOLD = 2.0  # standard deviations

# Minimum rolling PPR to be considered a "relevant" player for DFS
MIN_ROLL_PPR = 3.0


def load_data(conn):
    """Load player stats and all contextual data."""
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
        ORDER BY ps.player_id, ps.season, ps.week
    """, conn)

    odds = pd.read_sql_query("""
        SELECT game_id, event_id, season, week, home_team, away_team, game_script
        FROM game_scripts
    """, conn)

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

    props = pd.read_sql_query("""
        SELECT event_id, player_name, market, point as prop_line
        FROM player_props WHERE bookmaker='draftkings' AND outcome_type='Over'
    """, conn)
    props_wide = props.pivot_table(
        index=["event_id", "player_name"], columns="market",
        values="prop_line", aggfunc="first"
    ).reset_index()
    props_wide.columns.name = None
    prop_rename = {
        "player_pass_yds": "prop_pass_yds", "player_pass_tds": "prop_pass_tds",
        "player_rush_yds": "prop_rush_yds", "player_reception_yds": "prop_rec_yds",
        "player_receptions": "prop_receptions", "player_rush_reception_yds": "prop_rush_rec_yds",
        "player_anytime_td": "prop_anytime_td",
    }
    props_wide = props_wide.rename(columns=prop_rename)

    team_arch = pd.read_sql_query(
        "SELECT team, season, unit, archetype, style FROM team_archetypes", conn
    )

    bt_edges = {}
    try:
        for _, r in pd.read_sql_query(
            "SELECT player_arch, def_arch, delta FROM backtest_individual WHERE significant=1", conn
        ).iterrows():
            bt_edges[(r["player_arch"], r["def_arch"])] = r["delta"]
    except Exception:
        pass

    print(f"  {len(stats)} player-games, {len(props_wide)} prop entries")
    return stats, odds, spreads, totals, props_wide, team_arch, bt_edges


def build_features(stats, odds, spreads, totals, props, team_arch, bt_edges, conn=None):
    """Build complete feature set with rolling stats, interactions, and explosion target."""
    print("Building features...")

    stats = stats.sort_values(["player_id", "season", "week"]).copy()

    # --- Rolling player stats ---
    roll_cols = ["ppr", "completions", "attempts", "passing_yards", "passing_tds",
                 "interceptions", "passing_epa", "carries", "rushing_yards", "rushing_tds",
                 "rushing_epa", "targets", "receptions", "receiving_yards", "receiving_tds",
                 "receiving_epa", "target_share"]

    for col in roll_cols:
        stats[f"roll_{col}"] = stats.groupby("player_id")[col].transform(
            lambda x: x.shift(1).ewm(alpha=ROLLING_ALPHA, min_periods=1, adjust=False).mean()
        )

    stats["roll_games"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # --- Volatility / ceiling features (critical for explosion prediction) ---
    stats["roll_ppr_std"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).std()
    )
    stats["roll_ppr_max"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).max()
    )
    stats["roll_ppr_min"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).min()
    )
    # Coefficient of variation (how "boom/bust" is this player?)
    stats["roll_ppr_cv"] = stats["roll_ppr_std"] / stats["roll_ppr"].replace(0, np.nan)

    # Recent trend: last 2 games vs last 6
    stats["roll_ppr_recent"] = stats.groupby("player_id")["ppr"].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean()
    )
    stats["roll_ppr_trend"] = stats["roll_ppr_recent"] - stats["roll_ppr"]

    # Explosion history: how often has this player exploded before?
    stats["prev_explosion"] = stats.groupby("player_id").apply(
        lambda g: ((g["ppr"].shift(1) - g["roll_ppr"].shift(1)) >=
                   EXPLOSION_THRESHOLD * g["roll_ppr_std"].shift(1)).rolling(6, min_periods=1).mean(),
    ).droplevel(0) if len(stats) > 0 else 0

    # Days since last explosion (recency)
    def weeks_since_explosion(g):
        is_exp = ((g["ppr"] - g["roll_ppr"]) >= EXPLOSION_THRESHOLD * g["roll_ppr_std"]) & (g["roll_ppr_std"] > 0)
        result = pd.Series(np.nan, index=g.index)
        last_exp = np.nan
        for i, (idx, row) in enumerate(g.iterrows()):
            result.loc[idx] = last_exp
            if is_exp.loc[idx]:
                last_exp = 0
            elif not np.isnan(last_exp):
                last_exp += 1
        return result

    stats["weeks_since_explosion"] = stats.groupby("player_id").apply(
        weeks_since_explosion
    ).droplevel(0)

    # --- Target variable ---
    stats["explosion"] = (
        ((stats["ppr"] - stats["roll_ppr"]) >= EXPLOSION_THRESHOLD * stats["roll_ppr_std"])
        & (stats["roll_ppr_std"] > 0)
    ).astype(int)

    # --- Team rolling stats ---
    team_off = stats.groupby(["team", "season", "week"]).agg(
        team_pass_yds=("passing_yards", "sum"), team_rush_yds=("rushing_yards", "sum"),
        team_ppr_total=("ppr", "sum"), team_pass_epa=("passing_epa", "sum"),
    ).reset_index()
    for col in ["team_pass_yds", "team_rush_yds", "team_ppr_total", "team_pass_epa"]:
        team_off[f"roll_{col}"] = team_off.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(alpha=ROLLING_ALPHA, min_periods=1, adjust=False).mean()
        )

    team_def = stats.groupby(["opponent", "season", "week"]).agg(
        opp_ppr_total=("ppr", "sum"), opp_pass_epa=("passing_epa", "sum"),
        opp_pass_yds=("passing_yards", "sum"), opp_rush_yds=("rushing_yards", "sum"),
    ).reset_index().rename(columns={"opponent": "team"})
    for col in ["opp_ppr_total", "opp_pass_epa", "opp_pass_yds", "opp_rush_yds"]:
        team_def[f"roll_{col}"] = team_def.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(alpha=ROLLING_ALPHA, min_periods=1, adjust=False).mean()
        )

    # --- Dynamic archetype ---
    arch_features = {
        "QB": ["roll_attempts", "roll_passing_yards", "roll_passing_tds",
               "roll_carries", "roll_rushing_yards", "roll_passing_epa"],
        "RB": ["roll_carries", "roll_rushing_yards", "roll_rushing_tds",
               "roll_targets", "roll_receptions", "roll_receiving_yards", "roll_rushing_epa"],
        "WR": ["roll_targets", "roll_receptions", "roll_receiving_yards",
               "roll_receiving_tds", "roll_target_share", "roll_receiving_epa"],
        "TE": ["roll_targets", "roll_receptions", "roll_receiving_yards",
               "roll_receiving_tds", "roll_target_share", "roll_receiving_epa"],
    }
    arch_labels = {
        "QB": ["Game Manager", "High-Volume Passer", "Dual-Threat"],
        "RB": ["Depth", "Rotational", "Starter", "Workhorse"],
        "WR": ["Depth", "WR3/Flex", "WR2", "Alpha WR1"],
        "TE": ["Blocking/Depth", "Secondary Receiver", "Elite Receiving"],
    }

    stats["dyn_archetype"] = "Unknown"
    for pos in POSITIONS:
        feats = arch_features[pos]
        labels = arch_labels[pos]
        pos_data = stats.loc[stats["position"] == pos, feats].dropna()
        if len(pos_data) < len(labels) * 10:
            continue
        scaler = StandardScaler()
        X = scaler.fit_transform(pos_data.values)
        km = KMeans(n_clusters=len(labels), n_init=20, random_state=42)
        km.fit(X)
        sorted_idx = np.argsort(km.cluster_centers_[:, 0])
        label_map = {sorted_idx[i]: labels[i] for i in range(len(labels))}
        preds = km.predict(X)
        stats.loc[pos_data.index, "dyn_archetype"] = [label_map[p] for p in preds]

    # --- Merge everything ---
    df = stats.copy()

    # Event lookup
    home_lk = odds[["event_id", "season", "week", "home_team", "away_team"]].copy()
    home_lk = home_lk.rename(columns={"home_team": "team"})
    home_lk["is_home"] = 1
    away_lk = odds[["event_id", "season", "week", "away_team", "home_team"]].copy()
    away_lk = away_lk.rename(columns={"away_team": "team"})
    away_lk["is_home"] = 0
    event_lk = pd.concat([home_lk, away_lk], ignore_index=True)
    df = df.merge(event_lk[["team", "season", "week", "event_id", "is_home"]],
                  on=["team", "season", "week"], how="left")

    df = df.merge(spreads, on="event_id", how="left")
    df = df.merge(totals, on="event_id", how="left")

    # Props
    prop_cols = [c for c in props.columns if c.startswith("prop_")]
    if len(props) > 0:
        df = df.merge(props[["event_id", "player_name"] + prop_cols],
                      left_on=["event_id", "name"], right_on=["event_id", "player_name"],
                      how="left")
        if "player_name" in df.columns and "player_name" != "name":
            df = df.drop(columns=["player_name"], errors="ignore")

    # Team stats
    off_cols = [c for c in team_off.columns if c.startswith("roll_")]
    df = df.merge(team_off[["team", "season", "week"] + off_cols],
                  on=["team", "season", "week"], how="left")
    def_cols = [c for c in team_def.columns if c.startswith("roll_")]
    df = df.merge(team_def[["team", "season", "week"] + def_cols],
                  left_on=["opponent", "season", "week"],
                  right_on=["team", "season", "week"], how="left", suffixes=("", "_def"))
    df.drop(columns=["team_def"], errors="ignore", inplace=True)

    # Team archetypes
    for unit, prefix in [("offense", "own"), ("defense", "opp")]:
        ta = team_arch[team_arch["unit"] == unit][["team", "season", "archetype", "style"]].copy()
        ta = ta.rename(columns={"archetype": f"{prefix}_{unit}_arch", "style": f"{prefix}_{unit}_style"})
        merge_col = "team" if unit == "offense" else "opponent"
        df = df.merge(ta, left_on=[merge_col, "season"], right_on=["team", "season"],
                      how="left", suffixes=("", f"_{prefix}"))
        df.drop(columns=[f"team_{prefix}"], errors="ignore", inplace=True)

    # --- Predicted game script features ---
    # Use spread + O/U buckets to compute probability of each script
    print("  Computing predicted game script probabilities...")

    # Build script probability lookup from the prediction factors table
    script_probs_ou = {}
    script_probs_sp = {}
    if conn is not None:
        try:
            spf = pd.read_sql_query("SELECT * FROM script_prediction_factors", conn)
            for _, r in spf.iterrows():
                key = (r["factor_type"], r["factor_value"], r["game_script"])
                if r["factor_type"] == "over_under":
                    script_probs_ou[key] = r["probability"]
                else:
                    script_probs_sp[key] = r["probability"]
        except Exception:
            pass

    # Load game script tendencies (2H play-calling shifts per script × role)
    tendencies = {}
    if conn is not None:
        try:
            for _, r in pd.read_sql_query("SELECT * FROM game_script_tendencies", conn).iterrows():
                tendencies[(r["game_script"], r["role"])] = {
                    "delta_pass_rate": r["delta_pass_rate"],
                    "delta_deep_rate": r["delta_deep_rate"],
                    "delta_air_yards": r["delta_air_yards"],
                    "delta_no_huddle": r["delta_no_huddle"],
                    "delta_epa": r["delta_epa"],
                }
        except Exception:
            pass

    # Bucket each game's O/U and spread
    def ou_bucket(val):
        if pd.isna(val): return None
        if val <= 38: return "Low (≤38)"
        if val <= 42: return "Med-Low (38-42)"
        if val <= 46: return "Medium (42-46)"
        if val <= 50: return "Med-High (46-50)"
        return "High (50+)"

    def sp_bucket(val):
        if pd.isna(val): return None
        val = abs(val)
        if val <= 2.5: return "Pickem (0-2.5)"
        if val <= 5: return "Close (3-5)"
        if val <= 8: return "Moderate (5.5-8)"
        if val <= 15: return "Big (8.5-15)"
        return "Huge (15+)"

    df["_ou_bkt"] = df["over_under"].apply(ou_bucket)
    df["_sp_bkt"] = df["abs_spread"].apply(sp_bucket)

    # All script types
    all_scripts = [
        "Shootout", "High-Scoring Pulled Away", "Wire-to-Wire Blowout",
        "2nd Half Blowout", "Comeback/Competitive", "Steady Build",
        "Low-Scoring Seesaw", "Defensive Slugfest",
    ]

    # Compute blended script probabilities from O/U and spread lookups
    for script in all_scripts:
        col_name = f"gs_prob_{script.replace(' ', '_').replace('-', '_').replace('/', '_').lower()}"
        probs = []
        for _, row in df.iterrows():
            p_ou = script_probs_ou.get(("over_under", row["_ou_bkt"], script), None)
            p_sp = script_probs_sp.get(("spread", row["_sp_bkt"], script), None)
            if p_ou is not None and p_sp is not None:
                probs.append((p_ou + p_sp) / 2)  # simple blend
            elif p_ou is not None:
                probs.append(p_ou)
            elif p_sp is not None:
                probs.append(p_sp)
            else:
                probs.append(0.125)  # uniform prior (1/8)
        df[col_name] = probs

    df.drop(columns=["_ou_bkt", "_sp_bkt"], inplace=True)

    # Composite game environment scores from script probabilities × tendencies
    # For each player: expected 2H pass rate delta, weighted by script probs
    # Determine if player is likely on the favored or underdog side
    # If spread < 0, home is favored. Player on home team → likely "winner" role
    df["likely_role"] = "loser"
    df.loc[(df["is_home"] == 1) & (df["abs_spread"].fillna(0) > 0), "likely_role"] = "winner"
    df.loc[(df["is_home"] == 0) & (df["abs_spread"].fillna(0) > 0), "likely_role"] = "loser"
    # If spread is close, could be either
    df.loc[df["abs_spread"].fillna(99) <= 1.5, "likely_role"] = "neutral"

    # Compute expected tendencies weighted by script probabilities
    for tend_key in ["delta_pass_rate", "delta_deep_rate", "delta_no_huddle", "delta_epa"]:
        vals = []
        for _, row in df.iterrows():
            role = row["likely_role"]
            if role == "neutral":
                role = "winner"  # default
            expected = 0.0
            for script in all_scripts:
                col_name = f"gs_prob_{script.replace(' ', '_').replace('-', '_').replace('/', '_').lower()}"
                prob = row.get(col_name, 0.125)
                tend = tendencies.get((script, role), {}).get(tend_key, 0.0)
                expected += prob * tend
            vals.append(expected)
        df[f"gs_exp_{tend_key}"] = vals

    # Game script composite scores (higher-level)
    gs_shootout = f"gs_prob_shootout"
    gs_blowout_w2w = f"gs_prob_wire_to_wire_blowout"
    gs_blowout_2h = f"gs_prob_2nd_half_blowout"
    gs_highscoring = f"gs_prob_high_scoring_pulled_away"
    gs_defslugs = f"gs_prob_defensive_slugfest"

    # Probability of a high-scoring game (shootout + high-scoring pulled away)
    df["gs_high_scoring_prob"] = df.get(gs_shootout, 0) + df.get(gs_highscoring, 0)
    # Probability of any blowout
    df["gs_blowout_prob"] = df.get(gs_blowout_w2w, 0) + df.get(gs_blowout_2h, 0)
    # Probability of low scoring
    df["gs_low_scoring_prob"] = df.get(gs_defslugs, 0) + df.get("gs_prob_low_scoring_seesaw", 0)

    # Cross script probs with player profile
    df["ix_shootout_x_tier"] = df.get(gs_shootout, 0) * df.get("player_tier", 2)
    df["ix_blowout_x_tier"] = df["gs_blowout_prob"] * df.get("player_tier", 2)
    df["ix_highscoring_x_ppr"] = df["gs_high_scoring_prob"] * df["roll_ppr"].fillna(0)
    df["ix_gs_pass_delta_x_targets"] = df["gs_exp_delta_pass_rate"] * df["roll_target_share"].fillna(0)

    # Drop likely_role string column (not a feature)
    df.drop(columns=["likely_role"], inplace=True)

    gs_count = len([c for c in df.columns if c.startswith("gs_")])
    print(f"  Created {gs_count} game script features")

    # --- Encode categoricals ---
    cat_cols = ["position", "dyn_archetype", "own_offense_arch", "opp_defense_arch",
                "opp_defense_style"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))

    # --- Interaction features ---
    def_tier = {"Elite": 4, "Above-Average": 3, "Above-Avg Pressure": 3,
                "Above-Avg Opportunistic": 3, "Middle-of-Pack": 2, "Bottom-Tier": 1}
    player_tier = {
        "Dual-Threat": 4, "High-Volume Passer": 3, "Game Manager": 1,
        "Workhorse": 4, "Starter": 3, "Rotational": 2, "Depth": 1,
        "Alpha WR1": 4, "WR2": 3, "WR3/Flex": 2,
        "Elite Receiving": 4, "Secondary Receiver": 3, "Blocking/Depth": 1,
    }

    df["opp_def_tier"] = df["opp_defense_arch"].map(def_tier).fillna(2)
    df["player_tier"] = df["dyn_archetype"].map(player_tier).fillna(2)

    # Backtesting edge
    df["bt_edge"] = df.apply(
        lambda r: bt_edges.get((r.get("dyn_archetype", ""), r.get("opp_defense_arch", "")), 0.0), axis=1
    )

    # Interactions tuned for explosion prediction
    df["ix_ceiling_x_defweak"] = df["roll_ppr_max"].fillna(0) * (5 - df["opp_def_tier"]) / 4
    df["ix_vol_x_defweak"] = df["roll_ppr_cv"].fillna(0) * (5 - df["opp_def_tier"])
    df["ix_ppr_x_defweak"] = df["roll_ppr"].fillna(0) * (5 - df["opp_def_tier"]) / 4
    df["ix_ou_x_tier"] = df["over_under"].fillna(44) * df["player_tier"] / 4
    df["ix_trend_x_defweak"] = df["roll_ppr_trend"].fillna(0) * (5 - df["opp_def_tier"])

    for prop_col in ["prop_rec_yds", "prop_rush_yds", "prop_pass_yds"]:
        if prop_col in df.columns:
            df[f"ix_{prop_col}_x_defweak"] = df[prop_col].fillna(0) * (5 - df["opp_def_tier"]) / 4

    print(f"  {len(df)} rows, {df['explosion'].sum()} explosions ({df['explosion'].mean():.1%})")
    return df


def get_features(df):
    """Select feature columns."""
    roll = [c for c in df.columns if c.startswith("roll_")]
    prop = [c for c in df.columns if c.startswith("prop_")]
    enc = [c for c in df.columns if c.endswith("_enc")]
    ix = [c for c in df.columns if c.startswith("ix_")]
    gs = [c for c in df.columns if c.startswith("gs_")]

    feats = roll + prop + enc + ix + gs + [
        "abs_spread", "over_under", "is_home", "bt_edge",
        "player_tier", "opp_def_tier", "prev_explosion", "weeks_since_explosion",
    ]
    feats = [f for f in feats if f in df.columns]
    return list(dict.fromkeys(feats))


def walk_forward_classify(df):
    """Walk-forward classification for explosions."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD EXPLOSION PREDICTION")
    print("=" * 80)

    feat_cols = get_features(df)
    print(f"  Using {len(feat_cols)} features")

    # Filter
    df = df.dropna(subset=["ppr", "roll_ppr_std"]).copy()
    df = df[df["roll_games"] >= MIN_HISTORY].copy()
    df = df[df["roll_ppr_std"] > 0].copy()
    # Focus on DFS-relevant players
    df = df[df["roll_ppr"] >= MIN_ROLL_PPR].copy()

    df["season_week"] = df["season"] * 100 + df["week"]
    all_weeks = sorted(df["season_week"].unique())

    start_idx = next(i for i, sw in enumerate(all_weeks) if sw % 100 >= WALK_FORWARD_START)

    predictions = []

    print(f"  Data: {len(df)} player-games, {df['explosion'].sum()} explosions ({df['explosion'].mean():.1%})")
    print(f"  Train weeks: {len(all_weeks[:start_idx])}, Predict weeks: {len(all_weeks[start_idx:])}")

    for i in range(start_idx, len(all_weeks)):
        sw = all_weeks[i]
        train = df[df["season_week"] < sw]
        test = df[df["season_week"] == sw]

        if len(train) < 100 or len(test) == 0 or train["explosion"].sum() < 10:
            continue

        X_train = train[feat_cols].fillna(0).values
        y_train = train["explosion"].values
        X_test = test[feat_cols].fillna(0).values

        # Use scale_pos_weight to handle class imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()

        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=30, reg_alpha=0.2, reg_lambda=0.2,
            scale_pos_weight=neg / max(pos, 1),
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        test = test.copy()
        test["explosion_prob"] = probs
        predictions.append(test)

    all_preds = pd.concat(predictions, ignore_index=True)

    # Feature importance from last model
    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return all_preds, importance


def evaluate(preds, importance):
    """Comprehensive evaluation of explosion predictions."""
    print("\n" + "=" * 80)
    print("EXPLOSION PREDICTION RESULTS")
    print("=" * 80)

    actual = preds["explosion"].values
    probs = preds["explosion_prob"].values

    auc = roc_auc_score(actual, probs)
    ap = average_precision_score(actual, probs)
    print(f"\n  Total predictions: {len(preds)}")
    print(f"  Actual explosions: {actual.sum()} ({actual.mean():.1%})")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")

    # Find optimal thresholds for different DFS strategies
    print("\n  THRESHOLD ANALYSIS:")
    print(f"  {'Threshold':>10s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>6s}  {'Flagged':>7s}  {'True Exp':>8s}  {'Strategy'}")
    print("  " + "-" * 80)

    for thresh, strategy in [(0.10, "Wide net (cash games)"),
                              (0.15, "Moderate"),
                              (0.20, "Selective"),
                              (0.25, "Aggressive (GPP)"),
                              (0.30, "High-conviction"),
                              (0.40, "Ultra-selective")]:
        pred_pos = (probs >= thresh)
        if pred_pos.sum() == 0:
            continue
        prec = precision_score(actual, pred_pos)
        rec = recall_score(actual, pred_pos)
        f1 = f1_score(actual, pred_pos)
        print(f"  {thresh:10.2f}  {prec:8.1%}  {rec:6.1%}  {f1:6.3f}  "
              f"{pred_pos.sum():7d}  {(pred_pos & actual.astype(bool)).sum():8d}  {strategy}")

    # By position
    print("\n  BY POSITION:")
    print(f"  {'Pos':4s}  {'N':>6s}  {'Exp Rate':>8s}  {'AUC':>6s}  {'AP':>6s}  {'P@20%':>6s}  {'R@20%':>6s}")
    print("  " + "-" * 55)
    for pos in POSITIONS:
        p = preds[preds["position"] == pos]
        if p["explosion"].sum() < 5:
            continue
        pos_auc = roc_auc_score(p["explosion"], p["explosion_prob"])
        pos_ap = average_precision_score(p["explosion"], p["explosion_prob"])
        pred_pos = p["explosion_prob"] >= 0.20
        prec = precision_score(p["explosion"], pred_pos) if pred_pos.sum() > 0 else 0
        rec = recall_score(p["explosion"], pred_pos) if pred_pos.sum() > 0 else 0
        print(f"  {pos:4s}  {len(p):6d}  {p['explosion'].mean():7.1%}  "
              f"{pos_auc:6.4f}  {pos_ap:6.4f}  {prec:5.1%}  {rec:5.1%}")

    # By archetype
    print("\n  EXPLOSION RATE BY ARCHETYPE (actual vs model predicted):")
    arch_stats = preds.groupby("dyn_archetype").agg(
        n=("explosion", "count"),
        actual_rate=("explosion", "mean"),
        avg_prob=("explosion_prob", "mean"),
        explosions=("explosion", "sum"),
    ).sort_values("actual_rate", ascending=False)
    arch_stats = arch_stats[arch_stats["n"] >= 50]

    print(f"  {'Archetype':25s}  {'N':>5s}  {'Actual':>7s}  {'Predicted':>9s}  {'Explosions':>10s}")
    print("  " + "-" * 65)
    for idx, r in arch_stats.iterrows():
        print(f"  {idx:25s}  {int(r['n']):5d}  {r['actual_rate']:6.1%}  "
              f"{r['avg_prob']:8.1%}  {int(r['explosions']):10d}")

    # Against defense archetypes
    print("\n  EXPLOSION RATE BY OPPONENT DEFENSE:")
    def_stats = preds.groupby("opp_defense_arch").agg(
        n=("explosion", "count"),
        actual_rate=("explosion", "mean"),
        avg_prob=("explosion_prob", "mean"),
    ).sort_values("actual_rate", ascending=False)
    def_stats = def_stats[def_stats["n"] >= 50]

    print(f"  {'Defense':25s}  {'N':>5s}  {'Actual':>7s}  {'Predicted':>9s}")
    print("  " + "-" * 50)
    for idx, r in def_stats.iterrows():
        print(f"  {idx:25s}  {int(r['n']):5d}  {r['actual_rate']:6.1%}  {r['avg_prob']:8.1%}")

    # Top features
    print("\n  TOP 25 FEATURES:")
    for _, r in importance.head(25).iterrows():
        bar = "█" * int(r["importance"] / importance["importance"].max() * 25)
        tag = " ★" if r["feature"].startswith("ix_") or "explosion" in r["feature"] or r["feature"] == "bt_edge" else ""
        print(f"    {r['feature']:35s}  {r['importance']:5.0f}  {bar}{tag}")

    # Calibration: when model says 30%+, how often does it happen?
    print("\n  CALIBRATION (model confidence vs actual rate):")
    preds_c = preds.copy()
    preds_c["prob_bucket"] = pd.cut(preds_c["explosion_prob"],
                                     bins=[0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
                                     labels=["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"])
    cal = preds_c.groupby("prob_bucket").agg(
        n=("explosion", "count"),
        actual_rate=("explosion", "mean"),
        avg_prob=("explosion_prob", "mean"),
    )
    print(f"  {'Bucket':10s}  {'N':>6s}  {'Actual Rate':>11s}  {'Avg Predicted':>13s}")
    print("  " + "-" * 45)
    for idx, r in cal.iterrows():
        print(f"  {str(idx):10s}  {int(r['n']):6d}  {r['actual_rate']:10.1%}  {r['avg_prob']:12.1%}")

    # Show example high-confidence explosions that hit
    print("\n  HIGH-CONFIDENCE EXPLOSIONS THAT HIT (prob ≥ 25%, actually exploded):")
    hits = preds[(preds["explosion_prob"] >= 0.25) & (preds["explosion"] == 1)]
    hits = hits.nlargest(15, "explosion_prob")
    for _, r in hits.iterrows():
        print(f"    {r['name']:25s} {r['position']:2s} {int(r['season'])} Wk{int(r['week']):2d}  "
              f"prob={r['explosion_prob']:.0%}  actual={r['ppr']:.1f} PPR "
              f"(avg={r['roll_ppr']:.1f} +{r['ppr']-r['roll_ppr']:.1f})")

    print("\n  HIGH-CONFIDENCE MISSES (prob ≥ 30%, did NOT explode):")
    misses = preds[(preds["explosion_prob"] >= 0.30) & (preds["explosion"] == 0)]
    misses = misses.nlargest(10, "explosion_prob")
    for _, r in misses.iterrows():
        print(f"    {r['name']:25s} {r['position']:2s} {int(r['season'])} Wk{int(r['week']):2d}  "
              f"prob={r['explosion_prob']:.0%}  actual={r['ppr']:.1f} PPR (avg={r['roll_ppr']:.1f})")

    return preds


def save_results(conn, preds, importance):
    conn.executescript("""
        DROP TABLE IF EXISTS explosion_predictions;
        DROP TABLE IF EXISTS explosion_feature_importance;

        CREATE TABLE explosion_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT, player_name TEXT, position TEXT,
            team TEXT, season INTEGER, week INTEGER, opponent TEXT,
            explosion_prob REAL, actual_explosion INTEGER,
            actual_ppr REAL, roll_ppr REAL, roll_ppr_std REAL,
            dyn_archetype TEXT, opp_defense_arch TEXT
        );
        CREATE INDEX idx_exp_prob ON explosion_predictions(explosion_prob);
        CREATE INDEX idx_exp_season ON explosion_predictions(season, week);

        CREATE TABLE explosion_feature_importance (
            feature TEXT PRIMARY KEY, importance REAL
        );
    """)

    rows = []
    for _, r in preds.iterrows():
        rows.append((
            r["player_id"], r["name"], r["position"], r["team"],
            int(r["season"]), int(r["week"]), r["opponent"],
            round(r["explosion_prob"], 4), int(r["explosion"]),
            round(r["ppr"], 2),
            round(r["roll_ppr"], 2) if pd.notna(r.get("roll_ppr")) else None,
            round(r["roll_ppr_std"], 2) if pd.notna(r.get("roll_ppr_std")) else None,
            r.get("dyn_archetype"), r.get("opp_defense_arch"),
        ))
    conn.executemany("""
        INSERT INTO explosion_predictions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    for _, r in importance.iterrows():
        conn.execute("INSERT INTO explosion_feature_importance VALUES (?,?)",
                     (r["feature"], round(r["importance"], 4)))
    conn.commit()
    print(f"\n  Saved {len(rows)} predictions + {len(importance)} feature importances")


def main():
    print("NFL DFS Explosion Prediction")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    stats, odds, spreads, totals, props, team_arch, bt_edges = load_data(conn)
    df = build_features(stats, odds, spreads, totals, props, team_arch, bt_edges, conn=conn)
    preds, importance = walk_forward_classify(df)
    preds = evaluate(preds, importance)

    print("\nSaving...")
    save_results(conn, preds, importance)
    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
