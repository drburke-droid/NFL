"""
NFL Game Script Prediction
Tests which pre-game features predict game scripts:
  1. Betting lines (spread, over/under, moneyline)
  2. Previous game scripts for each team
  3. Team archetype matchups (offense vs defense)
  4. Home/away

Uses Random Forest + feature importance to find the strongest signals,
then builds a practical lookup table of script probabilities by feature bucket.
"""

import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")


def load_features(conn):
    """Build feature matrix: one row per game with all pre-game info."""

    # -- Base: game scripts with odds --
    # Get spread and total from DraftKings (or first available book)
    games = pd.read_sql_query("""
        SELECT gs.game_id, gs.event_id, gs.season, gs.week,
               gs.home_team, gs.away_team, gs.game_script,
               gs.total_pts, gs.final_margin
        FROM game_scripts gs
        WHERE gs.event_id IS NOT NULL
    """, conn)

    # -- Spread (home team perspective, negative = home favored) --
    spreads = pd.read_sql_query("""
        SELECT DISTINCT go.event_id,
               go.point as spread
        FROM game_odds go
        WHERE go.market = 'spreads'
        AND go.point < 0
        AND go.bookmaker = 'draftkings'
    """, conn)
    # If multiple, take first
    spreads = spreads.drop_duplicates(subset="event_id", keep="first")

    # -- Over/Under --
    totals = pd.read_sql_query("""
        SELECT DISTINCT go.event_id,
               go.point as over_under
        FROM game_odds go
        WHERE go.market = 'totals' AND go.outcome_name = 'Over'
        AND go.bookmaker = 'draftkings'
    """, conn)
    totals = totals.drop_duplicates(subset="event_id", keep="first")

    # -- Moneyline (convert to implied probability) --
    ml = pd.read_sql_query("""
        SELECT go.event_id, go.outcome_name, go.price as ml_price
        FROM game_odds go
        WHERE go.market = 'h2h' AND go.bookmaker = 'draftkings'
    """, conn)

    # Get home team moneyline
    # Need to figure out which outcome_name is the home team
    home_teams = games.set_index("event_id")["home_team"].to_dict()

    # nflverse uses abbreviations, odds DB uses full names
    ABBREV_TO_FULL = {
        "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
        "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
        "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
        "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
        "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
        "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    }

    # Get favorite ML implied win probability
    def ml_to_prob(ml_price):
        if ml_price < 0:
            return abs(ml_price) / (abs(ml_price) + 100)
        else:
            return 100 / (ml_price + 100)

    fav_probs = {}
    for eid, group in ml.groupby("event_id"):
        if len(group) >= 2:
            probs = group["ml_price"].apply(ml_to_prob)
            fav_probs[eid] = probs.max()  # Favorite's implied probability

    # -- Team archetypes --
    team_arch = pd.read_sql_query("""
        SELECT team, season, unit, archetype, style
        FROM team_archetypes
    """, conn)

    home_off = team_arch[team_arch["unit"] == "offense"].rename(
        columns={"archetype": "home_off_arch", "style": "home_off_style", "team": "home_team"}
    )[["home_team", "season", "home_off_arch", "home_off_style"]]

    home_def = team_arch[team_arch["unit"] == "defense"].rename(
        columns={"archetype": "home_def_arch", "style": "home_def_style", "team": "home_team"}
    )[["home_team", "season", "home_def_arch", "home_def_style"]]

    away_off = team_arch[team_arch["unit"] == "offense"].rename(
        columns={"archetype": "away_off_arch", "style": "away_off_style", "team": "away_team"}
    )[["away_team", "season", "away_off_arch", "away_off_style"]]

    away_def = team_arch[team_arch["unit"] == "defense"].rename(
        columns={"archetype": "away_def_arch", "style": "away_def_style", "team": "away_team"}
    )[["away_team", "season", "away_def_arch", "away_def_style"]]

    # -- Previous game script for each team --
    prev_scripts = pd.read_sql_query("""
        SELECT game_id, season, week, home_team, away_team, game_script
        FROM game_scripts ORDER BY season, week
    """, conn)

    # Build lag features per team
    team_games = []
    for _, row in prev_scripts.iterrows():
        team_games.append({"team": row["home_team"], "season": row["season"],
                           "week": row["week"], "script": row["game_script"], "role": "home"})
        team_games.append({"team": row["away_team"], "season": row["season"],
                           "week": row["week"], "script": row["game_script"], "role": "away"})

    tg = pd.DataFrame(team_games).sort_values(["team", "season", "week"])
    tg["prev_script"] = tg.groupby("team")["script"].shift(1)
    # Also: was the team winner or loser last game?
    # Streak: count consecutive same-scripts

    home_prev = tg[tg["role"] == "home"][["team", "season", "week", "prev_script"]].rename(
        columns={"team": "home_team", "prev_script": "home_prev_script"}
    )
    away_prev = tg[tg["role"] == "away"][["team", "season", "week", "prev_script"]].rename(
        columns={"team": "away_team", "prev_script": "away_prev_script"}
    )

    # -- Merge everything --
    df = games.copy()
    df = df.merge(spreads, on="event_id", how="left")
    df = df.merge(totals, on="event_id", how="left")
    df["fav_win_prob"] = df["event_id"].map(fav_probs)
    df["abs_spread"] = df["spread"].abs()

    df = df.merge(home_off, on=["home_team", "season"], how="left")
    df = df.merge(home_def, on=["home_team", "season"], how="left")
    df = df.merge(away_off, on=["away_team", "season"], how="left")
    df = df.merge(away_def, on=["away_team", "season"], how="left")
    df = df.merge(home_prev, on=["home_team", "season", "week"], how="left")
    df = df.merge(away_prev, on=["away_team", "season", "week"], how="left")

    # Derived matchup features
    # Offense tier scoring: Elite=4, Pass-First/Run-Heavy=2, Bottom-Tier=1
    off_tier = {"Elite": 4, "Pass-First": 3, "Run-Heavy": 3, "Bottom-Tier": 1}
    def_tier = {"Elite": 4, "Above-Average": 3, "Above-Avg Pressure": 3,
                "Above-Avg Opportunistic": 3, "Middle-of-Pack": 2, "Bottom-Tier": 1}

    df["home_off_tier"] = df["home_off_arch"].map(off_tier).fillna(2)
    df["away_off_tier"] = df["away_off_arch"].map(off_tier).fillna(2)
    df["home_def_tier"] = df["home_def_arch"].map(def_tier).fillna(2)
    df["away_def_tier"] = df["away_def_arch"].map(def_tier).fillna(2)

    # Matchup quality: offense vs opposing defense
    df["home_matchup"] = df["home_off_tier"] - df["away_def_tier"]  # positive = home offense has edge
    df["away_matchup"] = df["away_off_tier"] - df["home_def_tier"]
    df["matchup_diff"] = df["home_matchup"] - df["away_matchup"]  # positive = home has overall edge
    df["combined_off"] = df["home_off_tier"] + df["away_off_tier"]
    df["combined_def"] = df["home_def_tier"] + df["away_def_tier"]

    print(f"  Feature matrix: {len(df)} games, {len(df.columns)} columns")
    print(f"  Scripts: {df['game_script'].value_counts().to_dict()}")
    print(f"  Spread range: {df['spread'].min():.1f} to {df['spread'].max():.1f}")
    print(f"  O/U range: {df['over_under'].min():.1f} to {df['over_under'].max():.1f}")
    print(f"  Missing values: {df[['spread','over_under','fav_win_prob']].isna().sum().to_dict()}")

    return df


def run_model(df):
    """Train Random Forest to predict game scripts, extract feature importance."""
    print("\n" + "=" * 80)
    print("PREDICTIVE MODEL — RANDOM FOREST")
    print("=" * 80)

    # Encode categoricals
    cat_cols = ["home_off_arch", "home_off_style", "home_def_arch", "home_def_style",
                "away_off_arch", "away_off_style", "away_def_arch", "away_def_style",
                "home_prev_script", "away_prev_script"]

    df_model = df.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = df_model[col].fillna("Unknown")
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    feature_cols = [
        "spread", "abs_spread", "over_under", "fav_win_prob",
        "home_off_tier", "away_off_tier", "home_def_tier", "away_def_tier",
        "home_matchup", "away_matchup", "matchup_diff", "combined_off", "combined_def",
    ] + cat_cols

    # Drop rows with missing critical features
    df_model = df_model.dropna(subset=["spread", "over_under", "fav_win_prob"])
    print(f"\n  Using {len(df_model)} games with complete features")

    X = df_model[feature_cols].values
    y = df_model["game_script"].values

    # Cross-validated accuracy
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=10,
                                random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Baseline (most common class)
    from collections import Counter
    most_common = Counter(y).most_common(1)[0]
    print(f"  Baseline (always predict '{most_common[0]}'): {most_common[1]/len(y):.3f}")

    # Train on full data for feature importance
    rf.fit(X, y)

    # Feature importance
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  FEATURE IMPORTANCE:")
    for _, r in importances.iterrows():
        bar = "█" * int(r["importance"] * 100)
        print(f"    {r['feature']:25s}  {r['importance']:.4f}  {bar}")

    # Also try GBM
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15,
                                     learning_rate=0.05, random_state=42)
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")
    print(f"\n  GBM 5-fold CV accuracy: {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

    gb.fit(X, y)
    gb_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  GBM FEATURE IMPORTANCE:")
    for _, r in gb_imp.iterrows():
        bar = "█" * int(r["importance"] * 100)
        print(f"    {r['feature']:25s}  {r['importance']:.4f}  {bar}")

    return rf, gb, df_model, feature_cols


def analyze_feature_buckets(df):
    """Build practical lookup tables showing script probabilities by feature buckets."""
    print("\n" + "=" * 80)
    print("SCRIPT PROBABILITIES BY PRE-GAME FEATURES")
    print("=" * 80)

    df = df.dropna(subset=["spread", "over_under", "fav_win_prob"]).copy()

    # -- 1. Over/Under buckets --
    df["ou_bucket"] = pd.cut(df["over_under"],
                             bins=[0, 38, 42, 46, 50, 100],
                             labels=["Low (≤38)", "Med-Low (38-42)", "Medium (42-46)",
                                     "Med-High (46-50)", "High (50+)"])

    print("\n  1. OVER/UNDER → GAME SCRIPT PROBABILITY")
    ou_ct = pd.crosstab(df["ou_bucket"], df["game_script"], normalize="index")
    # Sort columns by total points
    script_order = df.groupby("game_script")["total_pts"].mean().sort_values(ascending=False).index
    ou_ct = ou_ct.reindex(columns=[c for c in script_order if c in ou_ct.columns])

    print(f"\n    {'O/U Bucket':18s}", end="")
    for col in ou_ct.columns:
        print(f"  {col[:14]:>14s}", end="")
    print(f"  {'N':>5s}")
    print("    " + "-" * (18 + 16 * len(ou_ct.columns) + 6))
    for idx, row in ou_ct.iterrows():
        n = df[df["ou_bucket"] == idx].shape[0]
        print(f"    {str(idx):18s}", end="")
        for col in ou_ct.columns:
            val = row.get(col, 0)
            marker = " **" if val > 0.20 else ""
            print(f"  {val:13.1%}{marker[:0]}", end="")
        print(f"  {n:5d}")

    # -- 2. Spread buckets --
    df["spread_bucket"] = pd.cut(df["abs_spread"],
                                  bins=[0, 2.5, 5, 8, 15, 30],
                                  labels=["Pick'em (0-2.5)", "Close (3-5)",
                                          "Moderate (5.5-8)", "Big (8.5-15)", "Huge (15+)"])

    print("\n  2. SPREAD SIZE → GAME SCRIPT PROBABILITY")
    sp_ct = pd.crosstab(df["spread_bucket"], df["game_script"], normalize="index")
    sp_ct = sp_ct.reindex(columns=[c for c in script_order if c in sp_ct.columns])

    print(f"\n    {'Spread':18s}", end="")
    for col in sp_ct.columns:
        print(f"  {col[:14]:>14s}", end="")
    print(f"  {'N':>5s}")
    print("    " + "-" * (18 + 16 * len(sp_ct.columns) + 6))
    for idx, row in sp_ct.iterrows():
        n = df[df["spread_bucket"] == idx].shape[0]
        print(f"    {str(idx):18s}", end="")
        for col in sp_ct.columns:
            val = row.get(col, 0)
            print(f"  {val:13.1%}", end="")
        print(f"  {n:5d}")

    # -- 3. Spread x O/U combined (the money buckets) --
    print("\n  3. SPREAD × O/U COMBINED (key DFS decision buckets)")
    df["combo_bucket"] = "Other"
    df.loc[(df["abs_spread"] <= 3) & (df["over_under"] >= 48), "combo_bucket"] = "Close + High Total"
    df.loc[(df["abs_spread"] <= 3) & (df["over_under"] < 42), "combo_bucket"] = "Close + Low Total"
    df.loc[(df["abs_spread"] >= 7) & (df["over_under"] >= 48), "combo_bucket"] = "Big Fav + High Total"
    df.loc[(df["abs_spread"] >= 7) & (df["over_under"] < 42), "combo_bucket"] = "Big Fav + Low Total"
    df.loc[(df["abs_spread"] <= 3) & (df["over_under"] >= 42) & (df["over_under"] < 48), "combo_bucket"] = "Close + Med Total"
    df.loc[(df["abs_spread"] >= 7) & (df["over_under"] >= 42) & (df["over_under"] < 48), "combo_bucket"] = "Big Fav + Med Total"

    combo_ct = pd.crosstab(df["combo_bucket"], df["game_script"], normalize="index")
    combo_ct = combo_ct.reindex(columns=[c for c in script_order if c in combo_ct.columns])

    print(f"\n    {'Bucket':24s}", end="")
    for col in combo_ct.columns:
        print(f"  {col[:14]:>14s}", end="")
    print(f"  {'N':>5s}")
    print("    " + "-" * (24 + 16 * len(combo_ct.columns) + 6))
    for idx in ["Close + High Total", "Close + Med Total", "Close + Low Total",
                "Big Fav + High Total", "Big Fav + Med Total", "Big Fav + Low Total", "Other"]:
        if idx not in combo_ct.index:
            continue
        row = combo_ct.loc[idx]
        n = df[df["combo_bucket"] == idx].shape[0]
        print(f"    {idx:24s}", end="")
        for col in combo_ct.columns:
            val = row.get(col, 0)
            print(f"  {val:13.1%}", end="")
        print(f"  {n:5d}")

    # -- 4. Team archetype matchups --
    print("\n  4. OFFENSE vs DEFENSE ARCHETYPE MATCHUPS → SCRIPT PROBABILITY")

    for matchup_name, off_col, def_col in [
        ("Home OFF vs Away DEF", "home_off_arch", "away_def_arch"),
        ("Away OFF vs Home DEF", "away_off_arch", "home_def_arch"),
    ]:
        print(f"\n    {matchup_name}:")
        matchup_ct = pd.crosstab(
            df[off_col] + " vs " + df[def_col],
            df["game_script"], normalize="index"
        )
        matchup_ct = matchup_ct.reindex(columns=[c for c in script_order if c in matchup_ct.columns])
        matchup_ct["n"] = pd.crosstab(df[off_col] + " vs " + df[def_col], df["game_script"]).sum(axis=1)

        # Only show matchups with enough samples
        matchup_ct = matchup_ct[matchup_ct["n"] >= 15].sort_values("n", ascending=False)

        if len(matchup_ct) == 0:
            continue

        # Find the most skewed matchups (highest deviation from average)
        avg_dist = df["game_script"].value_counts(normalize=True)
        matchup_ct["max_dev"] = matchup_ct.drop(columns=["n"]).apply(
            lambda row: max(abs(row.get(s, 0) - avg_dist.get(s, 0)) for s in script_order if s in row.index),
            axis=1
        )
        top_matchups = matchup_ct.nlargest(12, "max_dev")

        print(f"      {'Matchup':40s}", end="")
        for col in [c for c in script_order if c in top_matchups.columns and c not in ["n", "max_dev"]]:
            print(f"  {col[:12]:>12s}", end="")
        print(f"  {'N':>4s}")
        print("      " + "-" * (40 + 14 * min(8, len(script_order)) + 5))
        for idx, row in top_matchups.iterrows():
            print(f"      {idx:40s}", end="")
            for col in [c for c in script_order if c in top_matchups.columns and c not in ["n", "max_dev"]]:
                val = row.get(col, 0)
                print(f"  {val:11.0%}", end="  " if val > 0.25 else "  ")
            print(f"  {int(row['n']):4d}")

    # -- 5. Previous game script --
    print("\n  5. PREVIOUS GAME SCRIPT → NEXT GAME SCRIPT")
    prev_avail = df.dropna(subset=["home_prev_script"])
    if len(prev_avail) > 0:
        # Does coming off a blowout predict the next script?
        prev_ct = pd.crosstab(prev_avail["home_prev_script"], prev_avail["game_script"], normalize="index")
        prev_ct = prev_ct.reindex(columns=[c for c in script_order if c in prev_ct.columns])
        prev_ct["n"] = pd.crosstab(prev_avail["home_prev_script"], prev_avail["game_script"]).sum(axis=1)
        prev_ct = prev_ct[prev_ct["n"] >= 10]

        print(f"\n    {'Home Prev Script':27s}", end="")
        for col in [c for c in script_order if c in prev_ct.columns and c != "n"]:
            print(f"  {col[:12]:>12s}", end="")
        print(f"  {'N':>4s}")
        print("    " + "-" * (27 + 14 * min(8, len(script_order)) + 5))
        for idx, row in prev_ct.iterrows():
            print(f"    {idx:27s}", end="")
            for col in [c for c in script_order if c in prev_ct.columns and c != "n"]:
                val = row.get(col, 0)
                print(f"  {val:11.0%}", end="  ")
            print(f"  {int(row['n']):4d}")

    return df


def save_results(conn, df):
    """Save prediction lookup tables."""
    conn.executescript("""
        DROP TABLE IF EXISTS script_prediction_factors;

        CREATE TABLE script_prediction_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_type TEXT NOT NULL,
            factor_value TEXT NOT NULL,
            game_script TEXT NOT NULL,
            probability REAL NOT NULL,
            sample_size INTEGER NOT NULL
        );
    """)

    df = df.dropna(subset=["spread", "over_under"]).copy()
    df["ou_bucket"] = pd.cut(df["over_under"],
                             bins=[0, 38, 42, 46, 50, 100],
                             labels=["Low (≤38)", "Med-Low (38-42)", "Medium (42-46)",
                                     "Med-High (46-50)", "High (50+)"])
    df["spread_bucket"] = pd.cut(df["abs_spread"],
                                  bins=[0, 2.5, 5, 8, 15, 30],
                                  labels=["Pickem (0-2.5)", "Close (3-5)",
                                          "Moderate (5.5-8)", "Big (8.5-15)", "Huge (15+)"])

    rows = []
    for bucket_type, col in [("over_under", "ou_bucket"), ("spread", "spread_bucket")]:
        ct = pd.crosstab(df[col], df["game_script"], normalize="index")
        counts = pd.crosstab(df[col], df["game_script"]).sum(axis=1)
        for idx, row in ct.iterrows():
            n = counts[idx]
            for script in row.index:
                rows.append((bucket_type, str(idx), script, round(row[script], 4), int(n)))

    conn.executemany("""
        INSERT INTO script_prediction_factors (factor_type, factor_value, game_script, probability, sample_size)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    print(f"\n  Saved {len(rows)} prediction factor rows")


def main():
    print("NFL Game Script Prediction Analysis")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    print("\nLoading features...")
    df = load_features(conn)

    # ML model
    rf, gb, df_model, feature_cols = run_model(df)

    # Practical lookup tables
    df_analyzed = analyze_feature_buckets(df)

    # Save
    print("\nSaving prediction factors...")
    save_results(conn, df)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
