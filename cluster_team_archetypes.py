"""
NFL Team Archetype Clustering
Creates primary archetype + style sub-category for both team offenses and
team defenses using nflverse team-level game stats.

Offense archetypes based on: pass/rush balance, tempo, efficiency, explosiveness
Defense archetypes based on: points allowed, pressure, takeaways, rush/pass defense split
"""

import sqlite3
import os
import pandas as pd
import numpy as np
import nflreadpy as nflr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")
SEASONS = [2023, 2024, 2025]

TEAM_FULL_NAMES = {
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


def load_team_game_stats():
    """Load per-game team stats from nflverse."""
    print("Loading team game stats from nflverse...")
    ts = nflr.load_team_stats(seasons=SEASONS).to_pandas()
    # Filter to regular season only
    ts = ts[ts["season_type"] == "REG"].copy()
    print(f"  {len(ts)} team-game rows ({ts['season'].nunique()} seasons)")
    return ts


def build_offense_profiles(ts):
    """Aggregate per-game offensive averages for each team-season."""
    ts = ts.copy()

    # Derived columns
    ts["total_yards"] = ts["passing_yards"] + ts["rushing_yards"]
    ts["total_plays"] = ts["attempts"] + ts["carries"]
    ts["pass_rate"] = ts["attempts"] / ts["total_plays"].replace(0, np.nan)
    ts["yds_per_play"] = ts["total_yards"] / ts["total_plays"].replace(0, np.nan)
    ts["ypa"] = ts["passing_yards"] / ts["attempts"].replace(0, np.nan)
    ts["ypc"] = ts["rushing_yards"] / ts["carries"].replace(0, np.nan)
    ts["comp_pct"] = ts["completions"] / ts["attempts"].replace(0, np.nan)
    ts["total_tds"] = ts["passing_tds"] + ts["rushing_tds"] + ts["receiving_tds"]
    ts["turnover_rate"] = (ts["passing_interceptions"] + ts["rushing_fumbles_lost"] + ts["sack_fumbles_lost"]) / ts["total_plays"].replace(0, np.nan)
    ts["air_yds_pct"] = ts["passing_air_yards"] / ts["passing_yards"].replace(0, np.nan)
    ts["yac_pct"] = ts["passing_yards_after_catch"] / ts["passing_yards"].replace(0, np.nan)
    ts["total_epa"] = ts["passing_epa"] + ts["rushing_epa"]

    agg = ts.groupby(["team", "season"]).agg(
        games=("week", "count"),
        # Volume
        plays_pg=("total_plays", "mean"),
        pass_att_pg=("attempts", "mean"),
        rush_att_pg=("carries", "mean"),
        pass_rate=("pass_rate", "mean"),
        # Passing
        pass_yds_pg=("passing_yards", "mean"),
        pass_td_pg=("passing_tds", "mean"),
        ypa=("ypa", "mean"),
        comp_pct=("comp_pct", "mean"),
        air_yds_pct=("air_yds_pct", "mean"),
        yac_pct=("yac_pct", "mean"),
        pass_epa_pg=("passing_epa", "mean"),
        sacks_taken_pg=("sacks_suffered", "mean"),
        int_pg=("passing_interceptions", "mean"),
        # Rushing
        rush_yds_pg=("rushing_yards", "mean"),
        rush_td_pg=("rushing_tds", "mean"),
        ypc=("ypc", "mean"),
        rush_epa_pg=("rushing_epa", "mean"),
        # Totals
        total_yds_pg=("total_yards", "mean"),
        total_td_pg=("total_tds", "mean"),
        total_epa_pg=("total_epa", "mean"),
        turnover_pg=("turnover_rate", "mean"),
    ).reset_index()

    return agg.fillna(0)


def build_defense_profiles(ts):
    """Build defensive profiles using opponent offensive stats + own defensive stats."""
    ts = ts.copy()

    # For defense: what the opponent did offensively against us
    # We need to join each game with the opponent's offensive line
    # Since ts has both sides, we can self-join on game_id
    offense = ts[["game_id", "team", "season", "week",
                   "passing_yards", "rushing_yards", "attempts", "carries",
                   "passing_tds", "rushing_tds", "receiving_tds",
                   "passing_interceptions", "rushing_fumbles_lost", "sack_fumbles_lost",
                   "completions", "passing_epa", "rushing_epa",
                   "passing_air_yards", "passing_yards_after_catch"]].copy()

    defense_own = ts[["game_id", "team", "season", "week",
                       "def_sacks", "def_qb_hits", "def_interceptions",
                       "def_interception_yards", "def_pass_defended",
                       "def_fumbles_forced", "def_tackles_for_loss",
                       "def_tds"]].copy()

    # Opponent's offense = our defense's allowed stats
    # Join: our defense_own with opponent's offense on same game_id, different team
    merged = pd.merge(
        defense_own,
        offense,
        on=["game_id", "season", "week"],
        suffixes=("", "_opp"),
    )
    # Keep only rows where the teams differ (our def vs their offense)
    merged = merged[merged["team"] != merged["team_opp"]].copy()

    # Derived
    merged["opp_total_yards"] = merged["passing_yards"] + merged["rushing_yards"]
    merged["opp_total_plays"] = merged["attempts"] + merged["carries"]
    merged["opp_yds_per_play"] = merged["opp_total_yards"] / merged["opp_total_plays"].replace(0, np.nan)
    merged["opp_total_tds"] = merged["passing_tds"] + merged["rushing_tds"]
    merged["opp_total_epa"] = merged["passing_epa"] + merged["rushing_epa"]
    merged["opp_pass_rate"] = merged["attempts"] / merged["opp_total_plays"].replace(0, np.nan)
    merged["pressure_rate"] = (merged["def_sacks"] + merged["def_qb_hits"]) / merged["attempts"].replace(0, np.nan)
    merged["takeaways"] = merged["def_interceptions"] + merged["def_fumbles_forced"]

    agg = merged.groupby(["team", "season"]).agg(
        games=("week", "count"),
        # Yards allowed
        pass_yds_allowed_pg=("passing_yards", "mean"),
        rush_yds_allowed_pg=("rushing_yards", "mean"),
        total_yds_allowed_pg=("opp_total_yards", "mean"),
        yds_per_play_allowed=("opp_yds_per_play", "mean"),
        # Scoring allowed
        pass_td_allowed_pg=("passing_tds", "mean"),
        rush_td_allowed_pg=("rushing_tds", "mean"),
        total_td_allowed_pg=("opp_total_tds", "mean"),
        # EPA allowed
        pass_epa_allowed_pg=("passing_epa", "mean"),
        rush_epa_allowed_pg=("rushing_epa", "mean"),
        total_epa_allowed_pg=("opp_total_epa", "mean"),
        # Pressure
        sacks_pg=("def_sacks", "mean"),
        qb_hits_pg=("def_qb_hits", "mean"),
        pressure_rate=("pressure_rate", "mean"),
        tfl_pg=("def_tackles_for_loss", "mean"),
        # Takeaways
        int_pg=("def_interceptions", "mean"),
        ff_pg=("def_fumbles_forced", "mean"),
        takeaways_pg=("takeaways", "mean"),
        pass_defended_pg=("def_pass_defended", "mean"),
        def_tds_pg=("def_tds", "mean"),
        # Opponent tendencies forced
        opp_pass_rate=("opp_pass_rate", "mean"),
    ).reset_index()

    return agg.fillna(0)


# =====================================================================
# Clustering + Style Tagging
# =====================================================================

OFF_CLUSTER_FEATURES = [
    "pass_rate", "plays_pg", "pass_yds_pg", "rush_yds_pg",
    "ypa", "ypc", "comp_pct", "total_epa_pg",
    "pass_td_pg", "rush_td_pg", "air_yds_pct",
    "sacks_taken_pg", "turnover_pg",
]

DEF_CLUSTER_FEATURES = [
    "pass_yds_allowed_pg", "rush_yds_allowed_pg", "total_epa_allowed_pg",
    "sacks_pg", "qb_hits_pg", "pressure_rate", "tfl_pg",
    "int_pg", "ff_pg", "takeaways_pg", "pass_defended_pg",
    "total_td_allowed_pg",
]

OFF_STYLES = [
    ("Air Raid",         "pass_rate",      80, "ypa",            60),
    ("Deep Shot",        "air_yds_pct",    75),
    ("West Coast",       "comp_pct",       75, "air_yds_pct",    40),
    ("Power Run",        "rush_yds_pg",    75, "ypc",            40),
    ("Zone Run",         "rush_yds_pg",    70, "ypc",            60),
    ("RPO/Balanced",     "rush_yds_pg",    55, "pass_yds_pg",    55),
    ("Uptempo",          "plays_pg",       80),
    ("Explosive",        "total_epa_pg",   80),
    ("Ball Control",     "rush_att_pg",    70, "turnover_pg",    35),
    ("Turnover-Prone",   "turnover_pg",    80),
]

DEF_STYLES = [
    ("Pass Rush Dom.",   "sacks_pg",       75, "pressure_rate",  65),
    ("Blitz Heavy",      "qb_hits_pg",     75),
    ("Takeaway Machine", "takeaways_pg",   75),
    ("Lockdown Secondary", "pass_yds_allowed_pg", 25),  # low = good, use inverted
    ("Run Stuffing",     "rush_yds_allowed_pg",   25),  # low = good
    ("Bend Don't Break", "total_yds_allowed_pg",  60, "total_td_allowed_pg", 30),
    ("Aggressive",       "tfl_pg",         75),
    ("Ball Hawk",        "int_pg",         75),
]


def cluster_and_style(profiles, features, n_clusters, primary_namer, style_defs, unit_name):
    """Cluster into primary archetypes and assign style tags."""
    df = profiles.copy()
    feat_cols = [f for f in features if f in df.columns]
    X = df[feat_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    df["cluster"] = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, df["cluster"])

    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feat_cols)
    cluster_labels = primary_namer(centers)
    cluster_labels = _dedupe(cluster_labels)
    df["archetype"] = df["cluster"].map(cluster_labels)

    print(f"  {unit_name}: k={n_clusters} (silhouette={score:.3f})")

    # Style tagging
    df = _assign_style(df, style_defs, unit_name)
    df["full_archetype"] = df["archetype"] + " / " + df["style"]

    return df


def _assign_style(df, style_defs, unit_name):
    """Assign style based on percentile thresholds."""
    df = df.copy()
    is_defense = "Defense" in unit_name

    # Compute percentiles
    for col in df.select_dtypes(include=[np.number]).columns:
        df[f"_pct_{col}"] = df[col].rank(pct=True) * 100

    styles = []
    for _, row in df.iterrows():
        matched = []
        for trait_def in style_defs:
            name = trait_def[0]
            stat = trait_def[1]
            threshold = trait_def[2]

            pct = row.get(f"_pct_{stat}", 50)

            # For defensive "allowed" stats, low is good — the threshold IS the
            # percentile cutoff (e.g., 25 = bottom 25% = top defense)
            if "_allowed_" in stat:
                # Low percentile = good defense
                if pct <= threshold:
                    if len(trait_def) == 5:
                        stat2 = trait_def[3]
                        thresh2 = trait_def[4]
                        pct2 = row.get(f"_pct_{stat2}", 50)
                        if "_allowed_" in stat2:
                            if pct2 <= thresh2:
                                matched.append((name, 100 - pct))
                        else:
                            if pct2 >= thresh2:
                                matched.append((name, 100 - pct))
                    else:
                        matched.append((name, 100 - pct))
            else:
                if len(trait_def) == 5:
                    stat2 = trait_def[3]
                    thresh2 = trait_def[4]
                    pct2 = row.get(f"_pct_{stat2}", 50)
                    if "_allowed_" in stat2:
                        if pct >= threshold and pct2 <= thresh2:
                            matched.append((name, pct))
                    else:
                        if pct >= threshold and pct2 >= thresh2:
                            matched.append((name, pct))
                else:
                    if pct >= threshold:
                        matched.append((name, pct))

        if matched:
            matched.sort(key=lambda x: -x[1])
            styles.append(matched[0][0])
        else:
            styles.append("Balanced")

    df["style"] = styles
    pct_cols = [c for c in df.columns if c.startswith("_pct_")]
    df = df.drop(columns=pct_cols)
    return df


def _dedupe(labels):
    seen = {}
    for k, v in sorted(labels.items()):
        if v in seen:
            seen[v] += 1
            labels[k] = f"{v} {seen[v]}"
        else:
            seen[v] = 1
    return labels


# =====================================================================
# Primary archetype naming
# =====================================================================

def name_offense(centers):
    labels = {}
    # Rank by total EPA to get tiers
    centers["_epa_rank"] = centers["total_epa_pg"].rank(ascending=False)
    centers["_volume_rank"] = (centers["pass_yds_pg"] + centers["rush_yds_pg"]).rank(ascending=False)

    for idx, row in centers.iterrows():
        epa_r = row["_epa_rank"]
        pass_r = row["pass_rate"]
        rush = row["rush_yds_pg"]
        ypc = row.get("ypc", 0)

        if epa_r == 1:
            labels[idx] = "Elite"
        elif epa_r == 2:
            if pass_r > centers["pass_rate"].median():
                labels[idx] = "Pass-First"
            else:
                labels[idx] = "Run-Heavy"
        elif epa_r == 3:
            if rush > centers["rush_yds_pg"].median():
                labels[idx] = "Run-Heavy"
            else:
                labels[idx] = "Pass-First"
        else:
            labels[idx] = "Bottom-Tier"

    return labels


def name_defense(centers):
    labels = {}
    # Rank by EPA allowed (lower = better defense)
    centers["_epa_rank"] = centers["total_epa_allowed_pg"].rank(ascending=True)

    for idx, row in centers.iterrows():
        epa_r = row["_epa_rank"]
        pressure = row.get("pressure_rate", 0)
        takeaways = row.get("takeaways_pg", 0)

        if epa_r == 1:
            labels[idx] = "Elite"
        elif epa_r == 2:
            if pressure > centers["pressure_rate"].median():
                labels[idx] = "Above-Avg Pressure"
            elif takeaways > centers["takeaways_pg"].median():
                labels[idx] = "Above-Avg Opportunistic"
            else:
                labels[idx] = "Above-Average"
        elif epa_r == 3:
            labels[idx] = "Middle-of-Pack"
        else:
            labels[idx] = "Bottom-Tier"

    return labels


# =====================================================================
# Database + output
# =====================================================================

def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS team_archetypes;

        CREATE TABLE team_archetypes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            team_name TEXT,
            season INTEGER NOT NULL,
            unit TEXT NOT NULL,
            archetype TEXT NOT NULL,
            style TEXT NOT NULL,
            full_archetype TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            games INTEGER NOT NULL,
            UNIQUE(team, season, unit)
        );

        CREATE INDEX idx_team_arch_team ON team_archetypes(team);
        CREATE INDEX idx_team_arch_season ON team_archetypes(season);
        CREATE INDEX idx_team_arch_unit ON team_archetypes(unit);
        CREATE INDEX idx_team_arch_full ON team_archetypes(full_archetype);
    """)


def print_results(df, unit_name):
    """Print archetype x style grid."""
    for arch in sorted(df["archetype"].unique()):
        arch_df = df[df["archetype"] == arch]
        print(f"\n    {arch} ({len(arch_df)} team-seasons):")
        style_groups = arch_df.groupby("style").agg(
            count=("team", "size"),
        ).sort_values("count", ascending=False)

        for style_name, sg in style_groups.iterrows():
            teams = arch_df[arch_df["style"] == style_name].sort_values("season")
            names = ", ".join(f"{r['team']} ({r['season']})" for _, r in teams.head(4).iterrows())
            extra = f" +{len(teams)-4} more" if len(teams) > 4 else ""
            print(f"      / {style_name:20s} ({int(sg['count']):2d})  [{names}{extra}]")


def main():
    print("NFL Team Archetype Clustering")
    print("=" * 60)

    ts = load_team_game_stats()

    # Build profiles
    print("\nBuilding offense profiles...")
    off_profiles = build_offense_profiles(ts)
    print(f"  {len(off_profiles)} team-seasons")

    print("Building defense profiles...")
    def_profiles = build_defense_profiles(ts)
    print(f"  {len(def_profiles)} team-seasons")

    # Cluster
    print("\n--- OFFENSE ---")
    off_result = cluster_and_style(
        off_profiles, OFF_CLUSTER_FEATURES, n_clusters=4,
        primary_namer=name_offense, style_defs=OFF_STYLES, unit_name="Offense",
    )
    print_results(off_result, "Offense")

    print("\n\n--- DEFENSE ---")
    def_result = cluster_and_style(
        def_profiles, DEF_CLUSTER_FEATURES, n_clusters=4,
        primary_namer=name_defense, style_defs=DEF_STYLES, unit_name="Defense",
    )
    print_results(def_result, "Defense")

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    rows = []
    for _, r in off_result.iterrows():
        rows.append((
            r["team"], TEAM_FULL_NAMES.get(r["team"], r["team"]),
            int(r["season"]), "offense", r["archetype"], r["style"],
            r["full_archetype"], int(r["cluster"]), int(r["games"]),
        ))
    for _, r in def_result.iterrows():
        rows.append((
            r["team"], TEAM_FULL_NAMES.get(r["team"], r["team"]),
            int(r["season"]), "defense", r["archetype"], r["style"],
            r["full_archetype"], int(r["cluster"]), int(r["games"]),
        ))

    conn.executemany("""
        INSERT INTO team_archetypes
        (team, team_name, season, unit, archetype, style, full_archetype, cluster_id, games)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    # Final summary
    print("\n\n" + "=" * 60)
    print(f"Saved {len(rows)} team-season archetypes.")

    print("\nOFFENSE ARCHETYPE DISTRIBUTION:")
    for row in conn.execute("""
        SELECT full_archetype, COUNT(*) FROM team_archetypes
        WHERE unit='offense' GROUP BY full_archetype ORDER BY COUNT(*) DESC
    """).fetchall():
        print(f"  {row[0]:40s}  {row[1]:2d} team-seasons")

    print("\nDEFENSE ARCHETYPE DISTRIBUTION:")
    for row in conn.execute("""
        SELECT full_archetype, COUNT(*) FROM team_archetypes
        WHERE unit='defense' GROUP BY full_archetype ORDER BY COUNT(*) DESC
    """).fetchall():
        print(f"  {row[0]:40s}  {row[1]:2d} team-seasons")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
