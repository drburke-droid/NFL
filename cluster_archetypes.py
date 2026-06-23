"""
NFL Player Archetype Clustering
Clusters players at each position (QB, RB, WR, TE) into a primary archetype
(volume/role tier) plus a style sub-category based on stat percentiles within
their position group.

Example output: "Workhorse / Pass-Catching", "Alpha WR1 / Deep Threat"
"""

import sqlite3
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "nfl_odds.db")

MIN_GAMES = 4

# ---------- Clustering features (used for primary archetype) ----------

POSITION_FEATURES = {
    "QB": [
        "pass_att_pg", "comp_pct", "pass_yds_pg", "pass_td_pg", "int_pg",
        "rush_att_pg", "rush_yds_pg", "rush_td_pg",
        "sack_pg", "passing_epa_pg", "rushing_epa_pg",
        "pass_air_yds_pg", "yac_pg",
    ],
    "RB": [
        "rush_att_pg", "rush_yds_pg", "rush_td_pg", "ypc",
        "targets_pg", "rec_pg", "rec_yds_pg", "rec_td_pg",
        "rush_share", "target_share_pg",
        "rushing_epa_pg", "receiving_epa_pg", "fumbles_pg",
    ],
    "WR": [
        "targets_pg", "rec_pg", "rec_yds_pg", "rec_td_pg",
        "ypr", "target_share_pg", "air_yds_pg", "yac_pg",
        "rush_att_pg", "rush_yds_pg",
        "receiving_epa_pg", "wopr_pg",
    ],
    "TE": [
        "targets_pg", "rec_pg", "rec_yds_pg", "rec_td_pg",
        "ypr", "target_share_pg", "air_yds_pg", "yac_pg",
        "receiving_epa_pg",
    ],
}

# ---------- Style traits per position ----------
# Each trait: (name, stat, direction)
#   direction = "high" means top percentile triggers the tag

QB_STYLES = [
    ("Scrambler",        "rush_yds_pg",     75),
    ("Deep Ball",        "pass_air_yds_pg", 75),
    ("Short/Timing",     "comp_pct",        75, "pass_air_yds_pg", 40),  # high comp%, low air yards
    ("Gunslinger",       "pass_att_pg",     75, "int_pg", 60),           # high volume, higher INT rate
    ("Efficient",        "passing_epa_pg",  80),
    ("Turnover-Prone",   "int_pg",          80),
]

RB_STYLES = [
    ("Pass-Catching",    "rec_pg",          70),
    ("Power",            "rush_att_pg",     70, "ypc", 45),    # high volume, lower ypc = between-tackles grinder
    ("Explosive",        "ypc",             75),
    ("Goal-Line",        "rush_td_pg",      75),
    ("Versatile",        "rec_pg",          60, "rush_att_pg", 60),  # above avg in both
]

WR_STYLES = [
    ("Deep Threat",      "air_yds_pg",      75),
    ("YAC Monster",      "yac_pg",          75),
    ("Possession",       "rec_pg",          75, "air_yds_pg", 40),  # high receptions, shorter routes
    ("Red Zone",         "rec_td_pg",       75),
    ("Field Stretcher",  "ypr",             75),
    ("Gadget",           "rush_att_pg",     75),
]

TE_STYLES = [
    ("Seam Threat",      "air_yds_pg",      70),
    ("YAC",              "yac_pg",          70),
    ("Red Zone",         "rec_td_pg",       70),
    ("High-Volume",      "targets_pg",      70),
    ("Possession",       "rec_pg",          70, "air_yds_pg", 40),
]

STYLE_DEFS = {"QB": QB_STYLES, "RB": RB_STYLES, "WR": WR_STYLES, "TE": TE_STYLES}


def load_player_profiles(conn):
    """Aggregate per-game averages for each player-season."""
    df = pd.read_sql_query("""
        SELECT player_id, player_display_name as name, position, team, season,
               COUNT(*) as games,
               -- Passing
               AVG(attempts) as pass_att_pg,
               CASE WHEN SUM(attempts) > 0 THEN 1.0*SUM(completions)/SUM(attempts) ELSE 0 END as comp_pct,
               AVG(passing_yards) as pass_yds_pg,
               AVG(passing_tds) as pass_td_pg,
               AVG(passing_interceptions) as int_pg,
               AVG(sacks_suffered) as sack_pg,
               AVG(passing_epa) as passing_epa_pg,
               AVG(passing_air_yards) as pass_air_yds_pg,
               AVG(passing_yards_after_catch) as yac_pg,
               -- Rushing
               AVG(carries) as rush_att_pg,
               AVG(rushing_yards) as rush_yds_pg,
               AVG(rushing_tds) as rush_td_pg,
               CASE WHEN SUM(carries) > 0 THEN 1.0*SUM(rushing_yards)/SUM(carries) ELSE 0 END as ypc,
               AVG(rushing_epa) as rushing_epa_pg,
               AVG(rushing_fumbles) as fumbles_pg,
               -- Receiving
               AVG(targets) as targets_pg,
               AVG(receptions) as rec_pg,
               AVG(receiving_yards) as rec_yds_pg,
               AVG(receiving_tds) as rec_td_pg,
               CASE WHEN SUM(receptions) > 0 THEN 1.0*SUM(receiving_yards)/SUM(receptions) ELSE 0 END as ypr,
               AVG(target_share) as target_share_pg,
               AVG(receiving_air_yards) as air_yds_pg,
               AVG(receiving_yards_after_catch) as rec_yac_pg,
               AVG(receiving_epa) as receiving_epa_pg,
               AVG(wopr) as wopr_pg,
               -- Usage share proxy
               AVG(carries) / NULLIF(AVG(carries) + AVG(targets), 0) as rush_share,
               -- Fantasy
               AVG(fantasy_points) as fpts_pg,
               AVG(fantasy_points_ppr) as fpts_ppr_pg
        FROM nflv_weekly
        WHERE position IN ('QB','RB','WR','TE') AND season_type='REG'
        GROUP BY player_id, position, season
        HAVING COUNT(*) >= ?
    """, conn, params=(MIN_GAMES,))

    # For WR/TE yac, use receiving yac
    df["yac_pg"] = df.apply(
        lambda r: r["rec_yac_pg"] if r["position"] in ("WR", "TE") else r["yac_pg"], axis=1
    )

    return df.fillna(0)


# =====================================================================
# Primary archetype (K-Means clustering on volume/role)
# =====================================================================

def name_primary(centers_df, position):
    """Name primary archetype clusters based on cluster centers."""

    if position == "QB":
        labels = {}
        for idx, row in centers_df.iterrows():
            rush = row.get("rush_yds_pg", 0)
            pass_yds = row.get("pass_yds_pg", 0)
            epa = row.get("passing_epa_pg", 0)
            att = row.get("pass_att_pg", 0)
            if rush > centers_df["rush_yds_pg"].median() * 1.3:
                labels[idx] = "Dual-Threat"
            elif epa > centers_df["passing_epa_pg"].median() and pass_yds > centers_df["pass_yds_pg"].median():
                labels[idx] = "Franchise Passer"
            elif att > centers_df["pass_att_pg"].median():
                labels[idx] = "High-Volume Passer"
            else:
                labels[idx] = "Game Manager"
        return labels

    elif position == "RB":
        labels = {}
        centers_df["volume"] = centers_df["rush_att_pg"] + centers_df["targets_pg"]
        ranked = centers_df["volume"].rank(ascending=False)
        for idx, row in centers_df.iterrows():
            rush = row.get("rush_att_pg", 0)
            targets = row.get("targets_pg", 0)
            rank = ranked[idx]
            if rank == 1:
                labels[idx] = "Workhorse"
            elif rank == 2:
                labels[idx] = "Starter"
            elif rank == 3:
                labels[idx] = "Rotational"
            else:
                labels[idx] = "Depth"
        return labels

    elif position == "WR":
        labels = {}
        # Rank clusters by target volume to assign tiers
        ranked = centers_df["targets_pg"].rank(ascending=False)
        for idx, row in centers_df.iterrows():
            rank = ranked[idx]
            if rank == 1:
                labels[idx] = "Alpha WR1"
            elif rank == 2:
                labels[idx] = "WR2"
            elif rank == 3:
                labels[idx] = "WR3/Flex"
            else:
                labels[idx] = "Depth"
        return labels

    elif position == "TE":
        labels = {}
        ranked = centers_df["targets_pg"].rank(ascending=False)
        for idx, row in centers_df.iterrows():
            rank = ranked[idx]
            if rank == 1:
                labels[idx] = "Elite Receiving"
            elif rank == 2:
                labels[idx] = "Secondary Receiver"
            else:
                labels[idx] = "Blocking/Depth"
        return labels

    return {i: f"Cluster {i}" for i in centers_df.index}


def deduplicate_labels(labels):
    """If two clusters got the same name, append a number."""
    seen = {}
    for k, v in sorted(labels.items()):
        if v in seen:
            seen[v] += 1
            labels[k] = f"{v} {seen[v]}"
        else:
            seen[v] = 1
    return labels


def cluster_primary(profiles, position):
    """Cluster into primary archetypes."""
    pos_df = profiles[profiles["position"] == position].copy()
    features = [f for f in POSITION_FEATURES[position] if f in pos_df.columns]
    X = pos_df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use a fixed k per position for meaningful role tiers
    # Silhouette prefers k=2, but that's just "starter vs bench" — not useful
    n_clusters = {"QB": 3, "RB": 4, "WR": 4, "TE": 3}[position]
    best_k = n_clusters
    km_test = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    lab = km_test.fit_predict(X_scaled)
    best_score = silhouette_score(X_scaled, lab)

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    pos_df["cluster"] = km.fit_predict(X_scaled)

    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features)
    cluster_labels = deduplicate_labels(name_primary(centers, position))
    pos_df["archetype"] = pos_df["cluster"].map(cluster_labels)

    print(f"  {position}: k={best_k} (silhouette={best_score:.3f})")
    return pos_df


# =====================================================================
# Style sub-category (percentile-based trait tagging)
# =====================================================================

def assign_style(pos_df, position):
    """Assign a style tag to each player based on percentile thresholds."""
    style_defs = STYLE_DEFS[position]
    pos_df = pos_df.copy()

    # Precompute percentiles for all stats in this position group
    pctiles = {}
    for col in pos_df.select_dtypes(include=[np.number]).columns:
        pctiles[col] = pos_df[col].rank(pct=True) * 100
        pos_df[f"_pct_{col}"] = pctiles[col]

    styles = []
    for idx, row in pos_df.iterrows():
        matched = []
        for trait_def in style_defs:
            name = trait_def[0]
            primary_stat = trait_def[1]
            primary_threshold = trait_def[2]

            primary_pct = row.get(f"_pct_{primary_stat}", 0)

            if len(trait_def) == 5:
                # Compound trait: primary stat above threshold AND secondary stat relative
                secondary_stat = trait_def[3]
                secondary_threshold = trait_def[4]
                secondary_pct = row.get(f"_pct_{secondary_stat}", 50)

                if primary_pct >= primary_threshold and secondary_pct <= secondary_threshold:
                    matched.append((name, primary_pct))
            else:
                # Simple trait: stat above threshold
                if primary_pct >= primary_threshold:
                    matched.append((name, primary_pct))

        if matched:
            # Pick the strongest matching trait (highest percentile)
            matched.sort(key=lambda x: -x[1])
            styles.append(matched[0][0])
        else:
            styles.append("Balanced")

    pos_df["style"] = styles

    # Clean up percentile columns
    pct_cols = [c for c in pos_df.columns if c.startswith("_pct_")]
    pos_df = pos_df.drop(columns=pct_cols)

    return pos_df


# =====================================================================
# Database + output
# =====================================================================

def create_archetype_table(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS player_archetypes;

        CREATE TABLE player_archetypes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT NOT NULL,
            team TEXT NOT NULL,
            season INTEGER NOT NULL,
            archetype TEXT NOT NULL,
            style TEXT NOT NULL,
            full_archetype TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            games INTEGER NOT NULL,
            fpts_ppr_pg REAL,
            UNIQUE(player_id, season)
        );

        CREATE INDEX idx_arch_position ON player_archetypes(position);
        CREATE INDEX idx_arch_archetype ON player_archetypes(archetype);
        CREATE INDEX idx_arch_style ON player_archetypes(style);
        CREATE INDEX idx_arch_full ON player_archetypes(full_archetype);
        CREATE INDEX idx_arch_player ON player_archetypes(player_id);
        CREATE INDEX idx_arch_season ON player_archetypes(season);
    """)


def main():
    print("NFL Player Archetype Clustering")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    print(f"\nLoading player profiles (min {MIN_GAMES} games)...")
    profiles = load_player_profiles(conn)
    print(f"  {len(profiles)} player-seasons across {profiles['position'].nunique()} positions\n")

    create_archetype_table(conn)

    all_results = []
    for position in ["QB", "RB", "WR", "TE"]:
        print(f"--- {position} ---")
        pos_df = cluster_primary(profiles, position)
        pos_df = assign_style(pos_df, position)
        pos_df["full_archetype"] = pos_df["archetype"] + " / " + pos_df["style"]
        all_results.append(pos_df)

        # Print archetype x style breakdown
        for arch in sorted(pos_df["archetype"].unique()):
            arch_df = pos_df[pos_df["archetype"] == arch]
            print(f"\n    {arch} ({len(arch_df)} player-seasons):")
            style_groups = arch_df.groupby("style").agg(
                count=("player_id", "size"),
                avg_ppr=("fpts_ppr_pg", "mean"),
            ).sort_values("avg_ppr", ascending=False)

            for style_name, sg in style_groups.iterrows():
                top = arch_df[arch_df["style"] == style_name].nlargest(3, "fpts_ppr_pg")
                names = ", ".join(f"{r['name']}" for _, r in top.iterrows())
                print(f"      / {style_name:18s} ({int(sg['count']):3d})  avg {sg['avg_ppr']:5.1f} PPR  [{names}]")

        print()

    # Save to DB
    combined = pd.concat(all_results, ignore_index=True)
    rows = []
    for _, r in combined.iterrows():
        rows.append((
            r["player_id"], r["name"], r["position"], r["team"],
            int(r["season"]), r["archetype"], r["style"], r["full_archetype"],
            int(r["cluster"]), int(r["games"]), round(r["fpts_ppr_pg"], 2),
        ))

    conn.executemany("""
        INSERT INTO player_archetypes
        (player_id, player_name, position, team, season, archetype, style, full_archetype,
         cluster_id, games, fpts_ppr_pg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    # Final summary
    print("=" * 60)
    print(f"Saved {len(rows)} player-season archetypes.\n")

    print("FULL ARCHETYPE DISTRIBUTION (top 15 by avg PPR):")
    dist = conn.execute("""
        SELECT full_archetype, position, COUNT(*), ROUND(AVG(fpts_ppr_pg), 1)
        FROM player_archetypes
        GROUP BY full_archetype
        ORDER BY AVG(fpts_ppr_pg) DESC
        LIMIT 15
    """).fetchall()
    for d in dist:
        print(f"  {d[1]:2s}  {d[0]:40s}  {d[2]:3d} players  avg {d[3]:5.1f} PPR/gm")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
