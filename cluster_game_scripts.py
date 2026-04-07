"""
NFL Game Script Archetype Clustering
Classifies games into script archetypes based on quarter-by-quarter scoring
trajectories, then validates each archetype against actual play-calling patterns
from play-by-play data (pass rate, rush rate, air yards, tempo, etc.).

Game scripts drive DFS outcomes: blowouts kill passing for the leader,
shootouts inflate WR/QB value, negative game scripts boost pass-catching RBs.
"""

import sqlite3
import os
import sys
import pandas as pd
import numpy as np
import nflreadpy as nflr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")
SEASONS = [2023, 2024, 2025]


# =====================================================================
# Step 1: Build game-level scoring trajectory features
# =====================================================================

def build_game_features(conn):
    """Build scoring trajectory features for each game from quarter_scores."""
    print("Building game scoring features...")

    qs = pd.read_sql_query("""
        SELECT game_id, season, week, home_team, away_team, quarter,
               home_qtr_pts, away_qtr_pts, home_cum_pts, away_cum_pts,
               event_id
        FROM quarter_scores
        WHERE quarter <= 4
    """, conn)

    # Pivot to one row per game
    games = []
    for game_id, gdf in qs.groupby("game_id"):
        gdf = gdf.sort_values("quarter")
        row = {
            "game_id": game_id,
            "season": gdf.iloc[0]["season"],
            "week": gdf.iloc[0]["week"],
            "home_team": gdf.iloc[0]["home_team"],
            "away_team": gdf.iloc[0]["away_team"],
            "event_id": gdf.iloc[0]["event_id"],
        }

        for _, q in gdf.iterrows():
            qn = int(q["quarter"])
            row[f"home_q{qn}"] = q["home_qtr_pts"]
            row[f"away_q{qn}"] = q["away_qtr_pts"]
            row[f"combined_q{qn}"] = q["home_qtr_pts"] + q["away_qtr_pts"]
            row[f"diff_q{qn}"] = q["home_cum_pts"] - q["away_cum_pts"]  # home perspective

        # Skip games missing quarters (rare edge cases)
        if not all(f"combined_q{q}" in row for q in [1, 2, 3, 4]):
            continue

        # Derived features
        row["total_pts"] = sum(row.get(f"combined_q{q}", 0) for q in [1, 2, 3, 4])
        row["home_total"] = sum(row.get(f"home_q{q}", 0) for q in [1, 2, 3, 4])
        row["away_total"] = sum(row.get(f"away_q{q}", 0) for q in [1, 2, 3, 4])
        row["final_margin"] = abs(row["home_total"] - row["away_total"])

        # Halftime state
        row["half_diff"] = row.get("diff_q2", 0)  # home lead at half
        row["half_abs_diff"] = abs(row["half_diff"])
        row["half_total"] = row.get("combined_q1", 0) + row.get("combined_q2", 0)
        row["second_half_total"] = row.get("combined_q3", 0) + row.get("combined_q4", 0)

        # Scoring trajectory shape
        row["q1_share"] = row.get("combined_q1", 0) / max(row["total_pts"], 1)
        row["q2_share"] = row.get("combined_q2", 0) / max(row["total_pts"], 1)
        row["q3_share"] = row.get("combined_q3", 0) / max(row["total_pts"], 1)
        row["q4_share"] = row.get("combined_q4", 0) / max(row["total_pts"], 1)
        row["first_half_share"] = row["half_total"] / max(row["total_pts"], 1)

        # Momentum: did the lead change or grow?
        row["lead_change"] = 1 if (row.get("diff_q2", 0) * row.get("diff_q4", 0)) < 0 else 0
        row["lead_growth"] = abs(row.get("diff_q4", 0)) - abs(row.get("diff_q2", 0))

        # Winner's 2nd half scoring vs 1st half
        if row["home_total"] >= row["away_total"]:
            row["winner_1h"] = row.get("home_q1", 0) + row.get("home_q2", 0)
            row["winner_2h"] = row.get("home_q3", 0) + row.get("home_q4", 0)
            row["loser_1h"] = row.get("away_q1", 0) + row.get("away_q2", 0)
            row["loser_2h"] = row.get("away_q3", 0) + row.get("away_q4", 0)
        else:
            row["winner_1h"] = row.get("away_q1", 0) + row.get("away_q2", 0)
            row["winner_2h"] = row.get("away_q3", 0) + row.get("away_q4", 0)
            row["loser_1h"] = row.get("home_q1", 0) + row.get("home_q2", 0)
            row["loser_2h"] = row.get("home_q3", 0) + row.get("home_q4", 0)

        games.append(row)

    df = pd.DataFrame(games)
    print(f"  {len(df)} games with complete quarter data")
    return df


# =====================================================================
# Step 2: Cluster into game script archetypes
# =====================================================================

CLUSTER_FEATURES = [
    "total_pts", "final_margin", "half_abs_diff", "lead_growth",
    "combined_q1", "combined_q2", "combined_q3", "combined_q4",
    "first_half_share", "lead_change",
    "winner_1h", "winner_2h", "loser_1h", "loser_2h",
]


def name_scripts(centers, labels_map):
    """Name game script clusters based on their center profiles."""
    labels = {}
    for idx, row in centers.iterrows():
        total = row.get("total_pts", 0)
        margin = row.get("final_margin", 0)
        half_diff = row.get("half_abs_diff", 0)
        lead_change = row.get("lead_change", 0)
        lead_growth = row.get("lead_growth", 0)
        winner_2h = row.get("winner_2h", 0)
        loser_2h = row.get("loser_2h", 0)
        loser_1h = row.get("loser_1h", 0)

        if total > centers["total_pts"].quantile(0.75) and margin < centers["final_margin"].median():
            labels[idx] = "Shootout"
        elif margin > centers["final_margin"].quantile(0.75) and half_diff > centers["half_abs_diff"].median():
            labels[idx] = "Wire-to-Wire Blowout"
        elif margin > centers["final_margin"].quantile(0.7) and half_diff < centers["half_abs_diff"].median():
            labels[idx] = "2nd Half Blowout"
        elif total > centers["total_pts"].quantile(0.6) and margin > centers["final_margin"].median() and lead_growth > 0:
            labels[idx] = "High-Scoring Pulled Away"
        elif lead_change > centers["lead_change"].median() and total < centers["total_pts"].median():
            labels[idx] = "Low-Scoring Seesaw"
        elif lead_change > centers["lead_change"].median() and total > centers["total_pts"].median():
            labels[idx] = "Back-and-Forth"
        elif loser_2h > loser_1h * 1.3 and margin < centers["final_margin"].median():
            labels[idx] = "Comeback/Competitive"
        elif total < centers["total_pts"].quantile(0.25):
            labels[idx] = "Defensive Slugfest"
        elif margin < centers["final_margin"].quantile(0.25):
            labels[idx] = "Tight Throughout"
        else:
            labels[idx] = "Steady Build"

    return labels


def dedupe(labels):
    seen = {}
    for k, v in sorted(labels.items()):
        if v in seen:
            seen[v] += 1
            labels[k] = f"{v} {seen[v]}"
        else:
            seen[v] = 1
    return labels


def cluster_games(game_df):
    """Cluster games into script archetypes."""
    print("\nClustering game scripts...")
    feats = [f for f in CLUSTER_FEATURES if f in game_df.columns]
    X = game_df[feats].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test k=5-8 for game scripts
    best_k, best_score = 6, -1
    for k in range(5, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        lab = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, lab)
        if s > best_score:
            best_score, best_k = s, k

    print(f"  Using k={best_k} (silhouette={best_score:.3f})")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    game_df["cluster"] = km.fit_predict(X_scaled)

    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feats)
    script_labels = dedupe(name_scripts(centers, {}))
    game_df["game_script"] = game_df["cluster"].map(script_labels)

    # Print cluster profiles
    for cid in sorted(game_df["cluster"].unique()):
        label = script_labels[cid]
        cdf = game_df[game_df["cluster"] == cid]
        print(f"\n  {label} ({len(cdf)} games):")
        print(f"    Avg total pts: {cdf['total_pts'].mean():.1f}")
        print(f"    Avg final margin: {cdf['final_margin'].mean():.1f}")
        print(f"    Avg halftime deficit: {cdf['half_abs_diff'].mean():.1f}")
        print(f"    Lead changes: {cdf['lead_change'].mean():.0%}")
        print(f"    Winner 1H/2H: {cdf['winner_1h'].mean():.1f}/{cdf['winner_2h'].mean():.1f}")
        print(f"    Loser 1H/2H: {cdf['loser_1h'].mean():.1f}/{cdf['loser_2h'].mean():.1f}")

        samples = cdf.sample(min(3, len(cdf)), random_state=42)
        for _, s in samples.iterrows():
            print(f"    e.g. {s['away_team']}@{s['home_team']} Wk{s['week']} "
                  f"({s['away_total']:.0f}-{s['home_total']:.0f})")

    return game_df


# =====================================================================
# Step 3: Validate with play-by-play tendencies
# =====================================================================

def load_pbp_tendencies():
    """Load PBP and compute per-team-per-quarter tendencies for each game."""
    print("\nLoading play-by-play for tendency validation...")
    all_pbp = []
    for season in SEASONS:
        print(f"  {season}...")
        pbp = nflr.load_pbp(seasons=[season]).to_pandas()
        # Filter to actual plays
        pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
        pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
        pbp["is_deep"] = (pbp["pass_length"] == "deep").astype(int)
        pbp["qtr"] = pbp["qtr"].astype(int)
        all_pbp.append(pbp[["game_id", "posteam", "qtr", "is_pass", "is_deep",
                             "air_yards", "shotgun", "no_huddle", "epa",
                             "score_differential", "wp"]])

    pbp = pd.concat(all_pbp, ignore_index=True)
    print(f"  {len(pbp)} plays loaded")

    # Aggregate per team per game per half
    pbp["half"] = pbp["qtr"].apply(lambda q: 1 if q <= 2 else 2)

    game_tendencies = pbp.groupby(["game_id", "posteam", "half"]).agg(
        plays=("is_pass", "count"),
        pass_rate=("is_pass", "mean"),
        deep_rate=("is_deep", "mean"),
        avg_air_yards=("air_yards", "mean"),
        shotgun_rate=("shotgun", "mean"),
        no_huddle_rate=("no_huddle", "mean"),
        avg_epa=("epa", "mean"),
        avg_score_diff=("score_differential", "mean"),
        avg_wp=("wp", "mean"),
    ).reset_index()

    return game_tendencies


def validate_scripts(game_df, tendencies):
    """Show how play-calling changes by game script archetype."""
    print("\n" + "=" * 80)
    print("GAME SCRIPT VALIDATION — PLAY-CALLING TENDENCIES")
    print("=" * 80)

    # Join tendencies with game scripts
    # For each game, determine which team was winning/losing
    merged_rows = []
    for _, game in game_df.iterrows():
        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]

        if game["home_total"] >= game["away_total"]:
            winner, loser = home, away
        else:
            winner, loser = away, home

        for team, role in [(winner, "winner"), (loser, "loser")]:
            for half in [1, 2]:
                t = tendencies[
                    (tendencies["game_id"] == gid)
                    & (tendencies["posteam"] == team)
                    & (tendencies["half"] == half)
                ]
                if len(t) == 1:
                    row = t.iloc[0].to_dict()
                    row["game_script"] = game["game_script"]
                    row["role"] = role
                    row["half_label"] = f"{'1st' if half == 1 else '2nd'} Half"
                    merged_rows.append(row)

    mdf = pd.DataFrame(merged_rows)

    # Print tendencies by script x role x half
    print(f"\n  {'Script':25s} {'Role':7s} {'Half':8s} {'Plays':>5s} {'Pass%':>6s} {'Deep%':>6s} "
          f"{'AvgAir':>6s} {'Shotgun':>7s} {'Hurry':>6s} {'EPA':>6s}")
    print("  " + "-" * 105)

    for script in sorted(mdf["game_script"].unique()):
        for role in ["winner", "loser"]:
            for half in ["1st Half", "2nd Half"]:
                subset = mdf[
                    (mdf["game_script"] == script)
                    & (mdf["role"] == role)
                    & (mdf["half_label"] == half)
                ]
                if len(subset) < 5:
                    continue

                print(f"  {script:25s} {role:7s} {half:8s} "
                      f"{subset['plays'].mean():5.0f} "
                      f"{subset['pass_rate'].mean():5.1%} "
                      f"{subset['deep_rate'].mean():5.1%} "
                      f"{subset['avg_air_yards'].mean():6.1f} "
                      f"{subset['shotgun_rate'].mean():6.1%} "
                      f"{subset['no_huddle_rate'].mean():5.1%} "
                      f"{subset['avg_epa'].mean():+5.2f}")
        print()

    # Compute deltas: 2nd half vs 1st half shifts by script
    print("\n  2ND HALF SHIFTS (change from 1st half)")
    print(f"  {'Script':25s} {'Role':7s} {'ΔPass%':>7s} {'ΔDeep%':>7s} {'ΔAirYds':>7s} {'ΔHurry':>7s} {'ΔEPA':>7s}")
    print("  " + "-" * 80)

    shift_rows = []
    for script in sorted(mdf["game_script"].unique()):
        for role in ["winner", "loser"]:
            h1 = mdf[(mdf["game_script"] == script) & (mdf["role"] == role) & (mdf["half_label"] == "1st Half")]
            h2 = mdf[(mdf["game_script"] == script) & (mdf["role"] == role) & (mdf["half_label"] == "2nd Half")]
            if len(h1) < 5 or len(h2) < 5:
                continue

            d_pass = h2["pass_rate"].mean() - h1["pass_rate"].mean()
            d_deep = h2["deep_rate"].mean() - h1["deep_rate"].mean()
            d_air = h2["avg_air_yards"].mean() - h1["avg_air_yards"].mean()
            d_hurry = h2["no_huddle_rate"].mean() - h1["no_huddle_rate"].mean()
            d_epa = h2["avg_epa"].mean() - h1["avg_epa"].mean()

            print(f"  {script:25s} {role:7s} {d_pass:+6.1%} {d_deep:+6.1%} "
                  f"{d_air:+7.1f} {d_hurry:+6.1%} {d_epa:+6.2f}")

            shift_rows.append({
                "game_script": script, "role": role,
                "delta_pass_rate": round(d_pass, 4),
                "delta_deep_rate": round(d_deep, 4),
                "delta_air_yards": round(d_air, 2),
                "delta_no_huddle": round(d_hurry, 4),
                "delta_epa": round(d_epa, 3),
            })
        print()

    return mdf, pd.DataFrame(shift_rows)


# =====================================================================
# Save + DFS implications
# =====================================================================

def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS game_scripts;

        CREATE TABLE game_scripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            game_id TEXT NOT NULL UNIQUE,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            game_script TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            total_pts INTEGER,
            final_margin INTEGER,
            home_total INTEGER,
            away_total INTEGER,
            half_abs_diff INTEGER,
            lead_change INTEGER,
            FOREIGN KEY (event_id) REFERENCES games(event_id)
        );

        CREATE INDEX idx_gs_script ON game_scripts(game_script);
        CREATE INDEX idx_gs_event ON game_scripts(event_id);
        CREATE INDEX idx_gs_season ON game_scripts(season, week);
        CREATE INDEX idx_gs_teams ON game_scripts(home_team, away_team);

        DROP TABLE IF EXISTS game_script_tendencies;

        CREATE TABLE game_script_tendencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_script TEXT NOT NULL,
            role TEXT NOT NULL,
            delta_pass_rate REAL,
            delta_deep_rate REAL,
            delta_air_yards REAL,
            delta_no_huddle REAL,
            delta_epa REAL
        );
    """)


def print_dfs_implications(conn):
    """Print actionable DFS implications per game script."""
    print("\n" + "=" * 80)
    print("DFS IMPLICATIONS BY GAME SCRIPT")
    print("=" * 80)

    scripts = conn.execute("""
        SELECT game_script, COUNT(*) as n,
               ROUND(AVG(total_pts), 1) as avg_pts,
               ROUND(AVG(final_margin), 1) as avg_margin
        FROM game_scripts
        GROUP BY game_script
        ORDER BY AVG(total_pts) DESC
    """).fetchall()

    tendencies = pd.read_sql_query("SELECT * FROM game_script_tendencies", conn)

    for script, n, avg_pts, avg_margin in scripts:
        print(f"\n  {script} ({n} games, avg {avg_pts} pts, margin {avg_margin})")

        t = tendencies[tendencies["game_script"] == script]
        winner_t = t[t["role"] == "winner"]
        loser_t = t[t["role"] == "loser"]

        implications = []
        if len(winner_t) > 0:
            wt = winner_t.iloc[0]
            if wt["delta_pass_rate"] < -0.03:
                implications.append(f"Winner goes run-heavy 2H ({wt['delta_pass_rate']:+.1%} pass rate) → fade winner WRs late")
            if wt["delta_deep_rate"] < -0.02:
                implications.append(f"Winner stops throwing deep 2H ({wt['delta_deep_rate']:+.1%}) → fade deep threats on fav")
            if wt["delta_pass_rate"] > 0.03:
                implications.append(f"Winner keeps passing 2H ({wt['delta_pass_rate']:+.1%}) → winner stack stays hot")

        if len(loser_t) > 0:
            lt = loser_t.iloc[0]
            if lt["delta_pass_rate"] > 0.03:
                implications.append(f"Loser forced to pass 2H ({lt['delta_pass_rate']:+.1%}) → boost loser pass-catchers")
            if lt["delta_no_huddle"] > 0.02:
                implications.append(f"Loser goes hurry-up 2H ({lt['delta_no_huddle']:+.1%}) → volume spike for loser WR/TE")
            if lt["delta_deep_rate"] > 0.01:
                implications.append(f"Loser pushes deep 2H ({lt['delta_deep_rate']:+.1%}) → boom/bust for loser deep WRs")
            if lt["delta_epa"] < -0.1:
                implications.append(f"Loser EPA craters 2H ({lt['delta_epa']:+.2f}) → desperation = inefficient volume")

        if implications:
            for imp in implications:
                print(f"    → {imp}")
        else:
            print(f"    → Neutral script, no strong 2H tendencies")


def main():
    print("NFL Game Script Archetype Clustering")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    # Step 1: Build features
    game_df = build_game_features(conn)

    # Step 2: Cluster
    game_df = cluster_games(game_df)

    # Step 3: Validate with PBP
    tendencies = load_pbp_tendencies()
    merged, shifts = validate_scripts(game_df, tendencies)

    # Save
    print("\nSaving to database...")
    create_tables(conn)

    rows = []
    for _, r in game_df.iterrows():
        rows.append((
            r.get("event_id"), r["game_id"], int(r["season"]), int(r["week"]),
            r["home_team"], r["away_team"], r["game_script"], int(r["cluster"]),
            int(r["total_pts"]), int(r["final_margin"]),
            int(r["home_total"]), int(r["away_total"]),
            int(r["half_abs_diff"]), int(r["lead_change"]),
        ))

    conn.executemany("""
        INSERT INTO game_scripts
        (event_id, game_id, season, week, home_team, away_team, game_script, cluster_id,
         total_pts, final_margin, home_total, away_total, half_abs_diff, lead_change)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    if len(shifts) > 0:
        shifts.to_sql("game_script_tendencies", conn, if_exists="append", index=False)

    conn.commit()
    print(f"  Saved {len(rows)} game scripts + {len(shifts)} tendency rows")

    # DFS implications
    print_dfs_implications(conn)

    # Distribution
    print("\n\nGAME SCRIPT DISTRIBUTION:")
    for row in conn.execute("""
        SELECT game_script, COUNT(*), ROUND(AVG(total_pts),1), ROUND(AVG(final_margin),1)
        FROM game_scripts GROUP BY game_script ORDER BY COUNT(*) DESC
    """).fetchall():
        print(f"  {row[0]:25s}  {row[1]:3d} games  avg {row[2]:4.1f} pts  margin {row[3]:4.1f}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
