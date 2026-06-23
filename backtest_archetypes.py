"""
NFL Archetype Backtesting
Tests how player archetypes and archetype combos perform against different
defensive archetypes. Identifies exploitable edges for DFS.

Analysis layers:
  1. Individual: player archetype vs defense archetype
  2. Combo: pairs of player archetypes on the same team vs defense
  3. Statistical significance via bootstrap confidence intervals
"""

import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "nfl_odds.db")

# Minimum sample size for a matchup to be reportable
MIN_GAMES = 10
# Bootstrap params
N_BOOTSTRAP = 5000
CI_LEVEL = 0.90


def load_matchup_data(conn):
    """Load player game stats joined with archetypes."""
    df = pd.read_sql_query("""
        SELECT
            ps.player_id, ps.player_display_name as player_name,
            ps.position, ps.team, ps.season, ps.week, ps.opponent,
            ps.fantasy_points_ppr as ppr,
            ps.passing_yards, ps.rushing_yards, ps.receiving_yards,
            ps.passing_tds, ps.rushing_tds, ps.receiving_tds,
            pa.archetype as player_arch,
            pa.style as player_style,
            pa.full_archetype as player_full,
            ta_def.archetype as def_arch,
            ta_def.style as def_style,
            ta_def.full_archetype as def_full,
            ta_off.full_archetype as own_off_arch
        FROM player_stats ps
        JOIN player_archetypes pa
            ON pa.player_id = ps.player_id AND pa.season = ps.season
        JOIN team_archetypes ta_def
            ON ta_def.team = ps.opponent AND ta_def.season = ps.season
            AND ta_def.unit = 'defense'
        JOIN team_archetypes ta_off
            ON ta_off.team = ps.team AND ta_off.season = ps.season
            AND ta_off.unit = 'offense'
        WHERE ps.position IN ('QB','RB','WR','TE')
    """, conn)

    return df


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap confidence interval for the mean."""
    values = np.array(values, dtype=float)
    if len(values) < 3:
        return values.mean(), values.mean(), values.mean()
    means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return values.mean(), lo, hi


def compute_baselines(df):
    """Compute baseline PPR per position x player_archetype."""
    baselines = df.groupby(["position", "player_arch"]).agg(
        baseline_ppr=("ppr", "mean"),
        baseline_n=("ppr", "count"),
    ).reset_index()
    return baselines


# =====================================================================
# Layer 1: Individual player archetype vs defense archetype
# =====================================================================

def analyze_individual(df, baselines):
    """How does each player archetype perform vs each defense archetype?"""
    print("\n" + "=" * 80)
    print("LAYER 1: INDIVIDUAL ARCHETYPE vs DEFENSE")
    print("=" * 80)

    results = []

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df["position"] == pos]

        for p_arch in sorted(pos_df["player_arch"].unique()):
            arch_df = pos_df[pos_df["player_arch"] == p_arch]
            baseline = baselines[
                (baselines["position"] == pos) & (baselines["player_arch"] == p_arch)
            ]["baseline_ppr"].values[0]

            for d_arch in sorted(arch_df["def_arch"].unique()):
                matchup = arch_df[arch_df["def_arch"] == d_arch]
                n = len(matchup)
                if n < MIN_GAMES:
                    continue

                mean, lo, hi = bootstrap_ci(matchup["ppr"].values)
                delta = mean - baseline
                delta_pct = (delta / baseline * 100) if baseline > 0 else 0

                results.append({
                    "position": pos,
                    "player_arch": p_arch,
                    "def_arch": d_arch,
                    "n": n,
                    "avg_ppr": round(mean, 2),
                    "baseline": round(baseline, 2),
                    "delta": round(delta, 2),
                    "delta_pct": round(delta_pct, 1),
                    "ci_lo": round(lo, 2),
                    "ci_hi": round(hi, 2),
                    "significant": lo > baseline or hi < baseline,
                })

    results_df = pd.DataFrame(results)
    sig = results_df[results_df["significant"]].copy()

    # Print top edges (positive)
    print("\n  TOP POSITIVE EDGES (statistically significant):")
    print(f"  {'Pos':3s}  {'Player Archetype':22s}  {'vs Defense':22s}  {'N':>4s}  {'Avg PPR':>7s}  {'Base':>6s}  {'Delta':>6s}  {'Δ%':>6s}  {'90% CI':>14s}")
    print("  " + "-" * 115)
    top_pos = sig[sig["delta"] > 0].sort_values("delta", ascending=False).head(25)
    for _, r in top_pos.iterrows():
        print(f"  {r['position']:3s}  {r['player_arch']:22s}  {r['def_arch']:22s}  {r['n']:4d}  {r['avg_ppr']:7.2f}  {r['baseline']:6.2f}  {r['delta']:+6.2f}  {r['delta_pct']:+5.1f}%  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    # Print top vulnerabilities (negative)
    print("\n  TOP NEGATIVE EDGES (statistically significant):")
    print(f"  {'Pos':3s}  {'Player Archetype':22s}  {'vs Defense':22s}  {'N':>4s}  {'Avg PPR':>7s}  {'Base':>6s}  {'Delta':>6s}  {'Δ%':>6s}  {'90% CI':>14s}")
    print("  " + "-" * 115)
    top_neg = sig[sig["delta"] < 0].sort_values("delta").head(25)
    for _, r in top_neg.iterrows():
        print(f"  {r['position']:3s}  {r['player_arch']:22s}  {r['def_arch']:22s}  {r['n']:4d}  {r['avg_ppr']:7.2f}  {r['baseline']:6.2f}  {r['delta']:+6.2f}  {r['delta_pct']:+5.1f}%  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    return results_df


# =====================================================================
# Layer 2: Player archetype with STYLE vs defense STYLE (granular)
# =====================================================================

def analyze_granular(df, baselines):
    """Full archetype (primary + style) vs defense full archetype."""
    print("\n" + "=" * 80)
    print("LAYER 2: PLAYER STYLE vs DEFENSE STYLE (granular)")
    print("=" * 80)

    results = []

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df["position"] == pos]
        # Baseline per full archetype
        full_baselines = pos_df.groupby("player_full")["ppr"].mean().to_dict()

        for p_full in sorted(pos_df["player_full"].unique()):
            p_df = pos_df[pos_df["player_full"] == p_full]
            baseline = full_baselines[p_full]

            for d_full in sorted(p_df["def_full"].unique()):
                matchup = p_df[p_df["def_full"] == d_full]
                n = len(matchup)
                if n < MIN_GAMES:
                    continue

                mean, lo, hi = bootstrap_ci(matchup["ppr"].values)
                delta = mean - baseline

                results.append({
                    "position": pos,
                    "player_full": p_full,
                    "def_full": d_full,
                    "n": n,
                    "avg_ppr": round(mean, 2),
                    "baseline": round(baseline, 2),
                    "delta": round(delta, 2),
                    "ci_lo": round(lo, 2),
                    "ci_hi": round(hi, 2),
                    "significant": lo > baseline or hi < baseline,
                })

    results_df = pd.DataFrame(results)
    sig = results_df[results_df["significant"]].copy()

    print(f"\n  {len(sig)} significant matchups out of {len(results_df)} tested")

    print("\n  TOP GRANULAR EDGES (positive, significant):")
    print(f"  {'Pos':3s}  {'Player Archetype':35s}  {'vs Defense':35s}  {'N':>4s}  {'Avg':>6s}  {'Base':>6s}  {'Δ':>6s}  {'CI':>14s}")
    print("  " + "-" * 140)
    top = sig[sig["delta"] > 0].sort_values("delta", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  {r['position']:3s}  {r['player_full']:35s}  {r['def_full']:35s}  {r['n']:4d}  {r['avg_ppr']:6.2f}  {r['baseline']:6.2f}  {r['delta']:+6.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    print("\n  TOP GRANULAR EDGES (negative, significant):")
    print(f"  {'Pos':3s}  {'Player Archetype':35s}  {'vs Defense':35s}  {'N':>4s}  {'Avg':>6s}  {'Base':>6s}  {'Δ':>6s}  {'CI':>14s}")
    print("  " + "-" * 140)
    bot = sig[sig["delta"] < 0].sort_values("delta").head(20)
    for _, r in bot.iterrows():
        print(f"  {r['position']:3s}  {r['player_full']:35s}  {r['def_full']:35s}  {r['n']:4d}  {r['avg_ppr']:6.2f}  {r['baseline']:6.2f}  {r['delta']:+6.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    return results_df


# =====================================================================
# Layer 3: Combos — pairs of players on same team vs defense
# =====================================================================

def analyze_combos(df):
    """How do pairs of player archetypes on the same team perform vs defenses?"""
    print("\n" + "=" * 80)
    print("LAYER 3: ARCHETYPE COMBOS (same team) vs DEFENSE")
    print("=" * 80)

    # Build team-game-level: for each team/season/week, what archetypes were active?
    game_players = df.groupby(["team", "season", "week", "opponent", "def_arch"]).apply(
        lambda g: g[["player_id", "position", "player_arch", "player_style", "ppr"]].to_dict("records"),
    ).reset_index(name="players")

    results = []

    # Define interesting combo types
    combo_types = [
        ("QB", "WR"),
        ("QB", "RB"),
        ("QB", "TE"),
        ("RB", "WR"),
        ("WR", "WR"),
        ("WR", "TE"),
    ]

    for pos1, pos2 in combo_types:
        combo_data = []

        for _, game in game_players.iterrows():
            players = game["players"]
            def_arch = game["def_arch"]

            pos1_players = [p for p in players if p["position"] == pos1]
            pos2_players = [p for p in players if p["position"] == pos2]

            if pos1 == pos2:
                # Same position: pairs within
                for p1, p2 in combinations(pos1_players, 2):
                    combo_ppr = p1["ppr"] + p2["ppr"]
                    arch_combo = tuple(sorted([p1["player_arch"], p2["player_arch"]]))
                    style_combo = tuple(sorted([
                        f"{p1['player_arch']}/{p1['player_style']}",
                        f"{p2['player_arch']}/{p2['player_style']}",
                    ]))
                    combo_data.append({
                        "arch_combo": " + ".join(arch_combo),
                        "style_combo": " + ".join(style_combo),
                        "def_arch": def_arch,
                        "combo_ppr": combo_ppr,
                        "pos_combo": f"{pos1}+{pos2}",
                    })
            else:
                # Different positions
                for p1 in pos1_players:
                    for p2 in pos2_players:
                        combo_ppr = p1["ppr"] + p2["ppr"]
                        arch_combo = f"{p1['player_arch']} + {p2['player_arch']}"
                        style_combo = f"{p1['player_arch']}/{p1['player_style']} + {p2['player_arch']}/{p2['player_style']}"
                        combo_data.append({
                            "arch_combo": arch_combo,
                            "style_combo": style_combo,
                            "def_arch": def_arch,
                            "combo_ppr": combo_ppr,
                            "pos_combo": f"{pos1}+{pos2}",
                        })

        if not combo_data:
            continue

        combo_df = pd.DataFrame(combo_data)

        # Compute baselines per combo
        combo_baselines = combo_df.groupby("arch_combo")["combo_ppr"].mean().to_dict()

        # Test each combo vs defense
        for arch_combo in combo_df["arch_combo"].unique():
            baseline = combo_baselines[arch_combo]
            c_df = combo_df[combo_df["arch_combo"] == arch_combo]

            for d_arch in c_df["def_arch"].unique():
                matchup = c_df[c_df["def_arch"] == d_arch]
                n = len(matchup)
                if n < MIN_GAMES:
                    continue

                mean, lo, hi = bootstrap_ci(matchup["combo_ppr"].values)
                delta = mean - baseline

                results.append({
                    "pos_combo": c_df.iloc[0]["pos_combo"],
                    "arch_combo": arch_combo,
                    "def_arch": d_arch,
                    "n": n,
                    "avg_ppr": round(mean, 2),
                    "baseline": round(baseline, 2),
                    "delta": round(delta, 2),
                    "ci_lo": round(lo, 2),
                    "ci_hi": round(hi, 2),
                    "significant": lo > baseline or hi < baseline,
                })

    results_df = pd.DataFrame(results)
    sig = results_df[results_df["significant"]].copy()

    print(f"\n  {len(sig)} significant combo matchups out of {len(results_df)} tested")

    print("\n  TOP COMBO EDGES (positive, significant):")
    print(f"  {'Positions':8s}  {'Archetype Combo':45s}  {'vs Defense':22s}  {'N':>4s}  {'Avg':>7s}  {'Base':>7s}  {'Δ':>6s}  {'CI':>14s}")
    print("  " + "-" * 140)
    top = sig[sig["delta"] > 0].sort_values("delta", ascending=False).head(25)
    for _, r in top.iterrows():
        print(f"  {r['pos_combo']:8s}  {r['arch_combo']:45s}  {r['def_arch']:22s}  {r['n']:4d}  {r['avg_ppr']:7.2f}  {r['baseline']:7.2f}  {r['delta']:+6.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    print("\n  TOP COMBO EDGES (negative, significant):")
    print(f"  {'Positions':8s}  {'Archetype Combo':45s}  {'vs Defense':22s}  {'N':>4s}  {'Avg':>7s}  {'Base':>7s}  {'Δ':>6s}  {'CI':>14s}")
    print("  " + "-" * 140)
    bot = sig[sig["delta"] < 0].sort_values("delta").head(25)
    for _, r in bot.iterrows():
        print(f"  {r['pos_combo']:8s}  {r['arch_combo']:45s}  {r['def_arch']:22s}  {r['n']:4d}  {r['avg_ppr']:7.2f}  {r['baseline']:7.2f}  {r['delta']:+6.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    return results_df


# =====================================================================
# Save results
# =====================================================================

def save_results(conn, individual_df, granular_df, combo_df):
    """Save backtesting results to DB."""
    conn.executescript("""
        DROP TABLE IF EXISTS backtest_individual;
        DROP TABLE IF EXISTS backtest_granular;
        DROP TABLE IF EXISTS backtest_combos;

        CREATE TABLE backtest_individual (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position TEXT,
            player_arch TEXT,
            def_arch TEXT,
            n INTEGER,
            avg_ppr REAL,
            baseline REAL,
            delta REAL,
            delta_pct REAL,
            ci_lo REAL,
            ci_hi REAL,
            significant INTEGER
        );

        CREATE TABLE backtest_granular (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position TEXT,
            player_full TEXT,
            def_full TEXT,
            n INTEGER,
            avg_ppr REAL,
            baseline REAL,
            delta REAL,
            ci_lo REAL,
            ci_hi REAL,
            significant INTEGER
        );

        CREATE TABLE backtest_combos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pos_combo TEXT,
            arch_combo TEXT,
            def_arch TEXT,
            n INTEGER,
            avg_ppr REAL,
            baseline REAL,
            delta REAL,
            ci_lo REAL,
            ci_hi REAL,
            significant INTEGER
        );

        CREATE INDEX idx_bt_ind_sig ON backtest_individual(significant);
        CREATE INDEX idx_bt_gran_sig ON backtest_granular(significant);
        CREATE INDEX idx_bt_combo_sig ON backtest_combos(significant);
    """)

    if len(individual_df) > 0:
        individual_df["significant"] = individual_df["significant"].astype(int)
        individual_df.to_sql("backtest_individual", conn, if_exists="append", index=False)

    if len(granular_df) > 0:
        granular_df["significant"] = granular_df["significant"].astype(int)
        granular_df.to_sql("backtest_granular", conn, if_exists="append", index=False)

    if len(combo_df) > 0:
        combo_df["significant"] = combo_df["significant"].astype(int)
        combo_df.to_sql("backtest_combos", conn, if_exists="append", index=False)


def print_executive_summary(conn):
    """Print the most actionable findings."""
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY — MOST ACTIONABLE DFS EDGES")
    print("=" * 80)

    print("\n  SMASH SPOTS (play these matchups):")
    rows = conn.execute("""
        SELECT position, player_arch, def_arch, n, avg_ppr, baseline, delta, delta_pct, ci_lo, ci_hi
        FROM backtest_individual
        WHERE significant = 1 AND delta > 0
        ORDER BY delta_pct DESC
        LIMIT 15
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:3s} {r[1]:22s} vs {r[2]:22s}  n={r[3]:3d}  {r[4]:5.1f} PPR ({r[7]:+.1f}%)  [{r[8]:.1f}-{r[9]:.1f}]")

    print("\n  FADE SPOTS (avoid these matchups):")
    rows = conn.execute("""
        SELECT position, player_arch, def_arch, n, avg_ppr, baseline, delta, delta_pct, ci_lo, ci_hi
        FROM backtest_individual
        WHERE significant = 1 AND delta < 0
        ORDER BY delta_pct ASC
        LIMIT 15
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:3s} {r[1]:22s} vs {r[2]:22s}  n={r[3]:3d}  {r[4]:5.1f} PPR ({r[7]:+.1f}%)  [{r[8]:.1f}-{r[9]:.1f}]")

    print("\n  BEST COMBO STACKS:")
    rows = conn.execute("""
        SELECT pos_combo, arch_combo, def_arch, n, avg_ppr, baseline, delta, ci_lo, ci_hi
        FROM backtest_combos
        WHERE significant = 1 AND delta > 0
        ORDER BY delta DESC
        LIMIT 15
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:8s} {r[1]:45s} vs {r[2]:22s}  n={r[3]:3d}  {r[4]:5.1f} PPR (Δ={r[6]:+.1f})  [{r[7]:.1f}-{r[8]:.1f}]")

    print("\n  WORST COMBO STACKS (fade):")
    rows = conn.execute("""
        SELECT pos_combo, arch_combo, def_arch, n, avg_ppr, baseline, delta, ci_lo, ci_hi
        FROM backtest_combos
        WHERE significant = 1 AND delta < 0
        ORDER BY delta ASC
        LIMIT 15
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:8s} {r[1]:45s} vs {r[2]:22s}  n={r[3]:3d}  {r[4]:5.1f} PPR (Δ={r[6]:+.1f})  [{r[7]:.1f}-{r[8]:.1f}]")


def main():
    print("NFL Archetype Backtesting")
    print("=" * 80)

    np.random.seed(42)
    conn = sqlite3.connect(DB_PATH)

    print("\nLoading matchup data...")
    df = load_matchup_data(conn)
    print(f"  {len(df)} player-game matchups across {df['season'].nunique()} seasons")
    print(f"  Positions: {df.groupby('position')['ppr'].count().to_dict()}")

    baselines = compute_baselines(df)

    # Layer 1: Individual
    individual_df = analyze_individual(df, baselines)

    # Layer 2: Granular style matchups
    granular_df = analyze_granular(df, baselines)

    # Layer 3: Combos
    combo_df = analyze_combos(df)

    # Save
    print("\n\nSaving results to database...")
    save_results(conn, individual_df, granular_df, combo_df)

    # Executive summary
    print_executive_summary(conn)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
