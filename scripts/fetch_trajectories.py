"""
Build the career-trajectory dataset for the comp finder.

Pulls 2006-2025 seasonal stats (modern era where air-yards/target-share are
charted) and aligns every player-season to a CAREER YEAR (years_exp+1). Stores
per-game production + role metrics + static context as nflv_traj. Kept separate
from nflv_season so the existing season-model pipeline is untouched.
"""
import os, sqlite3
import numpy as np, pandas as pd
import nflreadpy as nflr

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
SEASONS = list(range(2006, 2026))
SKILL = ["QB", "RB", "WR", "TE"]


def pgify(df, cols, g):
    for c in cols:
        df[c + "_pg"] = df[c] / g
    return df


def main():
    print("Loading 2006-2025 seasonal stats ...")
    s = nflr.load_player_stats(SEASONS, summary_level="reg").to_pandas()
    s = s[s["position"].isin(SKILL)].copy()
    g = s["games"].clip(lower=1)
    s["ppg"] = s["fantasy_points_ppr"] / g
    s["total_tds"] = s[["passing_tds", "rushing_tds", "receiving_tds"]].sum(axis=1)
    for c in ["attempts", "passing_yards", "passing_tds", "passing_interceptions",
              "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
              "receiving_yards", "receiving_tds"]:
        if c in s.columns:
            s[c + "_pg"] = s[c] / g

    print("Loading rosters for age / experience ...")
    r = nflr.load_rosters(SEASONS).to_pandas()
    r["birth_year"] = pd.to_datetime(r["birth_date"], errors="coerce").dt.year
    r = r[["gsis_id", "season", "years_exp", "rookie_year", "entry_year",
           "birth_year", "height", "weight", "draft_number"]].rename(columns={"gsis_id": "player_id"})
    r = r.sort_values("years_exp").drop_duplicates(["player_id", "season"], keep="last")

    df = s.merge(r, on=["player_id", "season"], how="left")
    df["age"] = df["season"] - df["birth_year"]
    # career year: prefer years_exp+1; fallback to season - rookie/entry year + 1
    cy = df["years_exp"] + 1
    alt = df["season"] - df[["rookie_year", "entry_year"]].min(axis=1) + 1
    df["career_year"] = cy.where(cy.notna(), alt)
    df["draft_number"] = df["draft_number"].fillna(262)

    keep = ["player_id", "player_display_name", "position", "season", "career_year", "age",
            "years_exp", "rookie_year", "draft_number", "height", "weight", "games", "ppg",
            "total_tds", "target_share", "air_yards_share", "wopr", "passing_epa", "rushing_epa",
            "receiving_epa", "attempts_pg", "passing_yards_pg", "passing_tds_pg",
            "passing_interceptions_pg", "carries_pg", "rushing_yards_pg", "rushing_tds_pg",
            "targets_pg", "receptions_pg", "receiving_yards_pg", "receiving_tds_pg"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out[out["career_year"].notna() & (out["career_year"] >= 1) & (out["career_year"] <= 20)]

    con = sqlite3.connect(DB)
    out.to_sql("nflv_traj", con, if_exists="replace", index=False)
    con.close()
    print(f"\nnflv_traj: {len(out):,} player-seasons, {int(out.season.min())}-{int(out.season.max())}")
    print(f"unique players: {out.player_id.nunique():,}")
    # how many have a fully-observed career start (rookie_year >= 2006)?
    obs = out[out.rookie_year >= 2006]
    print(f"players entering 2006+ (fully observed arc): {obs.player_id.nunique():,}")
    print("career-year coverage:", out.groupby('career_year').size().head(8).to_dict())


if __name__ == "__main__":
    main()
