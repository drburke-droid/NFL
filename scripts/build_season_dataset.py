"""
Build the avenue-1 season-projection dataset.

One row per (player, target_season Y). Features use ONLY information knowable
before season Y: prior-season (Y-1) production/role + static context
(age, draft capital, experience, size, combine). Targets are season-Y outcomes:
PPG, total PPR, games played, and positional finish / VBD.

Writes table: season_dataset
"""
import os, sqlite3
import numpy as np
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
SKILL = ["QB", "RB", "WR", "TE"]
# Replacement ranks for VBD (typical 12-team league starters + flex baseline)
REPL_RANK = {"QB": 14, "RB": 30, "WR": 36, "TE": 14}


def build_season_frame(con):
    """Per-(player,season) enriched frame: production + role + static context + VBD.
    This is the shared basis for both the YoY dataset and forward projections."""
    season = pd.read_sql("SELECT * FROM nflv_season", con)
    weekly = pd.read_sql("SELECT * FROM nflv_weekly WHERE season_type='REG'", con)
    rosters = pd.read_sql("SELECT * FROM nflv_rosters", con)
    draft = pd.read_sql("SELECT * FROM nflv_draft", con)
    combine = pd.read_sql("SELECT * FROM nflv_combine", con)
    snaps = pd.read_sql("SELECT * FROM nflv_snaps WHERE game_type='REG'", con)

    season = season[season["position"].isin(SKILL)].copy()
    season["ppg"] = season["fantasy_points_ppr"] / season["games"].clip(lower=1)

    # --- per-game rate features from season totals ---
    for c in ["passing_yards","passing_tds","carries","rushing_yards","rushing_tds",
              "receptions","targets","receiving_yards","receiving_tds","attempts"]:
        season[f"{c}_pg"] = season[c] / season["games"].clip(lower=1)
    # TD regression signals: TDs per yard (high => positive TD luck likely to regress)
    season["rec_td_rate"] = season["receiving_tds"] / season["receiving_yards"].replace(0, np.nan)
    season["rush_td_rate"] = season["rushing_tds"] / season["rushing_yards"].replace(0, np.nan)
    season["total_tds"] = season[["passing_tds","rushing_tds","receiving_tds"]].sum(axis=1)

    # --- weekly consistency (std / boom rate) per player-season ---
    wk = weekly[weekly["position"].isin(SKILL)].copy()
    cons = wk.groupby(["player_id","season"]).agg(
        wk_games=("fantasy_points_ppr","size"),
        ppr_std=("fantasy_points_ppr","std"),
        ppr_mean=("fantasy_points_ppr","mean"),
    ).reset_index()
    cons["cv"] = cons["ppr_std"] / cons["ppr_mean"].replace(0, np.nan)
    season = season.merge(cons[["player_id","season","ppr_std","cv"]], on=["player_id","season"], how="left")

    # --- snaps -> season mean offense_pct (join via pfr_id) ---
    snap_season = snaps.groupby(["pfr_player_id","season"]).agg(
        off_pct=("offense_pct","mean"), snap_games=("offense_pct","size")).reset_index()
    pfr_map = rosters[["gsis_id","pfr_id","season"]].dropna(subset=["pfr_id"]).drop_duplicates(["pfr_id","season"])
    snap_season = snap_season.merge(pfr_map, left_on=["pfr_player_id","season"],
                                    right_on=["pfr_id","season"], how="left")
    snap_season = snap_season.rename(columns={"gsis_id":"player_id"})
    season = season.merge(snap_season[["player_id","season","off_pct"]],
                          on=["player_id","season"], how="left")

    # --- static context: draft capital, combine, birth/exp from rosters ---
    ros = rosters.copy()
    ros["birth_year"] = pd.to_datetime(ros["birth_date"], errors="coerce").dt.year
    ros_ctx = ros[["gsis_id","season","birth_year","years_exp","height","weight",
                   "draft_number","team"]].rename(columns={"gsis_id":"player_id"})
    ros_ctx = ros_ctx.sort_values("years_exp").drop_duplicates(["player_id","season"], keep="last")
    season = season.merge(ros_ctx, on=["player_id","season"], how="left")
    season["age"] = season["season"] - season["birth_year"]

    dcap = draft[["gsis_id","round","pick"]].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    dcap = dcap.rename(columns={"gsis_id":"player_id","round":"draft_round","pick":"draft_pick"})
    season = season.merge(dcap, on="player_id", how="left")
    # undrafted -> sentinel
    season["draft_pick"] = season["draft_pick"].fillna(262)
    season["draft_round"] = season["draft_round"].fillna(8)

    comb = combine[["pfr_id","forty","vertical","broad_jump","cone","shuttle","bench"]].drop_duplicates("pfr_id")
    comb = comb.merge(rosters[["gsis_id","pfr_id"]].dropna().drop_duplicates("pfr_id"), on="pfr_id", how="left")
    comb = comb.rename(columns={"gsis_id":"player_id"}).drop(columns=["pfr_id"])
    season = season.merge(comb.drop_duplicates("player_id"), on="player_id", how="left")

    # --- positional finish & VBD per actual season ---
    season["pos_finish"] = season.groupby(["season","position"])["fantasy_points_ppr"].rank(
        ascending=False, method="min")
    # replacement PPG per season/pos from the player at REPL_RANK by total points
    repl = []
    for (s, p), g in season.groupby(["season","position"]):
        gg = g.sort_values("fantasy_points_ppr", ascending=False).reset_index(drop=True)
        idx = min(REPL_RANK[p]-1, len(gg)-1)
        repl.append({"season": s, "position": p, "repl_ppg": gg.loc[idx, "ppg"]})
    repl = pd.DataFrame(repl)
    season = season.merge(repl, on=["season","position"], how="left")
    season["vbd"] = (season["ppg"] - season["repl_ppg"]) * season["games"]
    return season


# columns carried from a season into the "prior_*" feature block
FEAT_COLS = ["player_id","player_display_name","position","season","recent_team","games",
             "fantasy_points_ppr","ppg","ppr_std","cv","off_pct","target_share",
             "air_yards_share","wopr","passing_epa","rushing_epa","receiving_epa",
             "total_tds","rec_td_rate","rush_td_rate",
             "passing_yards_pg","passing_tds_pg","attempts_pg","carries_pg","rushing_yards_pg",
             "rushing_tds_pg","receptions_pg","targets_pg","receiving_yards_pg","receiving_tds_pg"]


def main():
    con = sqlite3.connect(DB)
    season = build_season_frame(con)
    con.close()

    # --- assemble Y-1 -> Y rows ---
    feat_cols = FEAT_COLS
    prior = season[feat_cols].copy()
    prior.columns = ["player_id","player_display_name","position","prior_season","prior_team"] + \
                    ["prior_"+c for c in feat_cols[5:]]
    prior["season"] = prior["prior_season"] + 1  # align to target year

    # two-years-ago ppg for trend
    prior2 = season[["player_id","season","ppg"]].rename(columns={"ppg":"prior2_ppg"})
    prior2["season"] = prior2["season"] + 2

    target = season[["player_id","season","position","recent_team","age","years_exp","height",
                     "weight","draft_round","draft_pick","forty","vertical","broad_jump","cone",
                     "shuttle","games","ppg","fantasy_points_ppr","pos_finish","vbd"]].copy()
    target = target.rename(columns={"games":"next_games","ppg":"next_ppg",
                                    "fantasy_points_ppr":"next_ppr_total",
                                    "pos_finish":"next_pos_finish","vbd":"next_vbd",
                                    "recent_team":"team"})

    df = target.merge(prior.drop(columns=["position","player_display_name"]),
                      on=["player_id","season"], how="left")
    df = df.merge(prior2, on=["player_id","season"], how="left")

    # team change flag
    df["team_change"] = (df["team"] != df["prior_team"]).astype(int)
    df.loc[df["prior_team"].isna(), "team_change"] = np.nan

    # keep rows that have a prior season (the projection premise)
    df = df[df["prior_season"].notna()].copy()

    df.to_sql("season_dataset", con if False else sqlite3.connect(DB), if_exists="replace", index=False)
    print(f"season_dataset: {len(df):,} rows, {df['season'].min():.0f}-{df['season'].max():.0f}")
    print("rows w/ prior production:", df["prior_ppg"].notna().sum())
    print("by position:\n", df["position"].value_counts())
    print("target-season coverage:\n", df.groupby("season").size())


if __name__ == "__main__":
    main()
