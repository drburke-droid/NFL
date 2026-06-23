"""
Extended nflverse ingestion for the season-projection (avenue 1) and
deepened weekly (avenue 2) deep dive.

Pulls many seasons of free nflverse data into new nflv_* tables in
db/nfl_odds.db. Idempotent: each table is dropped and recreated.

Spine player id = gsis_id (== player_stats.player_id). snaps/combine join
via pfr_id, which rosters provides (pfr_id <-> gsis_id).

Skips actual preseason game stats (weak fantasy signal, per scoping).
"""
import os, sqlite3, sys
import nflreadpy as nflr
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")

# 2011 start so 2012 has a prior season; snaps/depth begin 2012.
SEASON_STATS = list(range(2011, 2026))   # weekly + seasonal aggregates
SEASON_ROLE  = list(range(2012, 2026))   # snaps / depth / injuries era

def log(msg):
    print(msg, flush=True)

def write(con, name, df, cols=None):
    if df is None or len(df) == 0:
        log(f"  !! {name}: empty, skipped"); return
    if cols:
        df = df[[c for c in cols if c in df.columns]].copy()
    df.to_sql(name, con, if_exists="replace", index=False)
    log(f"  OK {name}: {len(df):,} rows x {df.shape[1]} cols")

def pl(df):
    return df.to_pandas()

def main():
    con = sqlite3.connect(DB)

    # ---- 1. Seasonal regular-season aggregates (avenue-1 spine) ----
    try:
        log("[1/9] seasonal reg aggregates ...")
        s = pl(nflr.load_player_stats(SEASON_STATS, summary_level="reg"))
        keep = ["player_id","player_display_name","position","position_group","season",
                "recent_team","games","completions","attempts","passing_yards","passing_tds",
                "passing_interceptions","passing_air_yards","passing_epa","passing_cpoe","pacr",
                "carries","rushing_yards","rushing_tds","rushing_first_downs","rushing_epa",
                "receptions","targets","receiving_yards","receiving_tds","receiving_air_yards",
                "receiving_first_downs","receiving_epa","racr","target_share","air_yards_share",
                "wopr","fantasy_points","fantasy_points_ppr"]
        write(con, "nflv_season", s, keep)
    except Exception as e:
        log(f"  ERR seasonal: {e}")

    # ---- 2. Weekly stats (consistency, availability, avenue-2) ----
    try:
        log("[2/9] weekly stats ...")
        w = pl(nflr.load_player_stats(SEASON_STATS, summary_level="week"))
        keep = ["player_id","player_display_name","position","position_group","season","week",
                "season_type","team","opponent_team","completions","attempts","passing_yards","passing_tds",
                "passing_interceptions","passing_epa","carries","rushing_yards","rushing_tds",
                "rushing_epa","receptions","targets","receiving_yards","receiving_tds",
                "receiving_air_yards","receiving_epa","target_share","air_yards_share","wopr",
                "fantasy_points","fantasy_points_ppr"]
        write(con, "nflv_weekly", w, keep)
    except Exception as e:
        log(f"  ERR weekly: {e}")

    # ---- 3. Rosters (age, experience, draft, size, id crosswalk) ----
    try:
        log("[3/9] rosters ...")
        r = pl(nflr.load_rosters(SEASON_STATS))
        keep = ["season","team","position","status","full_name","birth_date","height","weight",
                "college","gsis_id","pfr_id","years_exp","entry_year","rookie_year",
                "draft_club","draft_number"]
        write(con, "nflv_rosters", r, keep)
    except Exception as e:
        log(f"  ERR rosters: {e}")

    # ---- 4. Draft picks (draft capital) ----
    try:
        log("[4/9] draft picks ...")
        d = pl(nflr.load_draft_picks(True))
        keep = ["season","round","pick","team","gsis_id","pfr_player_id","pfr_player_name",
                "position","college","age"]
        write(con, "nflv_draft", d, keep)
    except Exception as e:
        log(f"  ERR draft: {e}")

    # ---- 5. Combine (athleticism) ----
    try:
        log("[5/9] combine ...")
        c = pl(nflr.load_combine(True))
        keep = ["season","draft_year","draft_round","draft_ovr","pfr_id","player_name","pos",
                "school","ht","wt","forty","bench","vertical","broad_jump","cone","shuttle"]
        write(con, "nflv_combine", c, keep)
    except Exception as e:
        log(f"  ERR combine: {e}")

    # ---- 6. Snap counts (role / usage) ----
    try:
        log("[6/9] snap counts ...")
        sn = pl(nflr.load_snap_counts(SEASON_ROLE))
        keep = ["season","week","game_type","player","pfr_player_id","position","team",
                "opponent","offense_snaps","offense_pct"]
        write(con, "nflv_snaps", sn, keep)
    except Exception as e:
        log(f"  ERR snaps: {e}")

    # ---- 7. Depth charts (depth rank) ----
    try:
        log("[7/9] depth charts ...")
        dc = pl(nflr.load_depth_charts(SEASON_ROLE))
        dc = dc[dc["position"].isin(["QB","RB","FB","WR","TE"])] if "position" in dc.columns else dc
        keep = ["season","week","game_type","club_code","gsis_id","full_name","position",
                "depth_position","depth_team","formation"]
        write(con, "nflv_depth", dc, keep)
    except Exception as e:
        log(f"  ERR depth: {e}")

    # ---- 8. ff_opportunity (expected fantasy points / xFP) ----
    try:
        log("[8/9] ff_opportunity ...")
        fo = pl(nflr.load_ff_opportunity(SEASON_ROLE, stat_type="weekly"))
        keep = ["season","week","posteam","player_id","full_name","position",
                "total_fantasy_points","total_fantasy_points_exp","total_fantasy_points_diff",
                "pass_fantasy_points_exp","rec_fantasy_points_exp","rush_fantasy_points_exp",
                "rec_attempt","rush_attempt","pass_attempt",
                "rec_touchdown_exp","rush_touchdown_exp","pass_touchdown_exp",
                "rec_yards_gained_exp","rush_yards_gained_exp"]
        write(con, "nflv_ff_opp", fo, keep)
    except Exception as e:
        log(f"  ERR ff_opportunity: {e}")

    # ---- 9. Current preseason rankings snapshot (for later application) ----
    try:
        log("[9/9] ff_rankings (current draft ECR/ADP snapshot) ...")
        fr = pl(nflr.load_ff_rankings("draft"))
        keep = ["player","id","pos","team","ecr","sd","best","worst",
                "player_owned_avg","player_owned_espn","player_owned_yahoo","bye","scrape_date"]
        write(con, "nflv_ff_rankings_current", fr, keep)
    except Exception as e:
        log(f"  ERR ff_rankings: {e}")

    con.commit(); con.close()
    log("DONE.")

if __name__ == "__main__":
    main()
