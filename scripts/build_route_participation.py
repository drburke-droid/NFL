"""Per-player-game route participation + coverage/pressure context from the FREE nflverse
participation feed (offense_players on every dropback) joined to PBP.

Why: the Sept-2026 DFS spec buys routes / TPRR / coverage / pressure from Fantasy Points Data;
nflverse participation carries the same dimensions (route of the targeted receiver, coverage
type, man/zone, was_pressure, time_to_throw, box count, personnel) for 2016-2025, complete
from 2023. A player on the field for a dropback ≈ ran a route (standard nflverse proxy).

Output: data/participation/routes_{S}.parquet, one row per (season, week, game_id, team, gsis_id):
  routes, team_dropbacks, route_share, targets_on_routes, tprr, rz_routes (yardline<=20),
  man_routes, zone_routes, pressure_routes, ttt_mean, plus team-level dropbacks/pressure/man rates.
Also data/participation/team_def_{S}.parquet: opponent defence coverage/man/pressure rates.
Usage: python scripts/build_route_participation.py 2016 2025
"""
import sys, os
import polars as pl, nflreadpy as nfl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "participation"); os.makedirs(OUT, exist_ok=True)
s0, s1 = int(sys.argv[1]), int(sys.argv[2])
SKILL = {"WR", "TE", "RB", "FB", "HB"}
players = nfl.load_players().select(["gsis_id", "position"]).drop_nulls("gsis_id").unique(subset=["gsis_id"]).rename({"gsis_id": "_pl", "position": "_pos"})
for s in range(s0, s1 + 1):
    part = nfl.load_participation([s])
    pbp = nfl.load_pbp([s]).select(["play_id", "game_id", "week", "season", "posteam", "defteam",
        "qb_dropback", "receiver_player_id", "yardline_100", "pass_attempt", "sack", "qb_scramble",
        "air_yards", "complete_pass", "yards_gained", "pass_touchdown"])
    part = part.with_columns(pl.col("play_id").cast(pl.Int64)); pbp = pbp.with_columns(pl.col("play_id").cast(pl.Int64))
    j = part.join(pbp, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner")
    d = j.filter(pl.col("qb_dropback") == 1).with_columns([
        pl.col("offense_players").str.split(";").alias("_pl"),
        (pl.col("defense_man_zone_type") == "MAN_COVERAGE").cast(pl.Int8).alias("is_man"),
        pl.col("was_pressure").cast(pl.Int8).alias("is_press"),
        (pl.col("yardline_100") <= 20).cast(pl.Int8).alias("is_rz"),
        (pl.col("air_yards") >= 15).cast(pl.Int8).alias("is_deep"),
    ])
    # team-game context (offence's dropbacks; defence's rates faced)
    tg = d.group_by(["season", "week", "nflverse_game_id", "posteam", "defteam"]).agg([
        pl.len().alias("team_dropbacks"), pl.col("is_man").mean().alias("def_man_rate"),
        pl.col("is_press").mean().alias("def_pressure_rate"), pl.col("time_to_throw").mean().alias("ttt_mean"),
        pl.col("defenders_in_box").mean().alias("box_mean"),
        (pl.col("defense_coverage_type").is_in(["COVER_2", "COVER_4", "COVER_6", "2_MAN"])).mean().alias("def_two_high_rate"),
        pl.col("pass_attempt").sum().alias("team_pass_att")])
    tg.write_parquet(os.path.join(OUT, f"team_def_{s}.parquet"))
    # explode to player-play rows
    e = d.select(["season", "week", "nflverse_game_id", "posteam", "defteam", "play_id", "_pl",
                  "receiver_player_id", "is_man", "is_press", "is_rz", "is_deep", "route", "time_to_throw",
                  "complete_pass", "yards_gained", "pass_touchdown"]) \
         .explode("_pl").join(players, on="_pl", how="left").filter(pl.col("_pos").is_in(list(SKILL)))
    e = e.with_columns([(pl.col("_pl") == pl.col("receiver_player_id")).cast(pl.Int8).alias("tgt")])
    p = e.group_by(["season", "week", "nflverse_game_id", "posteam", "defteam", "_pl"]).agg([
        pl.len().alias("routes"), pl.col("tgt").sum().alias("targets_on_routes"),
        (pl.col("tgt") * pl.col("is_man")).sum().alias("tgt_man"), pl.col("is_man").sum().alias("man_routes"),
        (pl.col("tgt") * pl.col("is_press")).sum().alias("tgt_press"), pl.col("is_press").sum().alias("press_routes"),
        pl.col("is_rz").sum().alias("rz_routes"), (pl.col("tgt") * pl.col("is_rz")).sum().alias("rz_targets"),
        (pl.col("tgt") * pl.col("is_deep")).sum().alias("deep_targets"),
        (pl.col("tgt") * pl.col("yards_gained")).sum().alias("yds_on_tgt"),
        (pl.col("tgt") * pl.col("complete_pass")).sum().alias("rec_on_tgt"),
        (pl.col("tgt") * pl.col("pass_touchdown")).sum().alias("td_on_tgt"),
        pl.col("_pos").first().alias("pos")]).rename({"_pl": "gsis_id"})
    p = p.join(tg.select(["season", "week", "nflverse_game_id", "posteam", "team_dropbacks", "def_man_rate",
                          "def_pressure_rate"]), on=["season", "week", "nflverse_game_id", "posteam"], how="left")
    p = p.with_columns([(pl.col("routes") / pl.col("team_dropbacks")).alias("route_share"),
                        (pl.col("targets_on_routes") / pl.col("routes")).alias("tprr"),
                        (pl.col("yds_on_tgt") / pl.col("routes")).alias("yprr")])
    p.write_parquet(os.path.join(OUT, f"routes_{s}.parquet"))
    print(s, "dropbacks", d.height, "player-games", p.height, "teams", tg.height, flush=True)
