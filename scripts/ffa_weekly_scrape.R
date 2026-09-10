# Weekly FFA raw stat lines for the SaberSim generator — TEMPLATE, untested on this machine
# (no R here; the working scrape lives on the home PC). Reproduce that script's settings here.
#
# Output must match data/ffanalytics/FFAn_weekly/raw_stats_2026_wk1.csv: one row per player with
# avg_type == "weighted", stat columns pass_yds, pass_tds, pass_int, rush_yds, rush_tds, rec,
# rec_yds, rec_tds, fumbles_lost (+ _sd), fg_*, xp, dst_*, injury_status, id, player, team, position.
# Include "rec" — the 2026 wk1 file lacked it and receptions had to be estimated.
#
# Usage:  Rscript scripts/ffa_weekly_scrape.R 2026 2
suppressPackageStartupMessages({ library(ffanalytics); library(dplyr); library(readr) })
args <- commandArgs(trailingOnly = TRUE)
season <- as.integer(args[1]); week <- as.integer(args[2])
pos <- c("QB", "RB", "WR", "TE", "K", "DST")
# sources: keep to the ones that were used for the 2023-25 backfill so the baseline is comparable
srcs <- c("CBS", "ESPN", "FantasyPros", "FantasySharks", "FFToday", "NumberFire", "FleaFlicker", "NFL")
raw <- scrape_data(src = srcs, pos = pos, season = season, week = week)
proj <- projections_table(raw, avg_type = "weighted") |> add_player_info()
# stat-level table (the generator scores it itself); keep the sd columns
out <- proj |> filter(avg_type == "weighted")
dir.create("data/ffanalytics/FFAn_weekly", showWarnings = FALSE, recursive = TRUE)
write_csv(out, sprintf("data/ffanalytics/FFAn_weekly/raw_stats_%d_wk%d.csv", season, week), na = "NA")
cat(sprintf("wrote raw_stats_%d_wk%d.csv (%d rows)\n", season, week, nrow(out)))
