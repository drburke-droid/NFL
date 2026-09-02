#!/usr/bin/env python3
"""Build the cheap-stars frame: draft-day price vs season positional finish.

One row per (player, season) for QB/RB/WR/TE that appears in the league-scored
FantasyFootballAnalytics preseason file (i.e. had a real auction market that
spring). Everything on the left of the equation is knowable BEFORE the auction:
FFA price/projection, age, experience, NFL draft capital, prior-season role and
production, team change, vacated opportunity on the new team. Everything on the
right is the season that followed, scored in THIS league's settings
(full PPR, 6-pt pass TD, -1 INT) with a positional finish rank.

Sources, in order of preference:
  1. db/nfl_odds.db (nflv_season / nflv_rosters / nflv_draft / nflv_snaps) when
     the synced DB is present -- same tables the rest of the repo uses.
  2. data/nflverse_cache/*.parquet, downloaded from the nflverse-data releases
     on first run. Keeps this study runnable on a fresh clone with no DB.

    python scripts/cheap_stars_build.py            # writes the cached frame
"""
import os
import re
import sqlite3
import sys
import urllib.request

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
FFA_DIR = os.path.join(ROOT, "data", "ffanalytics", "FFAn_league")
OUT = os.path.join(ROOT, "outputs", "models", "cheap_stars_frame.pkl")

SKILL = ("QB", "RB", "WR", "TE")
FIRST, LAST = 2012, 2025          # nflverse seasons pulled (FFA window is narrower)
REL = "https://github.com/nflverse/nflverse-data/releases/download"

# This league (ESPN 1211359110): full PPR, 6-pt passing TD, -1 INT, -1 fumble lost.
SCORING = {
    "passing_yards": 0.04, "passing_tds": 6.0, "interceptions": -1.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0,
    "receiving_2pt_conversions": 2.0, "special_teams_tds": 6.0,
    "fumbles_lost": -1.0,
}

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(s):
    s = str(s).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("-", " ").replace(",", "")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch(name, url):
    """Download a nflverse asset into the local cache once."""
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    os.makedirs(CACHE, exist_ok=True)
    print(f"  downloading {name} ...")
    urllib.request.urlretrieve(url, path)
    return path


# --------------------------------------------------------------------------- #
# raw loads
# --------------------------------------------------------------------------- #
def load_nflverse():
    """season stats / rosters / draft picks / snap counts, DB first then cache."""
    if os.path.exists(DB):
        con = sqlite3.connect(DB)
        try:
            season = pd.read_sql(
                "SELECT * FROM nflv_season WHERE season_type='REG'", con)
            rosters = pd.read_sql("SELECT * FROM nflv_rosters", con)
            draft = pd.read_sql("SELECT * FROM nflv_draft", con)
            snaps = pd.read_sql(
                "SELECT * FROM nflv_snaps WHERE game_type='REG'", con)
            print("sources: db/nfl_odds.db")
            return season, rosters, draft, snaps
        except Exception as exc:            # table missing -> fall through
            print(f"  DB present but unusable ({exc}); using nflverse cache")
        finally:
            con.close()

    print("sources: data/nflverse_cache (nflverse-data releases)")
    # --- season totals: per-season assets exist through 2024; 2025 is weekly-only
    frames = []
    for yr in range(FIRST, LAST + 1):
        try:
            p = fetch(f"player_stats_season_{yr}.parquet",
                      f"{REL}/player_stats/player_stats_season_{yr}.parquet")
            d = pd.read_parquet(p)
            frames.append(d[d.season_type == "REG"])
        except Exception:
            frames.append(weekly_to_season(yr))
    season = pd.concat(frames, ignore_index=True)

    rosters = pd.concat(
        [pd.read_parquet(fetch(f"roster_{y}.parquet", f"{REL}/rosters/roster_{y}.parquet"))
         for y in range(FIRST, LAST + 1)], ignore_index=True)
    draft = pd.read_parquet(
        fetch("draft_picks.parquet", f"{REL}/draft_picks/draft_picks.parquet"))
    snaps = []
    for y in range(FIRST, LAST + 1):
        try:
            s = pd.read_parquet(fetch(f"snap_counts_{y}.parquet",
                                      f"{REL}/snap_counts/snap_counts_{y}.parquet"))
            snaps.append(s[s.game_type == "REG"])
        except Exception:
            pass
    snaps = pd.concat(snaps, ignore_index=True)
    return season, rosters, draft, snaps


def weekly_to_season(yr):
    """Aggregate the modern weekly asset into the legacy season schema."""
    p = fetch(f"stats_player_week_{yr}.parquet",
              f"{REL}/stats_player/stats_player_week_{yr}.parquet")
    w = pd.read_parquet(p)
    w = w[w.season_type == "REG"].rename(columns={
        "passing_interceptions": "interceptions", "team": "recent_team"})
    for c in ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"):
        if c not in w.columns:
            w[c] = 0.0
    sums = ["completions", "attempts", "passing_yards", "passing_tds", "interceptions",
            "passing_2pt_conversions", "carries", "rushing_yards", "rushing_tds",
            "rushing_2pt_conversions", "receptions", "targets", "receiving_yards",
            "receiving_tds", "receiving_2pt_conversions", "receiving_air_yards",
            "special_teams_tds", "rushing_fumbles_lost", "receiving_fumbles_lost",
            "sack_fumbles_lost", "fantasy_points_ppr"]
    sums = [c for c in sums if c in w.columns]
    keys = ["player_id", "player_display_name", "position", "season"]
    g = w.groupby(keys, as_index=False)[sums].sum()
    grp = w.groupby(keys)
    g["games"] = grp.size().values
    g["recent_team"] = w.sort_values("week").groupby(keys)["recent_team"].last().values
    g["season_type"] = "REG"
    # exact season target/air-yards share from team weekly totals
    tm = w.groupby(["recent_team", "season"], as_index=False).agg(
        tm_tgt=("targets", "sum"), tm_ay=("receiving_air_yards", "sum"))
    plt = w.groupby(keys + ["recent_team"], as_index=False).agg(
        p_tgt=("targets", "sum"), p_ay=("receiving_air_yards", "sum"))
    plt = plt.merge(tm, on=["recent_team", "season"], how="left")
    plt["target_share"] = plt.p_tgt / plt.tm_tgt.replace(0, np.nan)
    plt["air_yards_share"] = plt.p_ay / plt.tm_ay.replace(0, np.nan)
    plt = plt.sort_values("p_tgt").drop_duplicates(keys, keep="last")
    g = g.merge(plt[keys + ["target_share", "air_yards_share"]], on=keys, how="left")
    g["wopr"] = 1.5 * g.target_share.fillna(0) + 0.7 * g.air_yards_share.fillna(0)
    return g


def load_ffa():
    """League-scored FFA preseason (wk0) files -> price/projection per season."""
    frames = []
    for f in sorted(os.listdir(FFA_DIR)):
        m = re.match(r"projections_(\d{4})_wk0\.csv$", f)
        if not m:
            continue
        d = pd.read_csv(os.path.join(FFA_DIR, f))
        d["season"] = int(m.group(1))
        frames.append(d)
    ffa = pd.concat(frames, ignore_index=True)
    ffa = ffa[ffa.position.isin(SKILL)].copy()
    ffa["nm"] = ffa.player.map(norm)
    # FFA's archive stamps each player's CURRENT age on every historical season --
    # unusable; age comes from nflverse rosters instead. `experience` is correct.
    return ffa.drop(columns=[c for c in ("age",) if c in ffa.columns])


# --------------------------------------------------------------------------- #
# frame
# --------------------------------------------------------------------------- #
def league_points(d):
    pts = np.zeros(len(d), dtype=float)
    fum = np.zeros(len(d), dtype=float)
    for c in ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"):
        if c in d.columns:
            fum += d[c].fillna(0).to_numpy(dtype=float)
    for col, w in SCORING.items():
        if col == "fumbles_lost":
            pts += w * fum
        elif col in d.columns:
            pts += w * d[col].fillna(0).to_numpy(dtype=float)
    return pts


def build():
    season, rosters, draft, snaps = load_nflverse()
    season = season[season.position.isin(SKILL) & season.season.between(FIRST, LAST)].copy()
    season["games"] = season["games"].fillna(0).clip(lower=0)
    season["lpts"] = league_points(season)
    season["ppg"] = season.lpts / season.games.clip(lower=1)
    season = (season.sort_values("lpts")
              .drop_duplicates(["player_id", "season"], keep="last"))

    # positional finish rank on season totals (what a roster actually banked)
    season["pos_rank"] = (season.groupby(["season", "position"])["lpts"]
                          .rank(ascending=False, method="min"))

    # --- static context -----------------------------------------------------
    ros = rosters.copy()
    ros["birth_year"] = pd.to_datetime(ros.birth_date, errors="coerce").dt.year
    ros = ros.rename(columns={"gsis_id": "player_id"})
    ros = (ros.sort_values("years_exp")
           .drop_duplicates(["player_id", "season"], keep="last"))
    season = season.merge(
        ros[["player_id", "season", "birth_year", "years_exp", "team", "height", "weight"]],
        on=["player_id", "season"], how="left")
    season["age"] = season.season - season.birth_year
    season["team"] = season.team.fillna(season.recent_team)

    dcap = (draft.rename(columns={"gsis_id": "player_id", "round": "draft_round",
                                  "pick": "draft_pick"})
            [["player_id", "draft_round", "draft_pick"]]
            .dropna(subset=["player_id"]).drop_duplicates("player_id"))
    season = season.merge(dcap, on="player_id", how="left")

    # --- snap share ---------------------------------------------------------
    sn = (snaps.groupby(["pfr_player_id", "season"], as_index=False)
          .agg(snap_pct=("offense_pct", "mean")))
    pfr = (rosters.rename(columns={"gsis_id": "player_id"})[["player_id", "pfr_id", "season"]]
           .dropna(subset=["pfr_id"]).drop_duplicates(["pfr_id", "season"]))
    sn = sn.merge(pfr, left_on=["pfr_player_id", "season"],
                  right_on=["pfr_id", "season"], how="left")
    season = season.merge(sn[["player_id", "season", "snap_pct"]].dropna(subset=["player_id"]),
                          on=["player_id", "season"], how="left")
    if season.snap_pct.max() and season.snap_pct.max() > 1.5:
        season["snap_pct"] = season.snap_pct / 100.0

    # --- per-game rates -----------------------------------------------------
    gp = season.games.clip(lower=1)
    for c in ("targets", "carries", "receiving_yards", "rushing_yards",
              "passing_yards", "attempts", "receptions"):
        if c in season.columns:
            season[f"{c}_pg"] = season[c].fillna(0) / gp
    season["touches_pg"] = season.get("carries_pg", 0) + season.get("receptions_pg", 0)

    # --- vacated opportunity: prior-season volume of players no longer on the
    #     roster in year Y (the "role opens" half of a breakout) --------------
    vac = season[["player_id", "season", "team", "targets", "carries"]].copy()
    vac["season"] = vac.season + 1            # carry Y-1 volume into year Y
    on_team = set(zip(season.player_id, season.season, season.team))
    vac["stays"] = [(p, s, t) in on_team for p, s, t in
                    zip(vac.player_id, vac.season, vac.team)]
    left = vac[~vac.stays].groupby(["team", "season"], as_index=False).agg(
        vac_targets=("targets", "sum"), vac_carries=("carries", "sum"))

    # --- prior-season (lag) frame ------------------------------------------
    lag_cols = ["ppg", "lpts", "games", "pos_rank", "snap_pct", "target_share",
                "air_yards_share", "wopr", "targets_pg", "carries_pg", "touches_pg",
                "receiving_yards_pg", "rushing_yards_pg", "team", "attempts_pg"]
    lag_cols = [c for c in lag_cols if c in season.columns]
    prior = season[["player_id", "season"] + lag_cols].copy()
    prior["season"] = prior.season + 1
    prior = prior.rename(columns={c: f"prior_{c}" for c in lag_cols})

    # career-best ppg over the 3 seasons before Y (min 6 games, so it means something)
    hist = season[["player_id", "season", "ppg", "games"]].copy()
    best = []
    for yr in range(FIRST + 1, LAST + 1):
        h = hist[(hist.season < yr) & (hist.season >= yr - 3) & (hist.games >= 6)]
        b = h.groupby("player_id", as_index=False).agg(career_best_ppg=("ppg", "max"))
        b["season"] = yr
        best.append(b)
    best = pd.concat(best, ignore_index=True)

    cur = season.copy()
    cur = cur.merge(prior, on=["player_id", "season"], how="left")
    cur = cur.merge(best, on=["player_id", "season"], how="left")
    cur = cur.merge(left, on=["team", "season"], how="left")
    cur["vac_targets"] = cur.vac_targets.fillna(0)
    cur["vac_carries"] = cur.vac_carries.fillna(0)
    cur["team_change"] = (cur.prior_team.notna() & (cur.prior_team != cur.team)).astype(int)
    cur["rookie"] = cur.prior_ppg.isna().astype(int)

    # --- FFA price ----------------------------------------------------------
    ffa = load_ffa()
    xw = cur[["player_id", "player_display_name", "position", "season"]].copy()
    xw["nm"] = xw.player_display_name.map(norm)
    xw = xw.drop_duplicates(["nm", "position", "season"])
    ffa = ffa.merge(xw[["nm", "position", "season", "player_id"]],
                    on=["nm", "position", "season"], how="left")
    keep = {"aav": "aav", "adp": "adp", "points": "ffa_points", "sd_pts": "ffa_sd",
            "floor": "ffa_floor", "ceiling": "ffa_ceiling", "points_vor": "ffa_vor",
            "rank": "ffa_rank", "position_rank": "ffa_pos_rank", "tier": "ffa_tier",
            "uncertainty": "ffa_unc", "dropoff": "ffa_dropoff", "experience": "ffa_exp"}
    ffa = ffa.rename(columns=keep)
    cols = ["player_id", "season", "position", "player"] + [v for v in keep.values() if v in ffa.columns]
    ffa_m = (ffa.dropna(subset=["player_id"])[cols]
             .sort_values("aav").drop_duplicates(["player_id", "season"], keep="last"))

    frame = ffa_m.merge(cur.drop(columns=["position"]), on=["player_id", "season"], how="left")
    frame["played"] = frame.games.notna() & (frame.games > 0)
    frame["games"] = frame.games.fillna(0)
    frame["lpts"] = frame.lpts.fillna(0.0)
    frame["ppg"] = frame.ppg.fillna(0.0)

    # players in the FFA file who never took a snap that year still finished
    # "unranked" -- give them a rank past the position's last real finisher.
    maxrank = frame.groupby(["season", "position"])["pos_rank"].transform("max")
    frame["pos_rank"] = frame.pos_rank.fillna(maxrank.fillna(200) + 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frame.to_pickle(OUT)
    print(f"\nwrote {OUT}  rows={len(frame)}  seasons={frame.season.min()}-{frame.season.max()}")
    print(frame.groupby("season").agg(n=("player_id", "size"),
                                      priced=("aav", lambda s: (s >= 1).sum())).to_string())
    return frame


if __name__ == "__main__":
    build()
