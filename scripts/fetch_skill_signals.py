"""Ingest play-by-play skill metrics, NGS season stats, and injury reports.

Creates three families of tables in db/nfl_odds.db:
  nflv_pbp_skill_wk  - per player-week-role advanced skill aggregates from PBP
  nflv_pbp_skill     - per player-season-role, with player-relative H1/H2 splits
  nflv_ngs_*         - Next Gen Stats (passing/receiving/rushing), all weeks
  nflv_injuries      - weekly injury report rows (2011+)

Skill metrics are per-play efficiency (EPA/play, CPOE, success rate, YAC over
expected, aDOT) so downstream work can separate skill change from volume change.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

DB = Path(__file__).resolve().parents[1] / "db" / "nfl_odds.db"
SEASONS = list(range(2011, 2026))
NGS_SEASONS = list(range(2016, 2026))

PBP_COLS = [
    "season", "week", "season_type", "posteam",
    "passer_player_id", "qb_dropback", "qb_epa", "epa", "cpoe",
    "complete_pass", "pass_attempt", "sack", "qb_scramble",
    "air_yards", "success", "interception", "pass_touchdown",
    "rusher_player_id", "rush_attempt", "rushing_yards", "rush_touchdown",
    "receiver_player_id", "receiving_yards", "yards_after_catch",
    "xyac_mean_yardage",
]


def weekly_skill_for_season(season: int) -> pd.DataFrame:
    pl_df = nfl.load_pbp(season)
    keep = [c for c in PBP_COLS if c in pl_df.columns]
    df = pl_df.select(keep).to_pandas()
    df = df[df["season_type"] == "REG"].copy()

    out = []

    # --- QB: dropback-level efficiency ---
    qb = df[(df["qb_dropback"] == 1) & df["passer_player_id"].notna() & df["epa"].notna()]
    g = qb.groupby(["passer_player_id", "week"])
    att = qb[qb["pass_attempt"] == 1]
    ga = att.groupby(["passer_player_id", "week"])
    qbw = pd.DataFrame({
        "plays": g.size(),
        "epa_play": g["qb_epa"].mean(),
        "success": g["success"].mean(),
        "cpoe": ga["cpoe"].mean(),
        "adot": ga["air_yards"].mean(),
        "sack_rate": g["sack"].mean(),
        "scramble_rate": g["qb_scramble"].mean(),
        "comp_pct": ga["complete_pass"].mean(),
        "int_rate": ga["interception"].mean(),
        "td_rate": ga["pass_touchdown"].mean(),
    }).reset_index().rename(columns={"passer_player_id": "gsis_id"})
    qbw["role"] = "pass"
    out.append(qbw)

    # --- Rusher: designed carries only (no scrambles) ---
    ru = df[(df["rush_attempt"] == 1) & (df["qb_scramble"] != 1)
            & df["rusher_player_id"].notna() & df["epa"].notna()]
    g = ru.groupby(["rusher_player_id", "week"])
    ruw = pd.DataFrame({
        "plays": g.size(),
        "epa_play": g["epa"].mean(),
        "success": g["success"].mean(),
        "ypc": g["rushing_yards"].mean(),
        "td_rate": g["rush_touchdown"].mean(),
    }).reset_index().rename(columns={"rusher_player_id": "gsis_id"})
    ruw["role"] = "rush"
    out.append(ruw)

    # --- Receiver: target-level efficiency ---
    tg = df[(df["pass_attempt"] == 1) & df["receiver_player_id"].notna() & df["epa"].notna()]
    tg = tg.copy()
    tg["yacoe"] = np.where(
        (tg["complete_pass"] == 1) & tg["xyac_mean_yardage"].notna(),
        tg["yards_after_catch"] - tg["xyac_mean_yardage"], np.nan)
    g = tg.groupby(["receiver_player_id", "week"])
    rew = pd.DataFrame({
        "plays": g.size(),
        "epa_play": g["epa"].mean(),
        "success": g["success"].mean(),
        "adot": g["air_yards"].mean(),
        "catch_rate": g["complete_pass"].mean(),
        "yacoe": g["yacoe"].mean(),
        "ypt": g["receiving_yards"].mean(),
    }).reset_index().rename(columns={"receiver_player_id": "gsis_id"})
    rew["role"] = "rec"
    out.append(rew)

    wk = pd.concat(out, ignore_index=True)
    wk["season"] = season
    return wk


def season_from_weekly(wk: pd.DataFrame) -> pd.DataFrame:
    """Collapse weekly rows to season rows with player-relative H1/H2 splits."""
    metrics = ["epa_play", "success", "cpoe", "adot", "sack_rate", "scramble_rate",
               "comp_pct", "int_rate", "td_rate", "ypc", "catch_rate", "yacoe", "ypt"]
    rows = []
    for (gsis, season, role), grp in wk.groupby(["gsis_id", "season", "role"]):
        grp = grp.sort_values("week")
        w = grp["plays"].to_numpy(dtype=float)
        rec = {"gsis_id": gsis, "season": season, "role": role,
               "games": len(grp), "plays": int(w.sum())}
        half = len(grp) // 2
        for m in metrics:
            if m not in grp or grp[m].isna().all():
                continue
            v = grp[m].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            rec[m] = np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
            if len(grp) >= 6:  # need enough games for a meaningful split
                for tag, sl in (("h1", slice(0, half)), ("h2", slice(half, None))):
                    vs, ws = v[sl], w[sl]
                    oks = ~np.isnan(vs)
                    rec[f"{m}_{tag}"] = (np.average(vs[oks], weights=ws[oks])
                                         if oks.any() else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)

    # ---------- PBP skill ----------
    all_wk = []
    for season in SEASONS:
        wk = weekly_skill_for_season(season)
        all_wk.append(wk)
        print(f"pbp {season}: {len(wk):,} player-week rows", flush=True)
    wk = pd.concat(all_wk, ignore_index=True)
    wk.to_sql("nflv_pbp_skill_wk", con, if_exists="replace", index=False)

    sea = season_from_weekly(wk)
    sea.to_sql("nflv_pbp_skill", con, if_exists="replace", index=False)
    print(f"nflv_pbp_skill: {len(sea):,} player-season-role rows")

    # ---------- NGS ----------
    for stat_type in ("passing", "receiving", "rushing"):
        ngs = nfl.load_nextgen_stats(NGS_SEASONS, stat_type=stat_type).to_pandas()
        ngs = ngs[ngs["season_type"] == "REG"]
        ngs.to_sql(f"nflv_ngs_{stat_type}", con, if_exists="replace", index=False)
        print(f"nflv_ngs_{stat_type}: {len(ngs):,} rows")

    # ---------- Injuries ----------
    inj = nfl.load_injuries(SEASONS).to_pandas()
    keep = [c for c in ["season", "week", "season_type", "team", "gsis_id",
                        "full_name", "position", "report_primary_injury",
                        "report_secondary_injury", "report_status",
                        "practice_status", "date_modified"] if c in inj.columns]
    inj[keep].to_sql("nflv_injuries", con, if_exists="replace", index=False)
    print(f"nflv_injuries: {len(inj):,} rows, seasons "
          f"{inj['season'].min()}-{inj['season'].max()}")

    con.close()


if __name__ == "__main__":
    main()
