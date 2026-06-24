"""
Ingest kicker and team-defense (DST) data for the draft tool, scored under the
user's custom league.

Kicker pts = 3*(FG 0-39) + 4*(FG 40-49) + 5*(FG 50+) + 1*PAT.
DST pts    = 0.5*sacks + INT + fumble_recoveries + 6*def_TD + 2*safeties + 2*blocks.

Writes nflv_kicking and nflv_team_def (season totals).
"""
import os, sqlite3
import nflreadpy as nflr
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
SEASONS = list(range(2011, 2026))


def main():
    con = sqlite3.connect(DB)

    # ---- Kickers ----
    k = nflr.load_player_stats(SEASONS, summary_level="reg").to_pandas()
    k = k[k["position"] == "K"].copy()
    for c in ["fg_made_0_19","fg_made_20_29","fg_made_30_39","fg_made_40_49","fg_made_50_59",
              "fg_made_60_","pat_made"]:
        k[c] = k[c].fillna(0)
    k["custom_pts"] = (3*(k["fg_made_0_19"]+k["fg_made_20_29"]+k["fg_made_30_39"])
                       + 4*k["fg_made_40_49"] + 5*(k["fg_made_50_59"]+k["fg_made_60_"])
                       + 1*k["pat_made"])
    kick = k[["season","player_id","player_display_name","recent_team","games","custom_pts"]] \
            .rename(columns={"player_display_name":"name","recent_team":"team"})
    kick.to_sql("nflv_kicking", con, if_exists="replace", index=False)

    # ---- Team defense (DST) ----
    ts = nflr.load_team_stats(seasons=SEASONS).to_pandas()
    ts = ts[ts["season_type"] == "REG"].copy()   # regular season only (exclude playoffs)
    dcomp = ["def_sacks","def_interceptions","def_tds","def_safeties","fumble_recovery_opp",
             "fg_blocked","pat_blocked"]
    for c in dcomp:
        if c not in ts.columns: ts[c] = 0
        ts[c] = ts[c].fillna(0)
    g = ts.groupby(["team","season"]).agg(
        games=("week","nunique"),
        sacks=("def_sacks","sum"), ints=("def_interceptions","sum"),
        def_tds=("def_tds","sum"), safeties=("def_safeties","sum"),
        fum_rec=("fumble_recovery_opp","sum"),
        fg_blk=("fg_blocked","sum"), pat_blk=("pat_blocked","sum")).reset_index()
    g["custom_pts"] = (0.5*g["sacks"] + g["ints"] + g["fum_rec"] + 6*g["def_tds"]
                       + 2*g["safeties"] + 2*(g["fg_blk"]+g["pat_blk"]))
    g.to_sql("nflv_team_def", con, if_exists="replace", index=False)

    con.close()
    print(f"nflv_kicking: {len(kick):,} kicker-seasons")
    print(kick[kick.season==2025].nlargest(5,"custom_pts")[["name","team","games","custom_pts"]].to_string(index=False))
    print(f"\nnflv_team_def: {len(g):,} team-seasons")
    print(g[g.season==2025].nlargest(5,"custom_pts")[["team","games","sacks","ints","def_tds","custom_pts"]].to_string(index=False))


if __name__ == "__main__":
    main()
