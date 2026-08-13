"""
Ingest the LEAGUE-SCORED FantasyFootballAnalytics preseason (wk0) projections,
2014-2026 (full PPR + 6-pt pass TD / -1 INT — our league's actual settings),
from data/ffanalytics/FFAn_league/.

Writes nflv_ffa_league — a parallel table to nflv_ffa_proj (standard scoring).
Same name->gsis matching as fetch_ffa.py: same-season crosswalk against
nflv_season, with a latest-season fallback for seasons not yet played (2026).
2014-15 are sparse stubs in the source archive, as in the standard set.
"""
import os, glob, re, sqlite3
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
SRC = os.path.join(ROOT, "data", "ffanalytics", "FFAn_league")
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(s):
    s = str(s).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("-", " ").replace(",", "")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    con = sqlite3.connect(DB)
    seas = pd.read_sql("""SELECT player_id, player_display_name, position, season
                          FROM nflv_season WHERE position IN ('QB','RB','WR','TE')""", con)
    seas["nm"] = seas["player_display_name"].map(norm)
    xwalk = seas[["nm","position","season","player_id"]].drop_duplicates(["nm","position","season"])

    frames = []
    for f in sorted(glob.glob(os.path.join(SRC, "projections_*_wk0.csv"))):
        yr = int(re.search(r"projections_(\d{4})_wk0", f).group(1))
        d = pd.read_csv(f)
        d["season"] = yr
        frames.append(d)
    ffa = pd.concat(frames, ignore_index=True)
    ffa = ffa[ffa["position"].isin(["QB","RB","WR","TE"])].copy()
    ffa["nm"] = ffa["player"].map(norm)

    merged = ffa.merge(xwalk, on=["nm","position","season"], how="left")

    # Seasons with no nflv_season coverage yet (the upcoming season) can't match
    # same-season; fall back to each player's most recent name->id mapping.
    played = set(seas["season"].unique())
    latest = (seas.sort_values("season")
              .drop_duplicates(["nm","position"], keep="last")
              .set_index(["nm","position"])["player_id"])
    future = merged["player_id"].isna() & ~merged["season"].isin(played)
    merged.loc[future, "player_id"] = (
        merged.loc[future].set_index(["nm","position"]).index.map(latest).values)

    rate = merged["player_id"].notna().mean()

    out = merged.rename(columns={
        "points":"ffa_points","sd_pts":"ffa_sd","floor":"ffa_floor","ceiling":"ffa_ceiling",
        "points_vor":"ffa_vor","rank":"ffa_rank","position_rank":"ffa_pos_rank",
        "tier":"ffa_tier","adp":"ffa_adp","aav":"ffa_aav","uncertainty":"ffa_uncertainty",
        "dropoff":"ffa_dropoff","age":"ffa_age","experience":"ffa_exp",
    })[["season","player_id","player","position","team","ffa_points","ffa_sd","ffa_floor",
        "ffa_ceiling","ffa_vor","ffa_rank","ffa_pos_rank","ffa_tier","ffa_adp","ffa_aav",
        "ffa_uncertainty","ffa_dropoff","ffa_age","ffa_exp"]]
    # ---- age repair -------------------------------------------------------
    # FFA's historical exports carry each player's CURRENT age on EVERY season:
    # the 2019 and 2022 files both list Tom Brady at 48 and Travis Kelce at 36.
    # Passed through raw, that made 58% of historical player-seasons look 31+ and
    # silently wrecks any age-based study. The CSV age is only right for the
    # upcoming season, so overwrite it with the true per-season age wherever we
    # have one and keep the CSV value only where we don't.
    truth = pd.read_sql("SELECT season, player_id, age FROM season_dataset "
                        "WHERE age IS NOT NULL", con).drop_duplicates(["season", "player_id"])
    before = out["ffa_age"].copy()
    out = out.merge(truth, on=["season", "player_id"], how="left")
    fixed = out["age"].notna() & (out["age"] != before)

    # season_dataset has no row for a player's ROOKIE year, so those would keep the
    # bad CSV age (Justin Jefferson showed 27 in 2020, his age-21 season). Anchor on
    # any season we do know and walk the age back/forward a year at a time.
    anchor = (truth.sort_values("season").drop_duplicates("player_id", keep="last")
              .set_index("player_id")[["season", "age"]])
    need = out["age"].isna() & out["player_id"].notna()
    if need.any():
        a_s = out.loc[need, "player_id"].map(anchor["season"])
        a_a = out.loc[need, "player_id"].map(anchor["age"])
        out.loc[need, "age"] = a_a - (a_s - out.loc[need, "season"])
        print(f"age repair: extrapolated {int(out.loc[need,'age'].notna().sum()):,} "
              f"seasons from each player's own age anchor (rookie years)")
    out["ffa_age"] = out["age"].where(out["age"].notna(), out["ffa_age"])
    out = out.drop(columns=["age"])
    print(f"age repair: {int(fixed.sum()):,} player-seasons corrected from season_dataset "
          f"({out['ffa_age'].notna().sum():,} of {len(out):,} have an age)")

    out.to_sql("nflv_ffa_league", con, if_exists="replace", index=False)
    con.close()

    print(f"nflv_ffa_league: {len(out):,} rows, seasons {int(out.season.min())}-{int(out.season.max())} (league scoring: PPR + 6-pt pass TD)")
    print(f"gsis match rate: {rate:.1%}  (matched {out['player_id'].notna().sum():,})")
    print("rows per season:\n", out.groupby("season").size().to_string())
    print("\nunmatched sample (top by points):")
    um = out[out.player_id.isna()].nlargest(8,"ffa_points")[["season","player","position","ffa_points"]]
    print(um.to_string(index=False))


if __name__ == "__main__":
    main()
