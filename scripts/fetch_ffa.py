"""
Ingest FantasyFootballAnalytics preseason (wk0) season projections, 2012-2025.

These are weighted multi-expert consensus projections: projected points, VOR,
floor/ceiling, rank, tier, ADP, projection uncertainty. Mapped to our gsis
player_id by normalized name + season + position against nflv_season.

Writes nflv_ffa_proj.
"""
import os, glob, re, sqlite3
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ffanalytics", "FFAn")
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
    rate = merged["player_id"].notna().mean()

    out = merged.rename(columns={
        "points":"ffa_points","sd_pts":"ffa_sd","floor":"ffa_floor","ceiling":"ffa_ceiling",
        "points_vor":"ffa_vor","rank":"ffa_rank","position_rank":"ffa_pos_rank",
        "tier":"ffa_tier","adp":"ffa_adp","aav":"ffa_aav","uncertainty":"ffa_uncertainty",
        "dropoff":"ffa_dropoff","age":"ffa_age","experience":"ffa_exp",
    })[["season","player_id","player","position","ffa_points","ffa_sd","ffa_floor","ffa_ceiling",
        "ffa_vor","ffa_rank","ffa_pos_rank","ffa_tier","ffa_adp","ffa_aav","ffa_uncertainty",
        "ffa_dropoff","ffa_age","ffa_exp"]]
    out.to_sql("nflv_ffa_proj", con, if_exists="replace", index=False)
    con.close()

    print(f"nflv_ffa_proj: {len(out):,} rows, seasons {int(out.season.min())}-{int(out.season.max())}")
    print(f"gsis match rate: {rate:.1%}  (matched {out['player_id'].notna().sum():,})")
    print("rows per season:\n", out.groupby("season").size().to_string())
    print("\nunmatched sample (top by points):")
    um = out[out.player_id.isna()].nlargest(8,"ffa_points")[["season","player","position","ffa_points"]]
    print(um.to_string(index=False))


if __name__ == "__main__":
    main()
