"""
College production ingest (CollegeFootballData ecosystem, keyless).

Source: sportsdataverse/cfbfastR-data play-level player stats parquets, 2014-2025:
  https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/player_stats/parquet/player_stats_{yr}.parquet
(No API key needed — this is the same public data the CFBD API serves.)

Aggregates to player-season production and DOMINATOR shares (player's share of
his team's receiving / total scrimmage output — the classic prospect signal):

  nflv_college_prod: season, team, conference, player, espn_id,
    receptions, rec_yds, rec_td, carries, rush_yds, rush_td,
    team_rec_yds, team_scrim_yds, dom_rec (rec share), dom_scrim (total share)

Re-run each spring after the season finalizes.
"""
import io, os, sqlite3, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/player_stats/parquet/player_stats_{yr}.parquet"
YEARS = range(2014, 2026)


def one_year(yr):
    req = urllib.request.Request(BASE.format(yr=yr), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        df = pd.read_parquet(io.BytesIO(r.read()))
    rec = df.dropna(subset=["reception_player_id"]).groupby(
        ["season", "team", "conference", "reception_player_id", "reception_player"]).agg(
        receptions=("reception_yds", "count"), rec_yds=("reception_yds", "sum")).reset_index() \
        .rename(columns={"reception_player_id": "espn_id", "reception_player": "player"})
    rectd = df[df.touchdown_player_id.notna() & (df.touchdown_player_id == df.reception_player_id)] \
        .groupby(["season", "team", "reception_player_id"]).size().rename("rec_td").reset_index() \
        .rename(columns={"reception_player_id": "espn_id"})
    rush = df.dropna(subset=["rush_player_id"]).groupby(
        ["season", "team", "conference", "rush_player_id", "rush_player"]).agg(
        carries=("rush_yds", "count"), rush_yds=("rush_yds", "sum")).reset_index() \
        .rename(columns={"rush_player_id": "espn_id", "rush_player": "player"})
    rushtd = df[df.touchdown_player_id.notna() & (df.touchdown_player_id == df.rush_player_id)] \
        .groupby(["season", "team", "rush_player_id"]).size().rename("rush_td").reset_index() \
        .rename(columns={"rush_player_id": "espn_id"})
    m = rec.merge(rectd, on=["season", "team", "espn_id"], how="outer") \
           .merge(rush, on=["season", "team", "conference", "espn_id", "player"], how="outer") \
           .merge(rushtd, on=["season", "team", "espn_id"], how="outer")
    for c in ["receptions", "rec_yds", "rec_td", "carries", "rush_yds", "rush_td"]:
        m[c] = m[c].fillna(0)
    tm = m.groupby(["season", "team"]).agg(team_rec_yds=("rec_yds", "sum"),
                                           team_rush_yds=("rush_yds", "sum"),
                                           team_rec_td=("rec_td", "sum"),
                                           team_rush_td=("rush_td", "sum")).reset_index()
    m = m.merge(tm, on=["season", "team"])
    m["team_scrim_yds"] = m.team_rec_yds + m.team_rush_yds
    m["dom_rec"] = np.where(m.team_rec_yds > 0,
                            0.5 * (m.rec_yds / m.team_rec_yds.clip(lower=1))
                            + 0.5 * (m.rec_td / m.team_rec_td.clip(lower=1)), 0)
    m["dom_scrim"] = (m.rec_yds + m.rush_yds) / m.team_scrim_yds.clip(lower=1)
    keep = ["season", "team", "conference", "espn_id", "player", "receptions", "rec_yds", "rec_td",
            "carries", "rush_yds", "rush_td", "team_rec_yds", "team_scrim_yds", "dom_rec", "dom_scrim"]
    return m[keep]


def main():
    frames = []
    for yr in YEARS:
        try:
            f = one_year(yr)
            frames.append(f)
            print(f"{yr}: {len(f):,} player-seasons")
        except Exception as e:
            print(f"{yr}: FAILED {e}")
    allf = pd.concat(frames, ignore_index=True)
    con = sqlite3.connect(DB)
    allf.to_sql("nflv_college_prod", con, if_exists="replace", index=False)
    con.close()
    print(f"\nnflv_college_prod: {len(allf):,} rows, seasons {allf.season.min()}-{allf.season.max()}")


if __name__ == "__main__":
    main()
