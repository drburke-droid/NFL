"""
Historical preseason ADP/ECR ingestion.

nflverse load_ff_rankings('all') archives FantasyPros consensus rankings with
scrape dates back to 2021. We take the LAST August scrape each year (final
preseason consensus, before Week 1) from page_type='redraft-overall' as that
season's market draft ranking (ECR ~= ADP), map to gsis_id, store nflv_adp.
"""
import os, sqlite3
import nflreadpy as nflr
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")


def main():
    d = nflr.load_ff_rankings("all").to_pandas()
    d = d[d["page_type"] == "redraft-overall"].copy()
    d["scrape_date"] = pd.to_datetime(d["scrape_date"], errors="coerce")
    d["yr"] = d["scrape_date"].dt.year
    d["mo"] = d["scrape_date"].dt.month
    aug = d[d["mo"] == 8].copy()

    # last August scrape per season = final preseason consensus
    rows = []
    for yr, g in aug.groupby("yr"):
        last = g["scrape_date"].max()
        snap = g[g["scrape_date"] == last].copy()
        snap = snap.sort_values("ecr")
        snap["season"] = int(yr)
        snap["adp_overall"] = range(1, len(snap) + 1)  # dense overall rank by ecr
        snap["pos_rank"] = snap.groupby("pos")["ecr"].rank(method="first")
        rows.append(snap[["season", "player", "id", "pos", "team", "ecr", "sd",
                          "adp_overall", "pos_rank"]])
    adp = pd.concat(rows, ignore_index=True)

    # map fantasypros id -> gsis_id
    try:
        ids = nflr.load_ff_playerids().to_pandas()
        idcol = "fantasypros_id" if "fantasypros_id" in ids.columns else \
                ("fantasypros" if "fantasypros" in ids.columns else None)
        if idcol and "gsis_id" in ids.columns:
            xwalk = ids[[idcol, "gsis_id"]].dropna().drop_duplicates(idcol)
            xwalk[idcol] = xwalk[idcol].astype(str)
            adp["id"] = adp["id"].astype(str)
            adp = adp.merge(xwalk, left_on="id", right_on=idcol, how="left")
            adp = adp.rename(columns={"gsis_id": "player_id"})
        else:
            adp["player_id"] = None
            print("  !! ff_playerids missing expected columns:", list(ids.columns)[:10])
    except Exception as e:
        print("  !! id crosswalk failed:", e)
        adp["player_id"] = None

    matched = adp["player_id"].notna().mean() if "player_id" in adp else 0
    con = sqlite3.connect(DB)
    adp.to_sql("nflv_adp", con, if_exists="replace", index=False)
    con.close()
    print(f"nflv_adp: {len(adp):,} rows, seasons {sorted(adp['season'].unique())}")
    print(f"gsis_id match rate: {matched:.1%}")
    print(adp.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
