"""
Ingest nflverse/OverTheCap historical contracts (spotrac-style salary data).

Source: https://github.com/nflverse/nflverse-data/releases/tag/contracts
Writes nflv_contracts: one row per contract with player gsis_id, position,
team, year_signed, years, total value, APY, guarantees, and APY as % of the
signing-year salary cap (the era-neutral "how paid is he" number).
"""
import io, os, sqlite3, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
# NOTE: use the .parquet asset — it's the live one (updated daily); the .csv.gz
# on the same release is a stale 2022 snapshot.
URL = "https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet"


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    df = pd.read_parquet(io.BytesIO(raw))
    keep = [c for c in ["player", "position", "team", "is_active", "year_signed", "years",
                        "value", "apy", "guaranteed", "apy_cap_pct", "inflated_value",
                        "inflated_apy", "inflated_guaranteed", "player_page", "otc_id",
                        "gsis_id", "date_of_birth", "draft_year", "draft_round"] if c in df.columns]
    df = df[keep]
    con = sqlite3.connect(DB)
    df.to_sql("nflv_contracts", con, if_exists="replace", index=False)
    n_rb = con.execute("SELECT COUNT(*) FROM nflv_contracts WHERE position='RB'").fetchone()[0]
    con.close()
    print(f"nflv_contracts: {len(df):,} contracts ({n_rb:,} RB), cols: {keep}")
    rb = df[(df.position == "RB")].sort_values("year_signed", ascending=False)
    print(rb[["player", "team", "year_signed", "years", "apy", "guaranteed", "apy_cap_pct"]]
          .head(8).to_string(index=False))


if __name__ == "__main__":
    main()
