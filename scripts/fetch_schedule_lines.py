"""
Game lines from nflverse schedules (free, 1999-present).

Closing spread/total/moneyline with 100% coverage back to 1999. Builds a per-team
long table with a DIRECTIONAL implied team total so the weekly/explosion models can
use game environment across the full history (not just the 3 Odds-API seasons).

spread_line convention (verified): positive => HOME favored.
team_spread here: negative => that team favored (matches the Odds-API code path).
implied_team_total = total/2 - team_spread/2.

Settled seasons are complete, but the CURRENT season's spread_line / total_line is a
LIVE rolling field: nflverse carries a line only while a book has the game on the
board. A fresh pull covers the next few weeks and nothing beyond, and a game that
comes off the board vanishes from it. Dropping unpriced games and then REPLACING the
table threw away every current-season game the latest pull happened not to carry —
that is how the committed parquets came to hold 78 of 272 games for 2026, with no
DET @ BUF or CIN @ HOU in week 2, which reached the week-2 sends as a blank Opp on
the K and DST rows. So this UPSERTS: the pull refreshes the games it carries and
leaves every other row of the table alone. --replace restores the old behaviour.

Writes nflv_game_lines; --export-parquet also refreshes the committed copies under
data/sabersim/ that sabersim_weekly.py and the studies actually read.
"""
import argparse, os, sqlite3
import nflreadpy as nflr
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
TABLE = "nflv_game_lines"
KEY = ["season", "week", "game_type", "team"]
# the committed exports the model code reads, each holding the seasons in its name
PARQUETS = {"game_lines_2015_2026.parquet": 2015, "game_lines_2023_2026.parquet": 2023}
PQ_COLS = ["season", "week", "team", "opp", "is_home", "game_total", "team_spread", "implied_team_total"]
# relocated franchises, normalized to nflv_weekly's current codes
ALIAS = {"OAK": "LV", "SD": "LAC", "SDG": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}


def build(s):
    """nflverse schedules -> the per-team long table. Unpriced games are dropped."""
    s = s.dropna(subset=["spread_line", "total_line"]).copy()
    for c in ("home_team", "away_team"):
        s[c] = s[c].replace(ALIAS)
    home = pd.DataFrame({
        "season": s["season"], "week": s["week"], "game_type": s["game_type"],
        "team": s["home_team"], "opp": s["away_team"], "is_home": 1,
        "game_total": s["total_line"], "team_spread": -s["spread_line"],
        "implied_team_total": s["total_line"] / 2 + s["spread_line"] / 2,
        "team_moneyline": s["home_moneyline"],
    })
    away = pd.DataFrame({
        "season": s["season"], "week": s["week"], "game_type": s["game_type"],
        "team": s["away_team"], "opp": s["home_team"], "is_home": 0,
        "game_total": s["total_line"], "team_spread": s["spread_line"],
        "implied_team_total": s["total_line"] / 2 - s["spread_line"] / 2,
        "team_moneyline": s["away_moneyline"],
    })
    return pd.concat([home, away], ignore_index=True)


def keys_of(df):
    return list(zip(df.season.astype(int), df.week.astype(int),
                    df.game_type.astype(str), df.team.astype(str)))


def upsert(con, fresh):
    """Fresh rows win on the key; rows this pull no longer carries are kept as they are."""
    try:
        old = pd.read_sql(f"SELECT * FROM {TABLE}", con)
    except Exception:
        return fresh, 0                                  # first run, no table yet
    if old.empty:
        return fresh, 0
    absent = [c for c in fresh.columns if c not in old.columns]
    if absent:                                           # different shape: do not guess, rebuild
        print(f"  existing table has no {absent} column(s) — rebuilding from this pull alone")
        return fresh, 0
    seen = set(keys_of(fresh))
    keep = old[[k not in seen for k in keys_of(old)]]
    return pd.concat([fresh, keep[fresh.columns]], ignore_index=True), len(keep)


def games_by_week(gl, season):
    g = gl[gl.season == season]
    return {int(w): int(n // 2) for w, n in g.groupby("week").size().items()}


def export_parquets(gl, force):
    for name, start in PARQUETS.items():
        fp = os.path.join(ROOT, "data", "sabersim", name)
        part = gl[gl.season >= start][PQ_COLS].sort_values(KEY[:2] + ["team"]).reset_index(drop=True)
        if os.path.exists(fp):
            prev = len(pd.read_parquet(fp))
            if len(part) < prev and not force:
                print(f"  {name}: {len(part):,} rows < the {prev:,} already there — NOT written "
                      f"(the pull lost games; pass --force to overwrite anyway)")
                continue
        part.to_parquet(fp, index=False)
        print(f"  wrote {name}: {len(part):,} team-rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replace", action="store_true",
                    help="rebuild the table from this pull alone instead of upserting (drops rows the pull lacks)")
    ap.add_argument("--export-parquet", action="store_true",
                    help="also refresh the committed data/sabersim/game_lines_*.parquet exports")
    ap.add_argument("--force", action="store_true", help="let --export-parquet write a smaller file than the one on disk")
    A = ap.parse_args()

    fresh = build(nflr.load_schedules().to_pandas())
    cur = int(fresh.season.max())
    print(f"pull: {len(fresh) // 2:,} priced games, seasons {int(fresh.season.min())}-{cur}"
          f" · {cur} games per week {games_by_week(fresh, cur)}")

    con = sqlite3.connect(DB)
    if A.replace:
        gl, kept = fresh, 0
        print("  --replace: rebuilding from this pull alone")
    else:
        gl, kept = upsert(con, fresh)
        print(f"  kept {kept:,} team-rows the pull no longer carries")
    gl.to_sql(TABLE, con, if_exists="replace", index=False)

    after = games_by_week(gl, cur)
    print(f"{TABLE}: {len(gl):,} team-rows, seasons {int(gl.season.min())}-{int(gl.season.max())}")
    print(f"  {cur} games per week after merge {after}  ({sum(after.values())} games)")

    # join-coverage check vs nflv_weekly (the model spine)
    wk = pd.read_sql("SELECT DISTINCT season, week, team FROM nflv_weekly "
                     "WHERE season>=2012 AND season_type='REG'", con)
    m = wk.merge(gl[gl.game_type == "REG"][["season", "week", "team"]].drop_duplicates(),
                 on=["season", "week", "team"], how="left", indicator=True)
    cov = (m["_merge"] == "both").mean()
    con.close()
    print(f"  join coverage vs nflv_weekly (2012+ REG): {cov:.1%}")
    if cov < 0.99:
        miss = m[m["_merge"] == "left_only"]["team"].value_counts().head()
        print("  unmatched team codes (sample):", miss.to_dict())

    if A.export_parquet:
        export_parquets(gl, A.force)


if __name__ == "__main__":
    main()
