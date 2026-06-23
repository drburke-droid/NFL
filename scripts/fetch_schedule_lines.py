"""
Historical game lines from nflverse schedules (free, 1999-2025).

Closing spread/total/moneyline with 100% coverage back to 1999. Builds a per-team
long table with a DIRECTIONAL implied team total so the weekly/explosion models can
use game environment across the full history (not just the 3 Odds-API seasons).

spread_line convention (verified): positive => HOME favored.
team_spread here: negative => that team favored (matches the Odds-API code path).
implied_team_total = total/2 - team_spread/2.

Writes nflv_game_lines.
"""
import os, sqlite3
import nflreadpy as nflr
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")


def main():
    s = nflr.load_schedules().to_pandas()
    s = s.dropna(subset=["spread_line", "total_line"]).copy()

    # normalize relocated franchises to nflv_weekly's current codes
    ALIAS = {"OAK":"LV","SD":"LAC","SDG":"LAC","STL":"LA","LAR":"LA","JAC":"JAX"}
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(ALIAS)

    home = pd.DataFrame({
        "season": s["season"], "week": s["week"], "game_type": s["game_type"],
        "team": s["home_team"], "opp": s["away_team"], "is_home": 1,
        "game_total": s["total_line"], "team_spread": -s["spread_line"],
        "implied_team_total": s["total_line"]/2 + s["spread_line"]/2,
        "team_moneyline": s["home_moneyline"],
    })
    away = pd.DataFrame({
        "season": s["season"], "week": s["week"], "game_type": s["game_type"],
        "team": s["away_team"], "opp": s["home_team"], "is_home": 0,
        "game_total": s["total_line"], "team_spread": s["spread_line"],
        "implied_team_total": s["total_line"]/2 - s["spread_line"]/2,
        "team_moneyline": s["away_moneyline"],
    })
    gl = pd.concat([home, away], ignore_index=True)

    con = sqlite3.connect(DB)
    gl.to_sql("nflv_game_lines", con, if_exists="replace", index=False)

    # join-coverage check vs nflv_weekly (the model spine)
    wk = pd.read_sql("SELECT DISTINCT season, week, team FROM nflv_weekly "
                     "WHERE season>=2012 AND season_type='REG'", con)
    m = wk.merge(gl[gl.game_type == "REG"][["season","week","team"]].drop_duplicates(),
                 on=["season","week","team"], how="left", indicator=True)
    cov = (m["_merge"] == "both").mean()
    con.close()

    print(f"nflv_game_lines: {len(gl):,} team-rows, seasons {int(gl.season.min())}-{int(gl.season.max())}")
    print(f"join coverage vs nflv_weekly (2012+ REG): {cov:.1%}")
    if cov < 0.99:
        miss = m[m["_merge"]=="left_only"]["team"].value_counts().head()
        print("  unmatched team codes (sample):", miss.to_dict())


if __name__ == "__main__":
    main()
