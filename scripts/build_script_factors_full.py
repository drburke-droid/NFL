"""
Rebuild script_prediction_factors on the FULL window (2012-2025) without the
Odds-API/event_id dependency: P(game_script | O/U bucket) and P(game_script |
spread bucket), computed from the extended game_scripts table joined to historical
game lines (nflv_game_lines). Bucket labels match predict_explosions.py exactly.
"""
import os, sqlite3
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")


def ou_bucket(v):
    if pd.isna(v): return None
    if v <= 38: return "Low (≤38)"
    if v <= 42: return "Med-Low (38-42)"
    if v <= 46: return "Medium (42-46)"
    if v <= 50: return "Med-High (46-50)"
    return "High (50+)"


def sp_bucket(v):
    if pd.isna(v): return None
    v = abs(v)
    if v <= 2.5: return "Pickem (0-2.5)"
    if v <= 5: return "Close (3-5)"
    if v <= 8: return "Moderate (5.5-8)"
    if v <= 15: return "Big (8.5-15)"
    return "Huge (15+)"


def main():
    con = sqlite3.connect(DB)
    gs = pd.read_sql("SELECT game_id,season,week,home_team,away_team,game_script FROM game_scripts", con)
    gl = pd.read_sql("""SELECT season,week,team,game_total,team_spread FROM nflv_game_lines
                        WHERE game_type='REG'""", con)
    # attach each game's O/U + spread via the home team's line row
    g = gs.merge(gl, left_on=["season","week","home_team"], right_on=["season","week","team"], how="left")
    g = g.dropna(subset=["game_total","team_spread"])
    g["ou_bkt"] = g["game_total"].apply(ou_bucket)
    g["sp_bkt"] = g["team_spread"].apply(sp_bucket)
    print(f"games with lines: {len(g):,} / {len(gs):,}  ({int(g.season.min())}-{int(g.season.max())})")

    rows = []
    for ftype, col in [("over_under", "ou_bkt"), ("spread", "sp_bkt")]:
        for bkt, gg in g.groupby(col):
            n = len(gg)
            for script, c in gg["game_script"].value_counts().items():
                rows.append({"factor_type": ftype, "factor_value": bkt,
                             "game_script": script, "probability": c/n, "n": n})
    out = pd.DataFrame(rows)
    out.to_sql("script_prediction_factors", con, if_exists="replace", index=False)
    con.close()
    print(f"script_prediction_factors rebuilt: {len(out)} rows")
    print(out.groupby('factor_type')['factor_value'].nunique().to_dict())


if __name__ == "__main__":
    main()
