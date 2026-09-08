"""Build Model_Burke input frames from nfl_odds.db player props.

Per market: one row per player-week with
  actual_ppr   = the actual stat (model target — column name is the contract's)
  baseline_proj= consensus CLOSING line (median across books' last pre-game snapshot)
  over_price / under_price (median American odds at the consensus line)
plus lagged usage features, game context, wind.
"""
import sqlite3, os, re, sys
import numpy as np, pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
# argv[1] = path to the model_burke package dir (the folder CONTAINING model_burke/)
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_BURKE_PKG", "")
if not PKG or not os.path.isdir(PKG):
    raise SystemExit("pass the model_burke package dir as argv[1] "
                     "(unzip model_burke_pkg.zip and point at its pkg/ folder)")
sys.path.insert(0, PKG)
from model_burke.features import build_lagged_features

DB = r"C:\Users\drbur\Documents\GitHub\NFL\db\nfl_odds.db"
MARKETS = {
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_rush_yds": "rushing_yards",
    "player_pass_yds": "passing_yards",
}

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)

con = sqlite3.connect(DB)
games = pd.read_sql("SELECT event_id, commence_time, season, week, home_team, away_team FROM games", con)
games["season"] = pd.to_numeric(games.season, errors="coerce").astype("Int64")
games["week"] = pd.to_numeric(games.week, errors="coerce").astype("Int64")
games = games.dropna(subset=["season", "week"])
games["season"] = games.season.astype(int); games["week"] = games.week.astype(int)

# ---- closing game context (spread magnitude + total) ----
go = pd.read_sql("""SELECT event_id, bookmaker, market, outcome_name, point, snapshot_time
                    FROM game_odds WHERE market IN ('spreads','totals')""", con)
go = go.merge(games[["event_id", "commence_time", "home_team"]], on="event_id")
go = go[go.snapshot_time <= go.commence_time]
go["rk"] = go.groupby(["event_id", "bookmaker", "market", "outcome_name"]).snapshot_time.rank(
    ascending=False, method="first")
go = go[go.rk == 1]
tot = go[go.market == "totals"].groupby("event_id").point.median().rename("game_total")
spr_home = go[(go.market == "spreads") & (go.outcome_name == go.home_team)]
spr = spr_home.groupby("event_id").point.median().rename("home_spread")
ctx = pd.concat([tot, spr], axis=1).reset_index()
wind = pd.read_sql("SELECT event_id, wind_kn FROM nflv_game_wind", con)

# ---- weekly actuals + lagged features ----
wk = pd.read_sql("""SELECT * FROM nflv_weekly WHERE season BETWEEN 2022 AND 2025
                    AND position IN ('QB','RB','WR','TE')""", con)
wk = wk.drop_duplicates(["player_id", "season", "week"])
lag = build_lagged_features(wk)
wk["nname"] = wk.player_display_name.map(norm)

# map team names (games uses full names) -> weekly team codes via nothing needed:
# join props->games gives season/week; player row found by (nname, season, week).

for market, statcol in MARKETS.items():
    pp = pd.read_sql(f"""SELECT event_id, bookmaker, player_name, outcome_type, price, point, snapshot_time
                         FROM player_props WHERE market='{market}'""", con)
    pp = pp.merge(games, on="event_id")
    pp = pp[pp.snapshot_time <= pp.commence_time]
    pp["rk"] = pp.groupby(["event_id", "bookmaker", "player_name", "outcome_type"]).snapshot_time.rank(
        ascending=False, method="first")
    pp = pp[pp.rk == 1]
    over = pp[pp.outcome_type == "Over"]
    under = pp[pp.outcome_type == "Under"]
    line = over.groupby(["event_id", "player_name", "season", "week"]).point.median().rename("line")
    op = over.groupby(["event_id", "player_name", "season", "week"]).price.median().rename("over_price")
    up = under.groupby(["event_id", "player_name", "season", "week"]).price.median().rename("under_price")
    nb = over.groupby(["event_id", "player_name", "season", "week"]).bookmaker.nunique().rename("n_books")
    L = pd.concat([line, op, up, nb], axis=1).reset_index()
    L["nname"] = L.player_name.map(norm)

    d = L.merge(wk[["nname", "season", "week", "player_id", "player_display_name", "position",
                    "team", "opponent_team", statcol]],
                on=["nname", "season", "week"], how="left")
    match = d.player_id.notna().mean()
    d = d.dropna(subset=["player_id"])
    # DNP rows: played=0 stats exist only if a weekly row exists; keep (they count vs unders)
    d = d.rename(columns={statcol: "actual_ppr", "line": "baseline_proj",
                          "player_display_name": "player"})
    d = d.merge(lag, on=["player_id", "season", "week"], how="left")
    d = d.merge(ctx, on="event_id", how="left").merge(wind, on="event_id", how="left")
    d["is_outdoor"] = d.wind_kn.notna().astype(int)          # wind table = outdoor games
    d["wind_kn"] = d.wind_kn.fillna(0)
    d = d.merge(games[["event_id", "home_team"]], on="event_id", how="left")
    # spread from the player's team perspective is unknown w/o team-name map; use magnitude
    d["spread"] = d.home_spread
    d["implied_team_total"] = d.game_total / 2 - d.home_spread.fillna(0) / 2
    d = d.drop_duplicates(["player_id", "season", "week"])
    out = os.path.join(SCRATCH, f"props_{market}.parquet")
    d.to_parquet(out)
    print(f"{market}: {len(d):,} rows, match {match:.1%}, seasons "
          f"{d.groupby('season').size().to_dict()}, books/row median {d.n_books.median():.0f}")
con.close()
