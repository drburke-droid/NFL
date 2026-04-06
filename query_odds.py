"""
NFL Odds Database Query Helper
Quick access to game odds, player props, and more for DFS analysis.

Usage:
    from query_odds import OddsDB
    db = OddsDB()

    # Get all games for a season
    games = db.games(season=2024)

    # Get spreads for a specific game
    odds = db.game_odds(home_team="Kansas City Chiefs", season=2024, market="spreads")

    # Get player props
    props = db.player_props(player="Patrick Mahomes", season=2024)

    # Get all props for a specific game
    props = db.game_props(home_team="Kansas City Chiefs", away_team="Baltimore Ravens", season=2024)

    # Raw SQL
    rows = db.query("SELECT * FROM game_odds WHERE bookmaker='draftkings' LIMIT 10")
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_odds.db")


class OddsDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def query(self, sql, params=()):
        """Run raw SQL and return list of dicts."""
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def games(self, season=None, team=None):
        """Get games, optionally filtered by season and/or team."""
        sql = "SELECT * FROM games WHERE 1=1"
        params = []
        if season:
            sql += " AND season = ?"
            params.append(season)
        if team:
            sql += " AND (home_team LIKE ? OR away_team LIKE ?)"
            params.extend([f"%{team}%", f"%{team}%"])
        sql += " ORDER BY commence_time"
        return self.query(sql, params)

    def game_odds(self, event_id=None, home_team=None, away_team=None,
                  season=None, market=None, bookmaker=None):
        """Get game odds (h2h, spreads, totals) with filters."""
        sql = """
            SELECT g.home_team, g.away_team, g.commence_time, g.season,
                   o.bookmaker, o.market, o.outcome_name, o.price, o.point,
                   o.snapshot_time, o.last_update
            FROM game_odds o
            JOIN games g ON o.event_id = g.event_id
            WHERE 1=1
        """
        params = []
        if event_id:
            sql += " AND o.event_id = ?"
            params.append(event_id)
        if home_team:
            sql += " AND g.home_team LIKE ?"
            params.append(f"%{home_team}%")
        if away_team:
            sql += " AND g.away_team LIKE ?"
            params.append(f"%{away_team}%")
        if season:
            sql += " AND g.season = ?"
            params.append(season)
        if market:
            sql += " AND o.market = ?"
            params.append(market)
        if bookmaker:
            sql += " AND o.bookmaker = ?"
            params.append(bookmaker)
        sql += " ORDER BY g.commence_time, o.bookmaker, o.market"
        return self.query(sql, params)

    def player_props(self, player=None, market=None, season=None,
                     bookmaker=None, team=None):
        """Get player props with filters."""
        sql = """
            SELECT g.home_team, g.away_team, g.commence_time, g.season,
                   p.bookmaker, p.market, p.player_name, p.outcome_type,
                   p.price, p.point, p.snapshot_time
            FROM player_props p
            JOIN games g ON p.event_id = g.event_id
            WHERE 1=1
        """
        params = []
        if player:
            sql += " AND p.player_name LIKE ?"
            params.append(f"%{player}%")
        if market:
            sql += " AND p.market = ?"
            params.append(market)
        if season:
            sql += " AND g.season = ?"
            params.append(season)
        if bookmaker:
            sql += " AND p.bookmaker = ?"
            params.append(bookmaker)
        if team:
            sql += " AND (g.home_team LIKE ? OR g.away_team LIKE ?)"
            params.extend([f"%{team}%", f"%{team}%"])
        sql += " ORDER BY g.commence_time, p.player_name, p.market"
        return self.query(sql, params)

    def game_props(self, home_team=None, away_team=None, season=None, game_date=None):
        """Get all player props for a specific game."""
        sql = """
            SELECT g.home_team, g.away_team, g.commence_time,
                   p.bookmaker, p.market, p.player_name, p.outcome_type,
                   p.price, p.point
            FROM player_props p
            JOIN games g ON p.event_id = g.event_id
            WHERE 1=1
        """
        params = []
        if home_team:
            sql += " AND g.home_team LIKE ?"
            params.append(f"%{home_team}%")
        if away_team:
            sql += " AND g.away_team LIKE ?"
            params.append(f"%{away_team}%")
        if season:
            sql += " AND g.season = ?"
            params.append(season)
        if game_date:
            sql += " AND g.commence_time LIKE ?"
            params.append(f"{game_date}%")
        sql += " ORDER BY p.player_name, p.market, p.bookmaker"
        return self.query(sql, params)

    def teams(self):
        """List all unique teams."""
        rows = self.query("SELECT DISTINCT home_team FROM games ORDER BY home_team")
        return [r["home_team"] for r in rows]

    def markets(self):
        """List all available prop markets."""
        game_mkts = self.query("SELECT DISTINCT market FROM game_odds ORDER BY market")
        prop_mkts = self.query("SELECT DISTINCT market FROM player_props ORDER BY market")
        return {
            "game_markets": [r["market"] for r in game_mkts],
            "prop_markets": [r["market"] for r in prop_mkts],
        }

    def bookmakers(self):
        """List all bookmakers in the data."""
        bks = self.query("""
            SELECT DISTINCT bookmaker FROM (
                SELECT bookmaker FROM game_odds
                UNION
                SELECT bookmaker FROM player_props
            ) ORDER BY bookmaker
        """)
        return [r["bookmaker"] for r in bks]

    def players(self, season=None, team=None):
        """List all unique players in props data."""
        sql = """
            SELECT DISTINCT p.player_name
            FROM player_props p
            JOIN games g ON p.event_id = g.event_id
            WHERE p.player_name != ''
        """
        params = []
        if season:
            sql += " AND g.season = ?"
            params.append(season)
        if team:
            sql += " AND (g.home_team LIKE ? OR g.away_team LIKE ?)"
            params.extend([f"%{team}%", f"%{team}%"])
        sql += " ORDER BY p.player_name"
        return [r["player_name"] for r in self.query(sql, params)]

    def summary(self):
        """Print database summary."""
        games = self.query("SELECT COUNT(*) as n FROM games")[0]["n"]
        odds = self.query("SELECT COUNT(*) as n FROM game_odds")[0]["n"]
        props = self.query("SELECT COUNT(*) as n FROM player_props")[0]["n"]

        print(f"NFL Odds Database Summary")
        print(f"{'='*50}")
        print(f"Total games:        {games:,}")
        print(f"Game odds rows:     {odds:,}")
        print(f"Player prop rows:   {props:,}")
        print()

        for season in [2023, 2024, 2025]:
            sg = self.query("SELECT COUNT(*) as n FROM games WHERE season=?", (season,))[0]["n"]
            so = self.query(
                "SELECT COUNT(*) as n FROM game_odds WHERE event_id IN (SELECT event_id FROM games WHERE season=?)",
                (season,),
            )[0]["n"]
            sp = self.query(
                "SELECT COUNT(*) as n FROM player_props WHERE event_id IN (SELECT event_id FROM games WHERE season=?)",
                (season,),
            )[0]["n"]
            gp = self.query(
                "SELECT COUNT(DISTINCT pp.event_id) as n FROM player_props pp JOIN games g ON pp.event_id=g.event_id WHERE g.season=?",
                (season,),
            )[0]["n"]
            print(f"{season} Season:")
            print(f"  Games: {sg}  |  Odds rows: {so:,}  |  Props: {sp:,} ({gp} games)")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = OddsDB()
    db.summary()

    print(f"\nBookmakers: {', '.join(db.bookmakers())}")
    print(f"\nMarkets: {db.markets()}")

    # Example queries
    print(f"\n{'='*50}")
    print("Example: Patrick Mahomes pass yards props (2024)")
    print(f"{'='*50}")
    props = db.player_props(player="Patrick Mahomes", market="player_pass_yds", season=2024, bookmaker="draftkings")
    for p in props[:5]:
        print(f"  {p['commence_time'][:10]} vs {p['away_team'] if 'Kansas City' in p['home_team'] else p['home_team']}: "
              f"{p['outcome_type']} {p['point']} @ {p['price']}")

    db.close()
