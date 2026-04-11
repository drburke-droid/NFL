"""Database helpers — connection management and common queries."""

from __future__ import annotations
import sqlite3
import pandas as pd
from typing import Optional
from src.utils.config import get_db_path


def get_conn(cfg: dict | None = None) -> sqlite3.Connection:
    return sqlite3.connect(get_db_path(cfg))


def read_sql(query: str, conn: sqlite3.Connection, params=None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def load_player_stats(conn: sqlite3.Connection,
                      positions: list[str] | None = None,
                      max_season: int | None = None,
                      max_week: int | None = None,
                      max_season_week: tuple[int, int] | None = None) -> pd.DataFrame:
    """Load player stats with optional time cutoff (strict < for walk-forward)."""
    where = ["1=1"]
    params: list = []

    if positions:
        placeholders = ",".join(["?"] * len(positions))
        where.append(f"ps.position IN ({placeholders})")
        params.extend(positions)

    if max_season_week:
        where.append("(ps.season * 100 + ps.week) < ?")
        params.append(max_season_week[0] * 100 + max_season_week[1])
    elif max_season is not None and max_week is not None:
        where.append("(ps.season * 100 + ps.week) < ?")
        params.append(max_season * 100 + max_week)

    sql = f"""
        SELECT ps.player_id, ps.player_display_name as name, ps.position,
               ps.team, ps.season, ps.week, ps.opponent,
               ps.completions, ps.attempts, ps.passing_yards, ps.passing_tds,
               ps.interceptions, ps.sacks, ps.passing_epa,
               ps.carries, ps.rushing_yards, ps.rushing_tds, ps.rushing_epa,
               ps.targets, ps.receptions, ps.receiving_yards, ps.receiving_tds,
               ps.receiving_epa, ps.target_share, ps.air_yards_share,
               ps.fantasy_points_ppr as ppr, ps.event_id
        FROM player_stats ps
        WHERE {' AND '.join(where)}
        ORDER BY ps.player_id, ps.season, ps.week
    """
    return pd.read_sql_query(sql, conn, params=params)


def load_game_odds(conn: sqlite3.Connection, bookmaker: str = "draftkings") -> dict:
    """Load spread, total, and props indexed by event_id."""
    spreads = pd.read_sql_query(f"""
        SELECT DISTINCT event_id, MIN(ABS(point)) as abs_spread
        FROM game_odds WHERE market='spreads' AND bookmaker=?
        GROUP BY event_id
    """, conn, params=[bookmaker])

    totals = pd.read_sql_query(f"""
        SELECT DISTINCT event_id, point as over_under
        FROM game_odds WHERE market='totals' AND outcome_name='Over' AND bookmaker=?
    """, conn, params=[bookmaker])
    totals = totals.drop_duplicates(subset="event_id", keep="first")

    props_raw = pd.read_sql_query(f"""
        SELECT event_id, player_name, market, point as prop_line
        FROM player_props WHERE bookmaker=? AND outcome_type='Over'
    """, conn, params=[bookmaker])

    props_wide = pd.DataFrame()
    if len(props_raw) > 0:
        props_wide = props_raw.pivot_table(
            index=["event_id", "player_name"], columns="market",
            values="prop_line", aggfunc="first"
        ).reset_index()
        props_wide.columns.name = None
        rename = {
            "player_pass_yds": "prop_pass_yds", "player_pass_tds": "prop_pass_tds",
            "player_rush_yds": "prop_rush_yds", "player_reception_yds": "prop_rec_yds",
            "player_receptions": "prop_receptions",
            "player_rush_reception_yds": "prop_rush_rec_yds",
            "player_anytime_td": "prop_anytime_td",
            "player_pass_attempts": "prop_pass_att",
            "player_pass_completions": "prop_pass_comp",
        }
        props_wide = props_wide.rename(columns=rename)

    return {"spreads": spreads, "totals": totals, "props": props_wide}


def load_game_event_lookup(conn: sqlite3.Connection) -> pd.DataFrame:
    """Map team+season+week to event_id and home/away."""
    games = pd.read_sql_query("""
        SELECT game_id, event_id, season, week, home_team, away_team
        FROM game_scripts
    """, conn)

    home = games[["event_id", "season", "week", "home_team", "away_team"]].copy()
    home = home.rename(columns={"home_team": "team"})
    home["is_home"] = 1

    away = games[["event_id", "season", "week", "away_team", "home_team"]].copy()
    away = away.rename(columns={"away_team": "team"})
    away["is_home"] = 0

    return pd.concat([home, away], ignore_index=True)
