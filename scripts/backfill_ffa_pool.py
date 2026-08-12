#!/usr/bin/env python3
"""Guarantee every FFA-projected player appears on the 2026 board.

FFA is refreshed with current news, so it is our freshest view of who is
actually relevant. The model pool (`board_2026`) is built from 2025 production
plus the 2026 rookie class, so it silently drops anyone who has neither --
players returning from a missed season (Jonathon Brooks, Tank Dell), the
unsigned (Joe Mixon), and veterans who sat out (Deshaun Watson). Meanwhile the
pool still carries long-retired names, so this is a coverage gap, not a
deliberate filter.

This runs AFTER build_2026_targets.py (which does `if_exists="replace"`, so any
backfill must happen downstream of it) and BEFORE report_2026_targets.py and
build_draft_tool.py, so both the report and the draft tool see the full pool.

Backfilled rows carry src='ffa': their projection is the FFA number passed
through, NOT a model output. build_draft_tool.py surfaces that as conf='ffa'
so the board never implies more confidence than we have.

    python scripts/backfill_ffa_pool.py
"""

import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
FFA_CSV = os.path.join(ROOT, "data", "ffanalytics", "FFAn_league",
                       "projections_2026_wk0.csv")
SEASON = 2026
SKILL = ("QB", "RB", "WR", "TE")

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)


def norm(n):
    n = str(n or "").lower().replace("'", "").replace(".", "").replace("-", " ")
    n = SUFFIX.sub("", n)
    return re.sub(r"[^a-z ]", "", n).strip()


def initial_last(n):
    """'Nathaniel Dell' -> 'n dell'; catches nickname swaps (Tank/Nathaniel Dell)."""
    p = norm(n).split()
    return f"{p[0][0]} {p[-1]}" if len(p) >= 2 else ""


def resolve_ids(names_positions, con):
    """Map (name, position) -> player_id using each player's most recent season."""
    seas = pd.read_sql(
        """SELECT player_id, player_display_name nm, position, MAX(season) ms
           FROM nflv_season GROUP BY player_id, player_display_name, position""", con)
    seas = seas.sort_values("ms").drop_duplicates(["player_id"], keep="last")
    exact, initl, last = {}, {}, {}
    for r in seas.itertuples(index=False):
        exact.setdefault((norm(r.nm), r.position), r.player_id)
        il = initial_last(r.nm)
        if il:
            initl.setdefault((il, r.position), []).append(r.player_id)
        parts = norm(r.nm).split()
        if parts:
            last.setdefault((parts[-1], r.position), []).append((r.player_id, r.nm))
    out = {}
    for nm, pos in names_positions:
        pid = exact.get((norm(nm), pos))
        if pid is None:
            c = initl.get((initial_last(nm), pos), [])
            pid = c[0] if len(c) == 1 else None   # only trust unambiguous hits
        if pid is None:
            # Nickname swaps change the first initial too (Tank vs Nathaniel Dell),
            # so fall back to surname + position -- but only when it is unique.
            parts = norm(nm).split()
            c = last.get((parts[-1], pos), []) if parts else []
            if len(c) == 1:
                pid = c[0][0]
                print(f"  [alias] '{nm}' -> '{c[0][1]}' ({pos}) id={pid}")
        out[(nm, pos)] = pid
    return out


def synth_id(name):
    """Stable stand-in id for a player nflv_season has never seen (2026 rookies).

    MUST be derived from the name alone so every table that mints one lands on the
    same value and merges line up.
    """
    return "NOID-" + re.sub(r"\s+", "-", norm(name))


def normalize_blank_ids(con):
    """Give every blank player_id a unique stand-in.

    pandas merges MATCH on NaN (SQL does not), so rows sharing a blank player_id
    cross-join. build_draft_tool.py does `drop_duplicates("player_id")` on
    nflv_late_breakout, which collapsed 6 rookies' distinct p_hit values down to
    one and then merged that single value onto all of them -- e.g. Deion Burks
    carried Stribling's 0.0355 instead of his own 0.0099, a 3.6x overstatement of
    a signal that drives the $1-3 lottery tags. Naming them apart fixes the join
    at the source, so real values attach to the right player and genuinely unknown
    ones stay NULL instead of borrowing a neighbour's.
    """
    cur = con.cursor()
    for table, namecol in (("board_2026", "player_display_name"),
                           ("nflv_late_breakout", "name")):
        cols = [r[1] for r in cur.execute(f"pragma table_info({table})")]
        if namecol not in cols or "player_id" not in cols:
            continue
        rows = cur.execute(
            f"SELECT DISTINCT {namecol} FROM {table} "
            f"WHERE player_id IS NULL OR TRIM(player_id)=''").fetchall()
        for (nm,) in rows:
            if not nm:
                continue
            cur.execute(
                f"UPDATE {table} SET player_id=? "
                f"WHERE {namecol}=? AND (player_id IS NULL OR TRIM(player_id)='')",
                (synth_id(nm), nm))
        if rows:
            print(f"  [ids] {table}: assigned stand-in ids to "
                  f"{len(rows)} name(s) with blank player_id")
    con.commit()


def main():
    con = sqlite3.connect(DB)
    normalize_blank_ids(con)
    board = pd.read_sql("SELECT * FROM board_2026", con)
    if "src" in board.columns:            # idempotent: strip a prior backfill
        board = board[board["src"] != "ffa"].drop(columns=["src"])
    board_n = {(norm(r.player_display_name), r.position) for r in board.itertuples(index=False)}
    board_il = {(initial_last(r.player_display_name), r.position) for r in board.itertuples(index=False)}

    ffa = pd.read_sql(
        f"""SELECT player_id, player, position, team, ffa_points, ffa_floor,
                   ffa_ceiling, ffa_age FROM nflv_ffa_league WHERE season={SEASON}""", con)
    ffa = ffa[ffa["position"].isin(SKILL) & ffa["ffa_points"].notna()]

    def present(r):
        return ((norm(r.player), r.position) in board_n
                or (initial_last(r.player), r.position) in board_il)

    miss = ffa[~ffa.apply(present, axis=1)].copy()
    print(f"FFA skill players: {len(ffa)} | already on board: {len(ffa) - len(miss)} "
          f"| backfilling: {len(miss)}")

    if not miss.empty:
        ids = resolve_ids(list(zip(miss["player"], miss["position"])), con)
        # NB: a SQL NULL arrives as float('nan'), which is TRUTHY -- test it
        # explicitly or an `or` chain short-circuits on the NaN.
        def pick(r):
            if pd.notna(r.player_id) and str(r.player_id).strip():
                return str(r.player_id)
            got = ids.get((r.player, r.position))
            if got:
                return got
            return synth_id(r.player)
        miss["player_id"] = [pick(r) for r in miss.itertuples(index=False)]

        # Same constants build_draft_tool.py uses, so the QB inversion round-trips.
        s25 = pd.read_sql(
            """SELECT player_id, games, passing_tds, passing_interceptions
               FROM nflv_season WHERE season=2025""", con).drop_duplicates("player_id")
        bm = board.merge(s25, on="player_id", how="left")
        isqb = bm["position"] == "QB"
        rate_td = float((bm["passing_tds"] / bm["games"].clip(lower=1))[isqb].median())
        rate_int = float((bm["passing_interceptions"] / bm["games"].clip(lower=1))[isqb].median())
        g_med = board.groupby("position")["proj_games"].median().to_dict()

        # Players with real 2025 usage keep their own rates; the rest take the median,
        # exactly as build_draft_tool.py's fillna does.
        own = s25.set_index("player_id")
        rows = []
        for r in miss.itertuples(index=False):
            G = float(g_med.get(r.position, 13.0))
            pts = float(r.ffa_points)
            if r.position == "QB":
                rt, ri = rate_td, rate_int
                if r.player_id in own.index:
                    o = own.loc[r.player_id]
                    gm = max(float(o["games"] or 0), 1.0)
                    if o["passing_tds"] is not None and float(o["games"] or 0) > 0:
                        rt = float(o["passing_tds"]) / gm
                        ri = float(o["passing_interceptions"] or 0) / gm
                ppg = pts / G - 2 * rt - ri
            else:
                ppg = pts / G
            rows.append({
                "player_id": r.player_id,
                "player_display_name": r.player,
                "position": r.position,
                "team": r.team,
                "age": r.ffa_age,
                "prior_ppg": np.nan, "prior_cv": np.nan, "prior_games": np.nan,
                "pred_ppg": ppg,
                "floor": (float(r.ffa_floor) / G) if pd.notna(r.ffa_floor) else np.nan,
                "ceiling": (float(r.ffa_ceiling) / G) if pd.notna(r.ffa_ceiling) else np.nan,
                "bust": np.nan, "boom": np.nan,
                "breakout_prob": np.nan, "hit_prob": np.nan,
                "proj_games": G,
                "is_rookie": 0,
                "src": "ffa",
            })
            print(f"  + {r.position:<3} {r.player:<24} FFA {pts:>6.1f} pts "
                  f"-> pred_ppg {ppg:5.2f} over {G:.1f} g   id={r.player_id}")
        add = pd.DataFrame(rows)
    else:
        add = pd.DataFrame(columns=list(board.columns) + ["src"])

    board["src"] = "model"
    out = pd.concat([board, add], ignore_index=True)
    out.to_sql("board_2026", con, if_exists="replace", index=False)
    print(f"board_2026: {len(board)} model + {len(add)} ffa = {len(out)} rows")

    # ---- kickers: not ingested into nflv_ffa_league, so read them from the CSV ----
    if os.path.exists(FFA_CSV):
        csv = pd.read_csv(FFA_CSV)
        fk = csv[(csv["position"] == "K") & csv["points"].notna()]
        kd = pd.read_sql(f"SELECT * FROM nflv_kdst_proj WHERE season={SEASON}", con)
        kd = kd[~((kd["position"] == "K") & (kd.get("conf") == "ffa"))]
        have = {norm(n) for n in kd.loc[kd["position"] == "K", "name"]}
        have_il = {initial_last(n) for n in kd.loc[kd["position"] == "K", "name"]}
        teams = set(pd.read_sql(
            f"SELECT DISTINCT team FROM nflv_rosters_2026", con)["team"].dropna())
        tmap = {"LVR": "LV", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}

        newk = []
        for r in fk.itertuples(index=False):
            if norm(r.player) in have or initial_last(r.player) in have_il:
                continue
            tm = tmap.get(r.team, r.team)
            if tm not in teams:
                tm = r.team
            newk.append({"position": "K", "name": r.player, "team": tm,
                         "proj_pts": float(r.points), "tier": None,
                         "conf": "ffa", "season": SEASON})
            print(f"  + K   {r.player:<24} FFA {float(r.points):>6.1f} pts  team={tm}")
        if newk:
            allk = pd.concat([kd, pd.DataFrame(newk)], ignore_index=True)
            other = pd.read_sql(
                f"SELECT * FROM nflv_kdst_proj WHERE season<>{SEASON}", con)
            pd.concat([other, allk], ignore_index=True).to_sql(
                "nflv_kdst_proj", con, if_exists="replace", index=False)
        print(f"nflv_kdst_proj {SEASON}: +{len(newk)} FFA kickers")

    con.commit()
    con.close()


if __name__ == "__main__":
    main()
