"""
Teammate-OPPORTUNITY features (vacated vs incoming touches), all seasons incl. 2026.

The season model uses a player's OWN prior stats, so it misses structural changes
in his situation. The validated, sizable one (test_opportunity_signals.py): a
RETURNING RB whose backfield LOST its lead back beats projection ~+1.7 PPG
(replicates 2013-19 & 2020-25). This builds the feature for every season and, for
2026, uses the CURRENT (post-FA/draft) rosters so the board can flag it now.

Per returning player (on his team in both Y-1 and Y):
  vac_rb_carries  prior-yr RB carries that LEFT the team (departures)
  inc_rb_carries  prior-yr RB carries that ARRIVED (FA/trade)
  vac_pc_targets / inc_pc_targets  same for WR+TE targets
  rook_rb / rook_wr  top-50 rookies drafted at the position
  vacated_role    1 if RB & vac>=150 & inc<100 & no top-50 rookie RB (the signal)

Writes nflv_opportunity.  (For 2026, fetches load_rosters(2026) for membership.)
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
TARGET = 2026


def roster_2026():
    """Current 2026 roster membership: gsis_id -> team (post free agency / draft)."""
    import nflreadpy as nfl
    df = nfl.load_rosters([TARGET])
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    df = df.rename(columns={"gsis_id": "player_id"})
    df = df[df.player_id.notna() & df.team.notna()][["player_id", "team", "position"]].copy()
    # persist for reproducibility
    con = sqlite3.connect(DB)
    df.assign(season=TARGET).to_sql("nflv_rosters_2026", con, if_exists="replace", index=False)
    con.close()
    return df


def main():
    con = sqlite3.connect(DB)
    s = pd.read_sql("""SELECT player_id, season, recent_team team, position, carries, targets
                       FROM nflv_season WHERE position IN ('RB','WR','TE')""", con)
    s = s.groupby(["player_id", "season"], as_index=False).agg(
        team=("team", "last"), position=("position", "last"),
        carries=("carries", "sum"), targets=("targets", "sum"))
    dr = pd.read_sql("SELECT season, team, position, pick FROM nflv_draft WHERE position IN ('RB','WR','TE')", con)
    con.close()

    vol = {(r.player_id, r.season): r for r in s.itertuples()}
    pos_of = {}                                   # latest known position per player
    for r in s.itertuples(): pos_of[r.player_id] = r.position
    members = {}
    for r in s.itertuples(): members.setdefault((r.team, r.season), []).append(r.player_id)

    # 2026 membership from current rosters (prior-year volume still comes from 2025 stats)
    r26 = roster_2026()
    for r in r26.itertuples():
        members.setdefault((r.team, TARGET), []).append(r.player_id)
        pos_of.setdefault(r.player_id, r.position)

    rook = {}
    for r in dr.itertuples():
        if r.pick and r.pick <= 50: rook.setdefault((r.team, r.season), []).append(r.position)

    rows = []
    for (team, Y), ids in members.items():
        if Y < 2013: continue
        prev = set(members.get((team, Y-1), [])); cur = set(ids)
        def tot(pidset, posset):
            c = t = 0.0
            for pid in pidset:
                v = vol.get((pid, Y-1))
                if v and v.position in posset: c += v.carries or 0; t += v.targets or 0
            return c, t
        vac_rb_c, _ = tot(prev - cur, {"RB"}); _, vac_pc_t = tot(prev - cur, {"WR", "TE"})
        inc_rb_c, _ = tot(cur - prev, {"RB"}); _, inc_pc_t = tot(cur - prev, {"WR", "TE"})
        rk = rook.get((team, Y), [])
        for pid in cur:
            if pid not in prev: continue                       # returning only
            pos = pos_of.get(pid)
            rows.append({"player_id": pid, "season": Y, "team": team, "position": pos,
                         "vac_rb_carries": vac_rb_c, "inc_rb_carries": inc_rb_c,
                         "vac_pc_targets": vac_pc_t, "inc_pc_targets": inc_pc_t,
                         "rook_rb": rk.count("RB"), "rook_wr": rk.count("WR") + rk.count("TE")})
    o = pd.DataFrame(rows)
    o["vacated_role"] = ((o.position == "RB") & (o.vac_rb_carries >= 150) &
                         (o.inc_rb_carries < 100) & (o.rook_rb == 0)).astype(int)

    con = sqlite3.connect(DB)
    o.to_sql("nflv_opportunity", con, if_exists="replace", index=False); con.close()
    print(f"nflv_opportunity: {len(o):,} returning player-seasons (2013-{TARGET})")
    v26 = o[(o.season == TARGET) & (o.vacated_role == 1)]
    print(f"\n2026 RBs flagged 'vacated lead role' ({len(v26)}):")
    nm = pd.read_sql("SELECT player_id, player_display_name FROM nflv_season WHERE season=2025",
                     sqlite3.connect(DB)).drop_duplicates("player_id").set_index("player_id").player_display_name.to_dict()
    for _, r in v26.sort_values("vac_rb_carries", ascending=False).iterrows():
        print(f"  {str(nm.get(r.player_id, r.player_id))[:22]:22s} {r.team}  vacated {r.vac_rb_carries:.0f} carries, incoming {r.inc_rb_carries:.0f}")


if __name__ == "__main__":
    main()
