"""One command to see where your league stands and what trades are worth proposing.

Everything the gm/ package does, without writing Python. Run it with no arguments for the title
race; add --trades for proposals.

    python scripts/gm_report.py
    python scripts/gm_report.py --trades
    python scripts/gm_report.py --trades --keeper-discount 1.0
    python scripts/gm_report.py --check          # what is stale and how to refresh it

The numbers are simulated, not looked up, so they move a little run to run. --sims raises the
count if you want them steadier; --seed makes any run reproducible.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEAGUE = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
BOARD = os.path.join(ROOT, "outputs", "draft_tool", "keepers_2026.js")
ESPN = os.path.join(ROOT, "outputs", "espn_league.json")
STATE = os.path.join(ROOT, "outputs", "espn_league_state.json")


def age_hours(path):
    if not os.path.exists(path):
        return None
    return (datetime.now(timezone.utc).timestamp() - os.path.getmtime(path)) / 3600


def cmd_check():
    """What the report reads, how old it is, and the exact command to refresh it."""
    rows = [
        (ESPN, "rosters, scoring, keeper prices", "Actions -> Pull ESPN league"),
        (STATE, "schedule, standings, playoff shape", "Actions -> Pull ESPN league"),
        (LEAGUE, "the league config the simulator loads", "python scripts/build_gm_league.py"),
        (BOARD, "keeper values and inflation bumps", "python scripts/predict_keepers.py"),
    ]
    print("input                              age      what it carries")
    stale = []
    for path, what, how in rows:
        a = age_hours(path)
        tag = "MISSING" if a is None else (f"{a:5.1f}h" + ("  STALE" if a > 48 else ""))
        print(f"  {os.path.relpath(path, ROOT):32s} {tag:14s} {what}")
        if a is None or a > 48:
            stale.append((path, how))
    if not stale:
        print("\neverything is fresh.")
        return
    print("\nto refresh, in this order:")
    seen = set()
    for _, how in stale:
        if how not in seen:
            seen.add(how); print(f"  {how}")
    print("\nThe ESPN pull needs ESPN_S2 / ESPN_SWID. They are repo Secrets for the Action; for a\n"
          "local run put them in data/espn_cookies.json (gitignored). See ESPN_SETUP.md.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", default=LEAGUE)
    ap.add_argument("--team", default=None, help="team_id to view as (default: my_team_id)")
    ap.add_argument("--trades", action="store_true", help="also search for trades worth proposing")
    ap.add_argument("--keeper-discount", type=float, default=0.0,
                    help="0 = judge trades on this season only; 1 = next season counts fully")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--shortlist", type=int, default=25, help="candidates to simulate exactly")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="report input freshness and exit")
    a = ap.parse_args()

    if a.check:
        return cmd_check()

    import numpy as np
    from gm.config import load
    from gm.players import ros_estimates
    from gm import simulate as sim

    cfg = load(a.league)
    me = a.team or cfg.raw.get("my_team_id")
    names = {t["team_id"]: t["name"] for t in cfg.teams}
    est = ros_estimates(cfg)
    done = sim.completed_weeks(cfg)
    need, future = sim.weeks_needed(cfg)

    print(f"{cfg.raw['name']}  ({len(cfg.teams)} teams, {cfg.raw['season']})")
    print(f"  weeks complete: {len(done)}   simulating {future[0]}-{future[-1]} "
          f"plus playoffs {cfg.playoff_weeks}   [{a.sims:,} seasons]")
    h = ros_estimates.last_ridge
    if h:
        print(f"  player means: fitted ridge through week {h['week']}, {h['fitted']}/{h['rostered']} "
              f"skill players fitted, FFA next-week projection for {h['ffa_coverage']:.0%}"
              + ("  [ffa stale: next week's file not uploaded yet]" if h["ffa_stale"] else ""))
    else:
        print("  player means: legacy blend (gm/ros_model.json not found)")
    print()
    scores = sim.player_team_scores(cfg, est, a.sims, need, rng=np.random.default_rng(a.seed),
                                    common_seed=a.seed)
    r = sim.run(cfg, scores)
    print(f"  {'team':30s} {'W-L':>6} {'PF':>7} {'proj/wk':>8} {'playoffs':>9} {'title':>7} {'seed':>6}")
    per_team = scores.mean(axis=(0, 2))
    for i, t in enumerate(sorted(cfg.teams, key=lambda x: -r["teams"][x["team_id"]]["p_title"])):
        o = r["teams"][t["team_id"]]
        j = [x["team_id"] for x in cfg.teams].index(t["team_id"])
        mark = "  <-- you" if t["team_id"] == me else ""
        rec = f"{t['record']['w']}-{t['record']['l']}"
        print(f"  {names[t['team_id']][:30]:30s} {rec:>6} {t['points_for']:7.1f} {per_team[j]:8.1f} "
              f"{o['p_playoffs']:9.3f} {o['p_title']:7.3f} {o['exp_seed']:6.2f}{mark}")

    if not a.trades:
        print("\n(add --trades to search for deals worth proposing)")
        return

    from gm.trades import find_trades
    board = None
    if os.path.exists(BOARD):
        from gm.keepers import load_board
        board = load_board(BOARD)
    print()
    print(f"searching trades for {names.get(me, me)}"
          + (f", next season weighted {a.keeper_discount:g}" if a.keeper_discount else
             ", this season only")
          + "  (a minute or two)")
    res = find_trades(cfg, est, me, shortlist=a.shortlist, top_n=a.top, n_sims=a.sims,
                      seed=a.seed, board=board, keeper_discount=a.keeper_discount)
    print(f"  {res['considered']} candidates cleared stage one, {res['simulated']} simulated, "
          f"{len(res['proposals'])} help both sides")
    if not res["proposals"]:
        print("\n  Nothing clears the bar. That is an answer: at this keeper discount every trade\n"
              "  on offer costs you more than it returns. Try --keeper-discount 0 to see the\n"
              "  win-now list, or raise --shortlist to simulate further down the ranking.")
        return
    for p in res["proposals"]:
        print()
        print(f"  GIVE  {', '.join(p['give'])}")
        print(f"  GET   {', '.join(p['get'])}   ({names.get(p['with_team'], p['with_team'])})")
        print(f"        you   P(title) {p['my_p_title']:+.4f}   P(playoffs) {p['my_p_playoffs']:+.4f}"
              f"   season pts {p['my_points']:+.0f}   keeper ${p['keeper_me']:+.0f}")
        print(f"        them  P(title) {p['their_p_title']:+.4f}   keeper ${p['keeper_them']:+.0f}")


if __name__ == "__main__":
    main()
