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
    ap.add_argument("--roster", action="store_true",
                    help="show my depth chart: ppg, bye, value over the waiver wire, weeks started")
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
    from gm.trades import waiver_levels, weekly_lineup, _roster_players
    fa = waiver_levels(est)
    weeks = sim.week_columns(cfg)
    print("  waiver wire (best free agent, pts/wk): " + "  ".join(f"{p} {v:.1f}" for p, v in fa.items())
          + "   byes and waiver fill are in the simulation")
    print()
    if a.roster:
        cmd_roster(cfg, est, me, fa, weeks, names)
    scores = sim.player_team_scores(cfg, est, a.sims, need, rng=np.random.default_rng(a.seed),
                                    common_seed=a.seed, weeks=weeks)
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
        for tag, pl in (("out", p["give_players"]), ("in ", p["get_players"])):
            for v in pl:
                print(f"        {tag}  {v['name']:24s} {v['pos']:3s} {v['mean']:5.1f} pts/wk  "
                      f"plays {v['p_play']:.2f}  bye wk{v['bye'] or '-'}  "
                      f"over waiver {v['mean'] * v['p_play'] - fa.get(v['pos'], 0):+.1f}")
        d = p["weekly_me"]; wk = p["weeks"]
        reg = [(w, x) for w, x in zip(wk, d) if w in cfg.regular_weeks]
        print(f"        your lineup by week: " + "  ".join(f"wk{w} {x:+.1f}" for w, x in reg))
        print(f"        average {np.mean(d):+.2f} pts/wk over {len(d)} weeks incl. playoffs")


def cmd_roster(cfg, est, me, fa, weeks, names):
    """My depth chart in one table: what each player is worth, when he sits, whether he starts."""
    from gm.trades import expected_lineup
    mine = [v for (t, _), v in est.items() if t == me]
    # how many remaining weeks each player is in the expected starting lineup
    starts = {v["name"]: 0 for v in mine}
    flex = cfg.raw["roster"].get("flex_eligibility", {})
    dedicated = {s: c for s, c in cfg.starters if s not in flex}
    flexes = [(s, c) for s, c in cfg.starters if s in flex]
    for w in weeks:
        avail = sorted([v for v in mine if v.get("bye") != w], key=lambda v: -v["mean"] * v["p_play"])
        used = {}
        chosen = []
        for v in avail:
            need = dedicated.get(v["pos"], 0)
            if used.get(v["pos"], 0) < need and v["mean"] * v["p_play"] >= fa.get(v["pos"], 0):
                used[v["pos"]] = used.get(v["pos"], 0) + 1; chosen.append(v)
        for slot, count in flexes:
            left = [v for v in avail if v["pos"] in flex[slot] and v not in chosen
                    and v["mean"] * v["p_play"] >= fa.get(v["pos"], 0)]
            chosen += left[:count]
        for v in chosen:
            starts[v["name"]] += 1
    print(f"  {names.get(me, me)}: depth chart  (waiver = best free agent at the position)")
    print(f"  {'player':24s} {'pos':3s} {'pts/wk':>7} {'plays':>6} {'bye':>5} {'vs waiver':>10} {'starts':>7}")
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        for v in sorted([v for v in mine if v["pos"] == pos], key=lambda v: -v["mean"]):
            print(f"  {v['name'][:24]:24s} {pos:3s} {v['mean']:7.1f} {v['p_play']:6.2f} "
                  f"{('wk' + str(v['bye'])) if v.get('bye') else '-':>5} "
                  f"{v['mean'] * v['p_play'] - fa.get(pos, 0):+10.1f} {starts[v['name']]:4d}/{len(weeks)}")
    print()


if __name__ == "__main__":
    main()
