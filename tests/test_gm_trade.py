"""Trade evaluation.

The load-bearing test is the empty trade: with player draws keyed to the player rather than the
roster, trading nothing must change nothing EXACTLY. If it does not, every delta this module
reports is partly Monte Carlo noise, and the deltas are small enough that noise would swamp them.
"""
import os, sys
import numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm.config import load
from gm.players import ros_estimates
from gm import simulate as sim

REAL = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
pytestmark = pytest.mark.skipif(not os.path.exists(REAL), reason="kuhn_2026.json not built")


@pytest.fixture(scope="module")
def league():
    c = load(REAL)
    return c, ros_estimates(c)


def find(est, name):
    return next((t, p) for (t, p), v in est.items() if v["name"] == name)


def by_prefix(cfg, pre):
    return next(t for t in cfg.teams if t["name"].startswith(pre))


# ---------- the noise floor must be zero ----------

def test_trading_nothing_changes_nothing_exactly(league):
    c, est = league
    r = sim.evaluate_trade(c, est, [], n_sims=3000, seed=1, played_weeks=[1])
    for t, b in r["before"].items():
        assert r["after"][t]["p_title"] == b["p_title"]
        assert r["after"][t]["exp_points_for"] == b["exp_points_for"]


def test_the_same_seed_reproduces_the_same_verdict(league):
    c, est = league
    ch = find(est, "Ja'Marr Chase")
    mv = [(ch[1], ch[0], by_prefix(c, "Ugh")["team_id"])]
    a = sim.evaluate_trade(c, est, mv, n_sims=2000, seed=4, played_weeks=[1])
    b = sim.evaluate_trade(c, est, mv, n_sims=2000, seed=4, played_weeks=[1])
    assert a["delta"] == b["delta"]


def test_a_player_stream_does_not_depend_on_his_roster():
    """The whole method rests on this: the same id draws the same season either side of a trade."""
    x = sim._player_rng(7, "00-0012345").normal(size=5)
    y = sim._player_rng(7, "00-0012345").normal(size=5)
    z = sim._player_rng(7, "00-0099999").normal(size=5)
    assert (x == y).all() and not (x == z).all()


# ---------- direction ----------

def test_giving_a_starter_away_for_nothing_hurts(league):
    c, est = league
    ch = find(est, "Ja'Marr Chase")
    r = sim.evaluate_trade(c, est, [(ch[1], ch[0], by_prefix(c, "Ugh")["team_id"])],
                           n_sims=6000, seed=2, played_weeks=[1])
    assert r["delta"][ch[0]]["p_playoffs"]["delta"] < 0
    assert r["delta"][ch[0]]["exp_points_for"]["delta"] < 0


def test_receiving_a_starter_for_nothing_helps(league):
    c, est = league
    ch = find(est, "Ja'Marr Chase")
    ugh = by_prefix(c, "Ugh")["team_id"]
    r = sim.evaluate_trade(c, est, [(ch[1], ch[0], ugh)], n_sims=6000, seed=2, played_weeks=[1])
    assert r["delta"][ugh]["exp_points_for"]["delta"] > 0


def test_a_surplus_quarterback_is_worth_almost_nothing_to_his_own_team(league):
    """Ugh carries three QBs in a one-QB league. The third cannot start, so losing him is free."""
    c, est = league
    ugh = by_prefix(c, "Ugh")
    qbs = sorted([est[(ugh["team_id"], e["player_id"])] for e in ugh["roster"]
                  if e["pos"] == "QB"], key=lambda v: -v["mean"])
    if len(qbs) < 3:
        pytest.skip("that roster no longer carries three quarterbacks")
    third = next(p for (t, p), v in est.items() if t == ugh["team_id"] and v is qbs[2])
    other = by_prefix(c, "Maple")["team_id"]
    r = sim.evaluate_trade(c, est, [(third, ugh["team_id"], other)],
                           n_sims=6000, seed=3, played_weeks=[1])
    lost = -r["delta"][ugh["team_id"]]["exp_points_for"]["delta"]
    assert lost < 15.0, f"a benched third QB cost {lost:.1f} season points to give away"


def test_the_same_trade_is_good_for_one_side_and_bad_for_the_other(league):
    """The win curve, not an opinion: a verdict has to be per team, not per trade."""
    c, est = league
    ch = find(est, "Ja'Marr Chase"); mah = find(est, "Patrick Mahomes")
    r = sim.evaluate_trade(c, est, [(ch[1], ch[0], mah[0]), (mah[1], mah[0], ch[0])],
                           n_sims=8000, seed=1, played_weeks=[1])
    a = r["delta"][ch[0]]["exp_points_for"]["delta"]     # Glass Joe: gains a QB it cannot start above
    b = r["delta"][mah[0]]["exp_points_for"]["delta"]    # Ugh: converts a benched QB into a starter
    assert b > 0 > a, f"expected one gain and one loss, got {a:+.1f} and {b:+.1f}"


def test_surplus_for_surplus_can_help_both_sides(league):
    """Positive-sum trades exist, and a value chart comparing two players cannot see them.

    Built rather than found: one team stacked at quarterback and thin at receiver, the other the
    reverse. Each gives up a player it could never start and receives one it can.
    """
    c, est = league
    a_id, b_id = c.teams[0]["team_id"], c.teams[1]["team_id"]
    built = {}
    for tid, many, few in ((a_id, "QB", "WR"), (b_id, "WR", "QB")):
        team = next(t for t in c.teams if t["team_id"] == tid)
        for e in team["roster"]:
            pos = e["pos"]
            mean = 22.0 if pos == many else (3.0 if pos == few else 8.0)
            built[(tid, str(e["player_id"]))] = {
                "name": e["name"], "pos": pos, "mean": mean, "sd": 1.0,
                "p_play": 1.0, "source": "built", "key": str(e["player_id"])}
    for (t, pid), v in est.items():
        built.setdefault((t, str(pid)), {**v, "key": str(pid)})
    spare_qb = [p for (t, p), v in built.items() if t == a_id and v["pos"] == "QB"]
    spare_wr = [p for (t, p), v in built.items() if t == b_id and v["pos"] == "WR"]
    if len(spare_qb) < 2 or len(spare_wr) < 3:
        pytest.skip("rosters lack the surplus this test needs")
    moves = [(spare_qb[1], a_id, b_id), (spare_wr[2], b_id, a_id)]
    r = sim.evaluate_trade(c, built, moves, n_sims=4000, seed=6, played_weeks=[1])
    ga = r["delta"][a_id]["exp_points_for"]["delta"]
    gb = r["delta"][b_id]["exp_points_for"]["delta"]
    assert ga > 0 and gb > 0, f"surplus-for-surplus should gain both: {ga:+.1f} and {gb:+.1f}"


# ---------- plumbing ----------

def test_moving_a_player_off_the_wrong_team_is_an_error(league):
    c, est = league
    ch = find(est, "Ja'Marr Chase")
    keyed = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    with pytest.raises(KeyError, match="not on team"):
        sim.apply_trade(keyed, [(ch[1], "nobody", "t1")])


def test_apply_trade_leaves_the_original_untouched(league):
    c, est = league
    keyed = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    ch = find(est, "Ja'Marr Chase")
    before = set(keyed)
    sim.apply_trade(keyed, [(ch[1], ch[0], by_prefix(c, "Ugh")["team_id"])])
    assert set(keyed) == before, "apply_trade must not mutate the estimates it was given"


def test_only_the_involved_teams_are_reported(league):
    c, est = league
    ch = find(est, "Ja'Marr Chase"); ugh = by_prefix(c, "Ugh")["team_id"]
    r = sim.evaluate_trade(c, est, [(ch[1], ch[0], ugh)], n_sims=1500, seed=5, played_weeks=[1])
    assert set(r["delta"]) == {ch[0], ugh}
    assert len(r["before"]) == len(c.teams)


def test_a_forced_cut_never_empties_a_dedicated_slot():
    """A manager taking on a receiver drops a bench back, not his only defence -- even when the
    defence carries the lowest mean on the roster (it usually does: 5.59 flat)."""
    mk = lambda pos, mean: {"pos": pos, "mean": mean, "p_play": 1.0, "name": f"{pos}{mean}"}
    roster = [mk("DST", 5.6), mk("K", 8.9), mk("QB", 20.0), mk("RB", 7.8), mk("RB", 8.6),
              mk("RB", 10.0), mk("WR", 12.0), mk("WR", 16.0), mk("TE", 9.0)]
    slots = {"QB": 1, "K": 1, "DST": 1, "RB": 2, "WR": 2, "TE": 1}
    cut = sim.forced_cuts(roster, capacity=8, starters=slots)
    assert [v["name"] for v in cut] == ["RB7.8"], "the lowest cuttable, not the lowest"
    assert [v["name"] for v in sim.forced_cuts(roster, 8)] == ["DST5.6"], "without slots it is the old rule"
    two = sim.forced_cuts(roster + [mk("WR", 6.0)], 8, slots)
    assert {v["name"] for v in two} == {"WR6.0", "RB7.8"}
    # every slot exactly filled: there is nothing honest to cut, so the old rule applies
    tight = [v for v in roster if v["name"] != "RB10.0"]
    assert [v["name"] for v in sim.forced_cuts(tight, 7, slots)] == ["DST5.6"]
