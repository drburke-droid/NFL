"""The trade search.

Its output is a list of proposals, so the load-bearing property is that nothing appears on it
unless the simulation says both sides gain -- a wishlist is easy and useless.
"""
import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm.config import load
from gm.players import ros_estimates
from gm import trades as T
from gm import simulate as sim

REAL = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
pytestmark = pytest.mark.skipif(not os.path.exists(REAL), reason="kuhn_2026.json not built")


@pytest.fixture(scope="module")
def league():
    c = load(REAL)
    return c, ros_estimates(c)


# ---------- stage one has to agree with the simulator about what a lineup is ----------

def test_expected_lineup_counts_only_startable_players(league):
    c, _ = league
    many = {"QB": [10] * 5, "RB": [10] * 5, "WR": [10] * 5, "TE": [10] * 5, "K": [10], "DST": [10]}
    assert T.expected_lineup(c, many) == pytest.approx(90.0), "9 starters at 10, bench ignored"


def test_expected_lineup_uses_the_flex(league):
    c, _ = league
    one_te = {"QB": [0], "RB": [0, 0], "WR": [0, 0], "TE": [5, 4], "K": [0], "DST": [0]}
    # TE1 fills the TE slot, TE2 takes the flex
    assert T.expected_lineup(c, one_te) == pytest.approx(9.0)


def test_expected_lineup_ignores_a_player_with_nowhere_to_play(league):
    c, _ = league
    a = {"QB": [20, 19, 18], "RB": [5, 5], "WR": [5, 5], "TE": [5], "K": [1], "DST": [1]}
    b = dict(a, QB=[20])
    assert T.expected_lineup(c, a) == T.expected_lineup(c, b), "a 2nd and 3rd QB cannot start"


def test_packages_are_singles_and_pairs():
    assert sorted(T._packages({"a": 1, "b": 2}, 1)) == [("a",), ("b",)]
    p = T._packages({"a": 1, "b": 2, "c": 3}, 2)
    assert len([x for x in p if len(x) == 1]) == 3
    assert len([x for x in p if len(x) == 2]) == 3


# ---------- the filter ----------

def test_every_proposal_helps_both_sides(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=6, top_n=5, n_sims=2000, seed=1,
                      played_weeks=[1])
    assert r["proposals"], "the search should find something for this roster"
    for p in r["proposals"]:
        assert p["my_p_title"] > 0 and p["their_p_title"] > 0


def test_proposals_are_ranked_by_my_own_gain(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=6, top_n=5, n_sims=2000, seed=1,
                      played_weeks=[1])
    d = [p["my_p_title"] for p in r["proposals"]]
    assert d == sorted(d, reverse=True)


def test_top_n_is_respected(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=6, top_n=2, n_sims=1500, seed=1,
                      played_weeks=[1])
    assert len(r["proposals"]) <= 2


def test_two_for_two_is_not_searched(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=3, top_n=3, n_sims=1500, seed=1,
                      played_weeks=[1])
    assert all(not (len(p["give"]) == 2 and len(p["get"]) == 2) for p in r["proposals"])


def test_one_for_one_only_finds_almost_nothing(league):
    """The result that motivated consolidation: a straight swap mostly moves value, not creates it."""
    c, est = league
    one = T.find_trades(c, est, c.raw["my_team_id"], shortlist=1, top_n=1, n_sims=1000, seed=1,
                        played_weeks=[1], max_package=1)
    two = T.find_trades(c, est, c.raw["my_team_id"], shortlist=1, top_n=1, n_sims=1000, seed=1,
                        played_weeks=[1], max_package=2)
    assert two["considered"] > one["considered"] * 10, \
        f"packages should open up far more mutual gains: {one['considered']} -> {two['considered']}"


def test_the_search_never_proposes_trading_with_yourself(league):
    c, est = league
    me = c.raw["my_team_id"]
    r = T.find_trades(c, est, me, shortlist=4, top_n=4, n_sims=1500, seed=1, played_weeks=[1])
    assert all(p["with_team"] != me for p in r["proposals"])
