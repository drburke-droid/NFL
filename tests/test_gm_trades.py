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


def _pl(**kw):
    return {k: {"mean": v, "p_play": 1.0} for k, v in kw.items()}


def test_packages_grow_with_the_allowed_size():
    assert sorted(T._packages(_pl(a=1, b=2), 1)) == [("a",), ("b",)]
    two = T._packages(_pl(a=1, b=2, c=3), 2)
    assert len([x for x in two if len(x) == 1]) == 3
    assert len([x for x in two if len(x) == 2]) == 3
    three = T._packages(_pl(a=1, b=2, c=3), 3)
    assert len([x for x in three if len(x) == 3]) == 1


def test_the_pool_keeps_the_best_players_not_the_first_ones():
    """Filler at the bottom of a full roster multiplies the search without adding a real trade."""
    got = T._packages(_pl(dud=1.0, star=30.0, mid=10.0), 1, pool=2)
    assert sorted(x[0] for x in got) == ["mid", "star"]


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


# ---------- rosters are full, so an uneven trade forces a cut ----------

def test_an_uneven_trade_forces_the_receiver_to_drop(league):
    c, est = league
    keyed = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    cap = sim.roster_capacity(c)
    me = c.raw["my_team_id"]
    other = next(t["team_id"] for t in c.teams if t["team_id"] != me)
    mine = [p for (t, p) in keyed if t == me][:1]
    theirs = [p for (t, p) in keyed if t == other][:2]
    moves = [(mine[0], me, other)] + [(x, other, me) for x in theirs]
    after = sim.apply_trade(keyed, moves, cap)
    size = lambda d, t: sum(1 for (x, _) in d if x == t)
    assert size(after, me) == cap, "taking two back for one must not leave me over the limit"
    assert size(after, other) == cap - 1


def test_the_dropped_player_is_the_least_useful_one(league):
    c, est = league
    keyed = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    me = c.raw["my_team_id"]
    other = next(t["team_id"] for t in c.teams if t["team_id"] != me)
    mine = [p for (t, p) in keyed if t == me]
    worst = min(mine, key=lambda p: keyed[(me, p)]["mean"] * keyed[(me, p)]["p_play"])
    theirs = [p for (t, p) in keyed if t == other][:2]
    moves = [(mine[0], me, other)] + [(x, other, me) for x in theirs]
    after = sim.apply_trade(keyed, moves, sim.roster_capacity(c))
    assert (me, worst) not in after, "a manager cuts his least useful player, not an arbitrary one"


def test_no_capacity_means_no_trimming(league):
    c, est = league
    keyed = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    me = c.raw["my_team_id"]
    other = next(t["team_id"] for t in c.teams if t["team_id"] != me)
    theirs = [p for (t, p) in keyed if t == other][:2]
    after = sim.apply_trade(keyed, [(x, other, me) for x in theirs])
    assert sum(1 for (x, _) in after if x == me) == sim.roster_capacity(c) + 2


def test_bigger_packages_open_up_more_trades(league):
    """Three-a-side finds far more mutual gain than two, which found far more than one."""
    c, est = league
    kw = dict(shortlist=1, top_n=1, n_sims=500, seed=1, played_weeks=[1])
    one = T.find_trades(c, est, c.raw["my_team_id"], max_package=1, max_combined=2, **kw)
    two = T.find_trades(c, est, c.raw["my_team_id"], max_package=2, max_combined=3, **kw)
    three = T.find_trades(c, est, c.raw["my_team_id"], max_package=3, max_combined=4, **kw)
    assert one["considered"] < two["considered"] < three["considered"]


def test_no_single_partner_can_fill_the_whole_list(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=12, top_n=8, n_sims=1500, seed=1,
                      played_weeks=[1], max_per_team=2)
    from collections import Counter
    if len(r["proposals"]) > 2:
        assert max(Counter(p["with_team"] for p in r["proposals"]).values()) <= 2


# ---------- the next-season term ----------

def _board():
    from gm import keepers as kp
    return kp.load_board()


def test_without_a_board_no_keeper_term_is_applied(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=4, top_n=3, n_sims=1500, seed=1)
    assert all(p["keeper_me"] == 0.0 and p["keeper_them"] == 0.0 for p in r["proposals"])


def test_keeper_deltas_are_reported_even_at_zero_weight(league):
    """Two honest numbers beat one score built on a made-up exchange rate."""
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=6, top_n=4, n_sims=1500, seed=1,
                      board=_board(), keeper_discount=0.0)
    assert r["proposals"]
    assert all("keeper_me" in p and "keeper_them" in p for p in r["proposals"])
    assert all(p["score_me"] == pytest.approx(p["my_p_title"]) for p in r["proposals"])


def test_the_win_curve_prices_a_point_differently_for_each_team(league):
    """Why one exchange rate will not do: a point a week is worth far more to a contender."""
    c, est = league
    keyed = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    best = max(c.teams, key=lambda t: t["points_for"])["team_id"]
    worst = min(c.teams, key=lambda t: t["points_for"])["team_id"]
    hot = T._title_slope(c, keyed, best, n_sims=2500, seed=1)
    cold = T._title_slope(c, keyed, worst, n_sims=2500, seed=1)
    assert hot > cold * 2, f"contender {hot:.5f} should value a point far above cellar {cold:.5f}"


def test_weighting_next_season_changes_which_trades_survive(league):
    """It mostly REJECTS: trades that buy this year by spending keeper value stop clearing."""
    c, est = league
    kw = dict(shortlist=15, top_n=8, n_sims=2500, seed=1, board=_board())
    now = T.find_trades(c, est, c.raw["my_team_id"], keeper_discount=0.0, **kw)
    later = T.find_trades(c, est, c.raw["my_team_id"], keeper_discount=1.0, **kw)
    assert now["considered"] > later["considered"], \
        "pricing next season must narrow what stage one is willing to put forward"
    key = lambda r: {(tuple(p["give"]), tuple(p["get"])) for p in r["proposals"]}
    assert key(now) != key(later), "the objective must actually change the answer"


def test_proposals_rank_by_the_chosen_objective(league):
    c, est = league
    r = T.find_trades(c, est, c.raw["my_team_id"], shortlist=10, top_n=5, n_sims=1500, seed=1,
                      board=_board(), keeper_discount=1.0)
    s = [p["score_me"] for p in r["proposals"]]
    assert s == sorted(s, reverse=True)
