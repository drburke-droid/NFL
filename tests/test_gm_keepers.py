"""Keeper valuation.

The rules that matter here are the ones a generic model would get wrong: only three keepers count,
a traded player pays his NEW owner's inflation bump, and a cut player's keeper value is simply
gone.
"""
import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm import keepers as kp

BOARD_JS = os.path.join(ROOT, "outputs", "draft_tool", "keepers_2026.js")
pytestmark = pytest.mark.skipif(not os.path.exists(BOARD_JS), reason="keeper board not built")


@pytest.fixture(scope="module")
def board():
    return kp.load_board()


def fake(**kw):
    e = {"value": 40, "basis": 10, "waiver": False}
    e.update(kw)
    return e


# ---------- the bump follows the owner, not the player ----------

def test_a_cheaper_bump_makes_the_same_player_a_better_keep():
    e = fake(value=40, basis=10)
    assert kp.surplus_under(e, 0) == 30
    assert kp.surplus_under(e, 5) == 25, "a contender pays more to keep the same player"


def test_a_waiver_pickup_keeps_at_a_dollar():
    assert kp.surplus_under(fake(value=40, basis=99, waiver=True), 5) == 40 - (1 + 5)


# ---------- only three count ----------

def test_only_the_best_three_surpluses_count(board):
    tid = next(iter(board))
    names = list(board[tid]["players"])
    vals = sorted((kp.surplus_under(board[tid]["players"][n], board[tid]["bump"]) for n in names),
                  reverse=True)
    assert kp.team_keeper_value(board, tid, names) == pytest.approx(sum(vals[:3]))


def test_a_fourth_good_keeper_adds_nothing(board):
    tid = next(iter(board))
    ranked = sorted(board[tid]["players"],
                    key=lambda n: -kp.surplus_under(board[tid]["players"][n], board[tid]["bump"]))
    three = kp.team_keeper_value(board, tid, ranked[:3])
    four = kp.team_keeper_value(board, tid, ranked[:4])
    assert three == four, "the fourth-best surplus is worth nothing, like a fourth startable WR"


def test_players_off_the_board_contribute_nothing(board):
    tid = next(iter(board))
    assert kp.team_keeper_value(board, tid, ["Nobody At All"]) == 0.0


def test_an_unknown_team_is_worth_nothing(board):
    assert kp.team_keeper_value(board, "no-such-team", ["anyone"]) == 0.0


# ---------- trades ----------

def test_cutting_a_keeper_loses_his_value(board):
    tid = next(iter(board))
    ranked = sorted(board[tid]["players"],
                    key=lambda n: -kp.surplus_under(board[tid]["players"][n], board[tid]["bump"]))
    d = kp.keeper_delta(board, {tid: ranked[:3]}, {tid: ranked[1:3]}, [tid])
    assert d[tid] < 0, "dropping the best keeper must cost keeper value"


def test_a_traded_player_pays_his_new_owners_bump(board):
    """The standings loop: what a keeper costs depends on where his OWNER finishes."""
    cheap = min(board, key=lambda t: board[t]["bump"])
    dear = max(board, key=lambda t: board[t]["bump"])
    if board[cheap]["bump"] == board[dear]["bump"]:
        pytest.skip("every team has the same bump this season")
    star = max(board[dear]["players"],
               key=lambda n: kp.surplus_under(board[dear]["players"][n], 0))
    to_cheap = kp.keeper_delta(board, {cheap: []}, {cheap: [star]}, [cheap])[cheap]
    to_dear = kp.keeper_delta(board, {dear: []}, {dear: [star]}, [dear])[dear]
    assert to_cheap > to_dear, "the same player is worth more to the owner with the cheaper bump"


def test_the_board_matches_the_current_rosters(board):
    """It is generated from espn_league.json, and a stale board silently values the wrong team."""
    import json
    lg = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json")))
    for t in lg["teams"]:
        on_board = board.get(str(t["id"]), {}).get("players", {})
        skill = [e["name"] for e in t["roster"] if e["pos"] in ("QB", "RB", "WR", "TE")]
        missing = [n for n in skill if n not in on_board]
        assert len(missing) <= 2, f"team {t['id']} has {missing} on its roster but not its board"


# ---------- next season, priced from this one ----------

def _curve():
    return kp.dollar_curve()


def test_the_dollar_curve_is_monotone_and_convex_at_running_back():
    xs, ys = _curve()["RB"]
    assert all(b >= a for a, b in zip(ys, ys[1:])), "more points never costs less"
    assert kp.dollars_at(_curve(), "RB", 14) - kp.dollars_at(_curve(), "RB", 11) >         kp.dollars_at(_curve(), "RB", 11) - kp.dollars_at(_curve(), "RB", 8), "the curve steepens"


def test_uncertainty_is_worth_money_on_a_convex_curve():
    """A young back at 7 ppg is worth more than $ at 7 ppg: the upside tail pays, the downside is floored."""
    c = _curve()
    assert kp.expected_dollars(c, "RB", 7.0, age=23) > kp.dollars_at(c, "RB", 7.0)
    assert kp.expected_dollars(c, "RB", 7.0, age=23) > kp.expected_dollars(c, "RB", 7.0, age=29),         "a young player drifts up, an old one down"


def test_next_season_board_prices_a_waiver_pickup_at_a_dollar_plus_the_bump():
    from gm.config import load
    from gm import players as P
    cfg = load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "gm", "leagues", "kuhn_2026.json"))
    est = P.ros_estimates(cfg)
    b = kp.next_season_board(cfg, est)
    adds = [e for t in b.values() for e in t["players"].values() if e["waiver"]]
    drafted = [e for t in b.values() for e in t["players"].values() if not e["waiver"]]
    assert adds and all(e["basis"] == 1.0 and e["cost"] == 1.0 + kp.NEXT_BUMP for e in adds)
    assert all(e["cost"] == e["basis"] + kp.NEXT_BUMP for e in drafted)
    # an injured player is valued at his healthy level (a rookie who has never played has none)
    hurt = [e for t in b.values() for e in t["players"].values() if e.get("injury") in ("OUT", "INJURY_RESERVE")]
    assert all(e["value"] >= 1.0 for e in hurt)
    assert any(e["healthy_ppg"] > 5 and e["value"] > 1.0 for e in hurt), "a hurt veteran keeps his level"
    assert all(e["surplus"] == round(e["value"] - e["cost"], 1) for e in adds + drafted)


def test_the_bump_is_expected_over_the_playoff_race():
    from gm.config import load
    cfg = load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "gm", "leagues", "kuhn_2026.json"))
    assert kp.expected_bump(cfg, 1.0) == 5.0 and kp.expected_bump(cfg, 0.0) == 3.0
    assert abs(kp.expected_bump(cfg, 0.35) - 3.7) < 1e-9
    from gm import players as P
    est = P.ros_estimates(cfg)
    b = kp.next_season_board(cfg, est, bump={cfg.raw["my_team_id"]: 3.7})
    me = b[cfg.raw["my_team_id"]]
    assert me["bump"] == 3.7 and all(e["cost"] == e["basis"] + 3.7 for e in me["players"].values())
    other = next(t for t in b if t != cfg.raw["my_team_id"])
    assert b[other]["bump"] == kp.NEXT_BUMP, "teams not given a bump keep the default"
