"""Strategy lab: rules replayed as 10% swipes on the sent line; week 3 stays out of the out-of-sample totals."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import fan_strategies as F


def _g():
    # two players, one stat each; A's actual is above the send line, B's below
    rows = []
    for fan, pl, pos, arrows, proj, wk in [("X", "A", "WR", 1, 16.0, 4), ("Y", "A", "WR", 1, 16.0, 4), ("Z", "A", "WR", -2, 16.0, 4),
                                           ("X", "B", "RB", -1, 10.0, 4), ("REB", "B", "RB", 1, 10.0, 4), ("X", "A", "WR", 1, 16.0, 3)]:
        base, actual = (50.0, 70.0) if pl == "A" else (60.0, 40.0)
        adj = base * (1 + 0.1 * arrows)
        rows.append(dict(fan=fan, week=wk, player=pl, pos=pos, stat="rec_yds", arrows=arrows, rule="s1", proj_pts=proj,
                         base=base, actual=actual, pending=False, no_send=False,
                         removed_pts=(abs(actual - base) - abs(actual - adj)) * 0.1))
    return pd.DataFrame(rows)


def test_crowd_majority_moves_each_player_once_the_way_most_callers_went():
    lab = {r["key"]: r for r in F.lab(_g())["rules"]}
    wk4 = lab["crowd"]["weeks"]["4"]
    # A: 2 boosts vs 1 fade -> boost, +10% of 50 = +5 yds closer = +0.5; B: 1-1 tie -> no call
    assert wk4 == {"calls": 1, "helped": 1, "pts": 0.5}


def test_fades_only_and_mid_tier_rule():
    lab = {r["key"]: r for r in F.lab(_g())["rules"]}
    assert lab["fades"]["weeks"]["4"]["calls"] == 2                       # A (Z faded) and B (X faded)
    assert lab["fade_mid_boosts"]["weeks"]["4"] == {"calls": 1, "helped": 0, "pts": -0.5}   # A projects 16 and was boosted


def test_swipe_size_rows_report_calls_as_made():
    lab = {r["key"]: r for r in F.lab(_g())["rules"]}
    assert lab["size_2"]["weeks"]["4"]["calls"] == 1                      # Z's 20% fade on A
    assert lab["size_1"]["weeks"]["4"]["calls"] == 4


def test_week_three_is_in_sample_only():
    out = F.lab(_g())
    crowd = next(r for r in out["rules"] if r["key"] == "crowd")
    assert crowd["out_of_sample"]["weeks"] == [4] and crowd["all"]["weeks"] == [3, 4]
    assert out["in_sample_weeks"] == [3]
