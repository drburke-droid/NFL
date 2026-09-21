"""The rolling bias must be conservative: silent on thin history, capped, and skill-only."""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import bias_correction as bcorr


def wk(week, **pos):
    """pos like QB=(63, -0.1) -> a week block shaped like docs/sabersim_accuracy.json."""
    return {"week": week, "by_pos": {p: {"n": n, "bias": b} for p, (n, b) in pos.items()}}


def acc(*weeks, season=2026):
    return {"season": season, "weeks": list(weeks)}


def test_weights_by_row_count_across_weeks():
    a = acc(wk(1, WR=(100, -1.0)), wk(2, WR=(300, -2.0)))
    b, m = bcorr.recent_bias(a)
    assert b == pytest.approx((100 * -1.0 + 300 * -2.0) / 400)
    assert m["rows"] == 400 and [x["week"] for x in m["weeks"]] == [1, 2]


def test_thin_history_corrects_nothing():
    b, m = bcorr.recent_bias(acc(wk(1, WR=(50, -3.0))))
    assert b == 0.0 and "50 graded rows" in m["reason"]


def test_no_accuracy_file_is_a_no_op():
    assert bcorr.recent_bias(None)[0] == 0.0
    assert bcorr.recent_bias({})[0] == 0.0
    assert bcorr.recent_bias(acc())[0] == 0.0


def test_wrong_season_is_a_no_op():
    a = acc(wk(1, WR=(500, -1.0)), season=2025)
    b, m = bcorr.recent_bias(a, season=2026)
    assert b == 0.0 and "2025" in m["reason"]


def test_cap_limits_a_single_wild_week():
    b, m = bcorr.recent_bias(acc(wk(1, WR=(400, -9.0))), cap=2.0)
    assert b == -2.0 and "capped" in m["reason"]
    assert m["raw"] == -9.0


def test_only_the_recent_window_counts():
    a = acc(*[wk(i, WR=(400, -5.0)) for i in range(1, 5)], wk(9, WR=(400, -1.0)))
    b, _ = bcorr.recent_bias(a, weeks=1)
    assert b == pytest.approx(-1.0), "a 1-week window must use week 9, not the old ones"


def test_window_picks_the_latest_weeks_even_if_unordered():
    a = acc(wk(5, WR=(400, -1.0)), wk(1, WR=(400, -9.0)))
    b, m = bcorr.recent_bias(a, weeks=1)
    assert b == pytest.approx(-1.0) and [x["week"] for x in m["weeks"]] == [5]


def test_kickers_are_excluded():
    a = acc(wk(1, WR=(400, -1.0), K=(400, +5.0)))
    b, _ = bcorr.recent_bias(a)
    assert b == pytest.approx(-1.0), "K has its own source and must not move the skill correction"


def test_malformed_blocks_are_skipped_not_fatal():
    a = {"season": 2026, "weeks": [
        {"week": 1, "by_pos": {"WR": {"n": 400, "bias": -1.0}, "RB": {"n": None, "bias": -9.0},
                               "TE": {"n": 50}, "QB": "nonsense"}},
        {"no_week_key": True}, None,
    ]}
    b, m = bcorr.recent_bias(a)
    assert b == pytest.approx(-1.0) and m["rows"] == 400


def test_a_positive_bias_is_returned_as_is():
    b, _ = bcorr.recent_bias(acc(wk(1, WR=(400, +0.6))))
    assert b == pytest.approx(0.6), "if the model ever runs low the correction must push up"


def test_real_accuracy_file_if_present():
    import json
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "sabersim_accuracy.json")
    if not os.path.exists(f):
        pytest.skip("no accuracy file in the checkout")
    b, m = bcorr.recent_bias(json.load(open(f)), season=2026)
    assert -2.0 <= b <= 2.0
    assert m["rows"] >= 0
