"""The rolling bias must be conservative: silent on thin history, capped, skill-only, and blind to
grades that only carry the old all-rows number."""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import bias_correction as bcorr


def wk(week, **pos):
    """pos like QB=(63, -0.1) -> a week block shaped like docs/sabersim_accuracy.json after 2026-09-24:
    n_played rows and the played-only median error. The all-rows fields are present but must be ignored."""
    return {"week": week, "by_pos": {p: {"n": n + 40, "bias": -9.0, "n_played": n, "med_err_played": b}
                                     for p, (n, b) in pos.items()}}


def acc(*weeks, season=2026):
    return {"season": season, "weeks": list(weeks)}


def test_weights_by_played_row_count_across_weeks():
    a = acc(wk(1, WR=(100, -1.0)), wk(2, WR=(300, -2.0)))
    b, m = bcorr.recent_bias(a)
    assert b == pytest.approx((100 * -1.0 + 300 * -2.0) / 400)
    assert m["rows"] == 400 and [x["week"] for x in m["weeks"]] == [1, 2]
    assert m["basis"] == "med_err_played"


def test_old_all_rows_grades_are_skipped_not_used():
    """A block from before the grader emitted med_err_played carries only n/bias: it must not
    feed the -9.0 in, and the meta must say why nothing was applied."""
    old = {"week": 1, "by_pos": {"WR": {"n": 500, "bias": -9.0}}}
    b, m = bcorr.recent_bias(acc(old))
    assert b == 0.0 and m["rows"] == 0 and m["skipped_weeks"] == [1]
    assert "graded before" in m["reason"]
    b, m = bcorr.recent_bias(acc(old, wk(2, WR=(400, -0.5))))
    assert b == pytest.approx(-0.5) and m["skipped_weeks"] == [1]


def test_thin_history_corrects_nothing():
    b, m = bcorr.recent_bias(acc(wk(1, WR=(50, -3.0))))
    assert b == 0.0 and "50 graded played rows" in m["reason"]


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
        {"week": 1, "by_pos": {"WR": {"n_played": 400, "med_err_played": -1.0},
                               "RB": {"n_played": None, "med_err_played": -9.0},
                               "TE": {"n_played": 50}, "QB": "nonsense"}},
        {"no_week_key": True}, None,
    ]}
    b, m = bcorr.recent_bias(a)
    assert b == pytest.approx(-1.0) and m["rows"] == 400


def test_a_positive_error_is_returned_as_is():
    b, _ = bcorr.recent_bias(acc(wk(1, WR=(400, +0.6))))
    assert b == pytest.approx(0.6), "if the send ever runs low the correction must push up"


def test_real_accuracy_file_if_present():
    import json
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "sabersim_accuracy.json")
    if not os.path.exists(f):
        pytest.skip("no accuracy file in the checkout")
    b, m = bcorr.recent_bias(json.load(open(f)), season=2026)
    assert -2.0 <= b <= 2.0
    assert m["rows"] >= 0
