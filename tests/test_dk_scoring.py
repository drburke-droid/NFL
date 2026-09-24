"""DraftKings scoring, and the promise that the page and the grader score in the same points.

The comparison with the sites only means something if three copies of the same rules agree: the
grader's (scripts/dk_scoring.py), the live card's (docs/fan.html), and the arrow steps shared by the
page and scripts/fan_rules.py. The parity tests below read the constants straight out of the page.
"""
import os, re, sys, json
import numpy as np, pandas as pd, pytest
from scipy.stats import gamma

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dk_scoring as dk
import fan_rules

PAGE = open(os.path.join(ROOT, "docs", "fan.html"), encoding="utf-8").read()


def js_object(name):
    """`const NAME = {a:1, b:2}` in the page -> {"a": 1.0, "b": 2.0}"""
    m = re.search(r"const %s = \{([^}]*)\}" % re.escape(name), PAGE)
    assert m, f"{name} not found in docs/fan.html"
    return {k.strip(): float(v) for k, v in (kv.split(":") for kv in m.group(1).split(","))}


def test_page_scores_in_the_same_points_as_the_grader():
    assert js_object("DK") == dk.LINEAR
    assert js_object("DK_BONUS_AT") == dk.BONUS_AT
    assert js_object("DK_SHAPE") == dk.SHAPE
    assert float(re.search(r"DK_BONUS_PTS = ([\d.]+)", PAGE).group(1)) == dk.BONUS_PTS


def test_page_and_recorder_mean_the_same_thing_by_a_press():
    assert js_object("STEP_U") == fan_rules.STEP_U
    assert re.search(r'const RULE = "(\w+)"', PAGE).group(1) in fan_rules.RULES


def test_linear_scoring_is_draftkings():
    # 250 pass yds (10) + 2 pass TD (8) + 1 INT (-1) + 20 rush yds (2) + 1 fumble (-1) = 18
    line = {"pass_yds": 250, "pass_tds": 2, "pass_int": 1, "rush_yds": 20, "fumbles_lost": 1}
    lin = sum(line[k] * dk.LINEAR[k] for k in line)
    assert lin == pytest.approx(18.0)
    # the bonus expectations ride on top: real for a 250-yard passer, a rounding error for 20 rush yds
    ev = 3 * (float(dk.p_bonus("pass_yds", 250)) + float(dk.p_bonus("rush_yds", 20)))
    assert dk.projected_points(line) == pytest.approx(lin + ev)
    assert 0.1 < ev < 1.0


def test_an_interception_costs_one_point_not_two():
    assert dk.LINEAR["pass_int"] == -1.0 and dk.LINEAR["fumbles_lost"] == -1.0


def test_bonus_probability_behaves():
    for stat in dk.BONUS_AT:
        p = dk.p_bonus(stat, np.array([0.0, 10, 50, 100, 150, 250, 400]))
        assert p[0] == 0.0
        assert np.all(np.diff(p) > 0), stat                   # more yards projected, likelier bonus
        assert np.all((p >= 0) & (p < 1))
    # the case that motivated expectations over a step: an 85-yard receiver is a real bonus threat
    assert 0.25 < float(dk.p_bonus("rec_yds", 85)) < 0.40
    assert float(dk.p_bonus("pass_yds", 300)) == pytest.approx(gamma.sf(300, 10.9, scale=300 / 10.9))


def test_actual_points_from_a_box_score():
    a = pd.DataFrame([
        # 310 pass yds, 3 TD, 1 INT, a 2-pt pass, a sack fumble lost: 12.4+12-1+2-1 +3 bonus = 27.4
        dict(position="QB", passing_yards=310, passing_tds=3, passing_interceptions=1,
             passing_2pt_conversions=1, fumbles_lost_total=1),
        # 8-104-1 receiving plus a kick-return TD: 8+10.4+6+6 +3 bonus = 33.4
        dict(position="WR", receptions=8, receiving_yards=104, receiving_tds=1, special_teams_tds=1),
        # 99 rush yds: one short of the bonus, so exactly 9.9
        dict(position="RB", rushing_yards=99),
    ])
    assert list(dk.actual_frame(a).round(2)) == [27.4, 33.4, 9.9]


def test_projection_frame_matches_the_scalar():
    rows = [{"pass_yds": 280.0, "pass_tds": 1.9, "pass_int": 0.7, "rush_yds": 20.0, "rush_tds": 0.2, "fumbles_lost": 0.1},
            {"rec": 6.8, "rec_yds": 82.4, "rec_tds": 0.6}]
    df = pd.DataFrame(rows)
    assert list(dk.projected_frame(df).round(6)) == [round(dk.projected_points(r), 6) for r in rows]
