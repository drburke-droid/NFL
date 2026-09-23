"""The shape features are the part of game_shape_study.py that can be silently wrong.

Every number the study reports is a mean over these columns, so a share that can exceed 1, a
padded zero row inflating the player count, or a spread that keeps the home team's sign would
change every conclusion without raising anything. These pin that down on a hand-built game.
"""
import os, sys
import numpy as np, pandas as pd, pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from game_shape_study import team_game_shapes, FEATS

COLS = ["completions", "attempts", "carries", "targets", "receptions",
        "passing_yards", "rushing_yards", "receiving_yards"]


def player(pid, pos, team, ppr, **kw):
    r = dict(player_id=pid, position=pos, team=team, fantasy_points_ppr=ppr,
             season=2024, week=1, season_type="REG", game_id="2024_01_AAA_BBB")
    for c in COLS:
        r[c] = kw.get(c, 0.0)
    return r


@pytest.fixture
def frames():
    pw = pd.DataFrame([
        # home team BBB: a QB, two RBs, two WRs, a TE
        player("q1", "QB", "BBB", 20.0, attempts=30, completions=20, passing_yards=250),
        player("r1", "RB", "BBB", 15.0, carries=20, rushing_yards=100),
        player("r2", "RB", "BBB", 5.0, carries=5, rushing_yards=20),
        player("w1", "WR", "BBB", 25.0, targets=10, receptions=8, receiving_yards=120),
        player("w2", "WR", "BBB", 10.0, targets=5, receptions=4, receiving_yards=40),
        player("t1", "TE", "BBB", 5.0, targets=3, receptions=3, receiving_yards=20),
        # a rostered player with no action at all: must not count as a player
        player("z9", "WR", "BBB", 0.0),
        # away team AAA: a QB who threw picks into a negative line, plus one back
        player("q2", "QB", "AAA", -2.0, attempts=40, completions=18, passing_yards=150),
        player("r3", "RB", "AAA", 12.0, carries=10, rushing_yards=60),
        # a kicker and a defender: not skill positions, must be excluded
        player("k1", "K", "BBB", 9.0),
        player("d1", "CB", "AAA", 3.0),
    ])
    sch = pd.DataFrame([dict(game_id="2024_01_AAA_BBB", season=2024, week=1,
                             home_team="BBB", away_team="AAA", home_score=31, away_score=17,
                             spread_line=6.5, total_line=44.5, roof="outdoors", div_game=0)])
    return pw, sch


def test_one_row_per_team_and_totals_are_skill_only(frames):
    tg = team_game_shapes(*frames, [2024])
    assert set(tg.team) == {"AAA", "BBB"}
    b = tg.set_index("team").loc["BBB"]
    # 20+15+5+25+10+5 = 80; the kicker's 9 and the idle WR's 0 are not in it
    assert b.tot == pytest.approx(80.0)
    assert b.qb == pytest.approx(20.0) and b.rb == pytest.approx(20.0)
    assert b.wr == pytest.approx(35.0) and b.te == pytest.approx(5.0)


def test_idle_players_do_not_pad_the_counts(frames):
    tg = team_game_shapes(*frames, [2024])
    b = tg.set_index("team").loc["BBB"]
    # 20, 15, 25 and 10 clear the bar; the two 5s and the idle WR do not
    assert b.n10 == 4
    assert b.n20 == 2                                  # 20 and 25
    # top1 is a share of the 80, so the idle 0.0 row cannot dilute it
    assert b.top1 == pytest.approx(25.0 / 80.0)
    # and the idle row is genuinely gone rather than scored as a zero
    assert b.hhi == pytest.approx(sum((x / 80.0) ** 2 for x in (20, 15, 5, 25, 10, 5)))


def test_shares_sum_to_one_and_stay_bounded_with_negative_lines(frames):
    tg = team_game_shapes(*frames, [2024])
    for _, r in tg.iterrows():
        assert sum(r[c] for c in ("qb_sh", "rb_sh", "wr_sh", "te_sh")) == pytest.approx(1.0)
        # AAA's QB is negative, so qb_sh is negative and tot is small -- but a SHARE-of-points
        # measure built off clipped values must never exceed 1
        assert 0.0 <= r.top1 <= 1.0
        assert 0.0 <= r.hhi <= 1.0


def test_spread_is_flipped_to_each_team(frames):
    tg = team_game_shapes(*frames, [2024]).set_index("team")
    # schedule says home (BBB) favoured by 6.5
    assert tg.loc["BBB", "spread"] == pytest.approx(6.5)
    assert tg.loc["AAA", "spread"] == pytest.approx(-6.5)


def test_margin_and_total_come_from_the_box_score(frames):
    tg = team_game_shapes(*frames, [2024]).set_index("team")
    assert tg.loc["BBB", "margin"] == 14 and tg.loc["AAA", "margin"] == -14
    assert (tg.game_total == 48).all()


def test_pass_share_uses_attempts_over_plays(frames):
    tg = team_game_shapes(*frames, [2024]).set_index("team")
    assert tg.loc["BBB", "pass_share"] == pytest.approx(30 / 55)   # 30 att, 25 car
    assert tg.loc["AAA", "pass_share"] == pytest.approx(40 / 50)


def test_every_clustering_feature_is_present_and_finite(frames):
    tg = team_game_shapes(*frames, [2024])
    for f in FEATS:
        assert f in tg.columns, f
        assert np.isfinite(tg[f]).all(), f
