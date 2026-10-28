"""League Values tab: which games count as full games, and what a keeper's remaining years are worth.

Omar Cooper Jr. is the case the tab was asked for: hurt after one catch in week 1 (7% of the snaps),
a $1 waiver add with three keeper years left. A backup who always plays a fifth of the snaps must
keep every game; a starter carted off early must lose that one.
"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import league_values_bake as V

# a dollar curve where $ = 2 x (ppg - 5) above 5 ppg, flat $1 below
CURVE = {"WR": ([0.0, 5.0, 30.0], [1.0, 1.0, 51.0])}


def _frames(rows):
    snaps = pd.DataFrame([{"season": 2026, "game_type": "REG", "pfr_player_id": p, "week": w, "offense_pct": pct}
                          for p, w, pct in rows])
    weekly = pd.DataFrame([{"player_id": "g" + p, "season": 2026, "week": w} for p, w, _ in rows])
    ids = pd.DataFrame({"pfr_id": sorted({p for p, _, _ in rows}), "gsis_id": ["g" + p for p in sorted({p for p, _, _ in rows})]})
    return weekly, snaps, ids


def test_an_injury_exit_is_a_partial_game_and_a_backup_role_is_not():
    weekly, snaps, ids = _frames([("coop", 1, 0.07), ("coop", 6, 0.82), ("coop", 7, 0.64),
                                  ("back", 1, 0.18), ("back", 2, 0.22), ("back", 3, 0.20)])
    assert V.partial_games(weekly, 2026, snaps, ids) == {("gcoop", 1)}


def test_no_snap_file_drops_nothing():
    weekly, _, ids = _frames([("coop", 1, 0.07)])
    assert V.partial_games(weekly, 2026, None, ids) == set()


def test_every_remaining_keeper_year_is_valued_and_a_cheap_young_player_beats_an_expensive_one():
    ages = [23, 24, 25]
    cheap = V.level_dist_value(CURVE, "WR", 15.0, ages, [6.0, 11.0, 16.0], 3)
    pricey = V.level_dist_value(CURVE, "WR", 15.0, ages, [66.0, 71.0, 76.0], 3)
    assert [k for k, _, _ in cheap] == [1, 2, 3]
    assert sum(v for _, v, _ in cheap) > 20 > sum(v for _, v, _ in pricey)
    assert all(v >= 0 for _, v, _ in pricey)       # an owner never keeps at a loss
    assert cheap[0][2] == pricey[0][2] > 0         # the market price does not depend on what keeping costs
    assert V.level_dist_value(CURVE, "WR", 15.0, [], [], 0) == []


def test_age_is_taken_on_the_first_of_september():
    assert V.age_at("2003-12-14", 2026) == 22      # Cooper: 22 this season, 23 next
    assert V.age_at("2003-08-31", 2026) == 23
    assert V.age_at(None, 2026) is None


def test_the_league_price_curve_is_fitted_to_its_own_bids_and_is_monotone(tmp_path):
    """Built from a draft file of real bids against the previous season's ppg. Two QBs and two WRs
    are below min_bids, so this checks the plumbing on the real file instead when it exists."""
    from gm import config, keepers as kp
    league = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
    drafts = os.path.join(ROOT, "outputs", "espn_drafts.json")
    if not (os.path.exists(league) and os.path.exists(drafts)):
        import pytest; pytest.skip("league files not present")
    try:
        c = kp.league_price_curve(config.load(league), drafts)
    except Exception as ex:                       # nflverse unreachable: nothing to check offline
        import pytest; pytest.skip(f"nflverse unavailable: {ex}")
    for pos, (xs, ys) in c.items():
        assert list(ys) == sorted(ys), pos
    if "QB" in c:                                  # the reason it exists: this league pays little for QBs
        assert kp.dollars_at(c, "QB", 24) < kp.dollars_at(kp.dollar_curve(), "QB", 24)
    assert kp.league_price_curve(config.load(league), str(tmp_path / "missing.json")) == {}
