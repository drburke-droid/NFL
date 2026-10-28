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


def test_rank_prices_are_the_rooms_kth_most_expensive_player_recent_auctions_first():
    """The top stays at the top: the k-th best player costs the k-th highest bid, so a room that pays
    $60+ for its best players prices the best projected player at $60+."""
    from gm import keepers as kp
    bids = [("WR", 0, b, 2025, "x") for b in (65, 48, 48, 32, 3)] + [("WR", 0, b, 2026, "y") for b in (65, 52, 48, 42)]
    r, p = kp.rank_prices(bids, min_bids=4)["WR"]
    assert list(r[:3]) == [1.0, 2.0, 3.0] and p[0] == 65.0
    assert abs(p[1] - (48 * 0.5 + 52 * 1.0) / 1.5) < 1e-9          # 2026 counts twice 2025 (RECENCY = 1)
    assert list(p) == sorted(p, reverse=True), "a better rank never costs less"
    assert kp.rank_prices(bids[:3], min_bids=4) == {}


def test_the_field_ages_too():
    """Prices are by rank, so a player moves only by how his drift differs from his position's."""
    from gm import keepers as kp
    young, _ = kp.drift_vs_field("WR", 23)
    old, _ = kp.drift_vs_field("WR", 29)
    assert young > 0 > old and abs(young + old) < 1e-9            # equal weights: symmetric about the field


def test_no_price_above_the_rooms_record():
    from gm import keepers as kp
    curve = {"RB": ([10.0, 25.0], [1.0, 63.0])}
    assert kp.expected_dollars(curve, "RB", 24.0, age=24, mult=1.66, cap=68.0) <= 68.0
    out = V.level_dist_value(curve, "RB", 24.0, [24], [70.0], 1, mult=1.66, cap=68.0)
    assert out[0][1] == 0.0 and out[0][2] <= 68.0                  # never worth keeping at $70 if he can't cost more than $68


def test_a_level_becomes_a_price_through_its_rank_against_this_seasons_levels(monkeypatch):
    from gm import keepers as kp
    bids = [("WR", r, b, 2024, str(r)) for r, b in ((1, 60), (2, 58), (5, 50), (6, 48), (11, 30), (12, 28), (40, 3), (41, 2))]
    monkeypatch.setattr(kp, "league_bids", lambda cfg, path=None: bids)
    levels = {"WR": [24.0, 21.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0]}
    c = kp.league_price_curve(None, levels=levels)
    assert kp.dollars_at(c, "WR", 24.0) >= kp.dollars_at(c, "WR", 18.0) >= kp.dollars_at(c, "WR", 10.0)
    assert abs(kp.dollars_at(c, "WR", 24.0) - 59.0) < 1.5                  # the best level is priced as WR1-2


def test_premiums_only_raise_a_price(monkeypatch):
    from gm import keepers as kp
    past = [("WR", 0, b, 2025, "p") for b in (65, 55, 50, 45, 40, 36, 32, 30, 28, 26, 24, 22, 3, 2)]
    bids = past + [("WR", 10.0, 65.0, 2026, "CeeDee Lamb"), ("WR", 11.0, 1.0, 2026, "Bargain Bin")]
    monkeypatch.setattr(kp, "league_bids", lambda cfg, path=None: bids)
    class C: raw = {"season": 2026}
    pr = kp.price_premium(C())
    assert pr[("ceedee lamb", "WR")] > 1.2                                   # $65 where the 10th WR costs ~$26
    assert pr[("bargain bin", "WR")] == 1.0                                  # a $1 pick carries no discount


def test_the_league_price_curve_on_the_real_league_prices_qbs_below_the_model():
    from gm import config, keepers as kp
    league = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
    if not os.path.exists(league):
        import pytest; pytest.skip("league file not present")
    try:
        c = kp.league_price_curve(config.load(league))
    except Exception as ex:                       # nflverse unreachable: nothing to check offline
        import pytest; pytest.skip(f"nflverse unavailable: {ex}")
    for pos, (xs, ys) in c.items():
        assert list(ys) == sorted(ys), pos
    if "QB" in c:                                  # this league pays little for quarterbacks
        assert kp.dollars_at(c, "QB", 22) < kp.dollars_at(kp.dollar_curve(), "QB", 22)
