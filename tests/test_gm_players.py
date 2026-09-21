"""Player estimates and the lineup optimiser.

The lineup tests matter most: a roster is worth the best nine it can field in a week, not the sum
of its parts, and getting that wrong would misprice every trade involving depth.
"""
import os, sys
import numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm.config import load
from gm import players as P
from gm import simulate as sim

REAL = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
pytestmark = pytest.mark.skipif(not os.path.exists(REAL), reason="kuhn_2026.json not built")


def stub(cfg, means):
    """{(team, player): est} with fixed means and no randomness, so lineups are decidable."""
    out = {}
    for t in cfg.teams:
        for e in t["roster"]:
            out[(t["team_id"], e["player_id"])] = {
                "name": e["name"], "pos": e["pos"], "mean": means.get(e["pos"], 10.0),
                "sd": 1e-9, "p_play": 1.0, "source": "stub"}
    return out


# ---------- the lineup is the best available, not the whole roster ----------

def test_team_score_is_the_starting_lineup_not_the_roster():
    c = load(REAL)
    est = stub(c, {"QB": 10, "RB": 10, "WR": 10, "TE": 10, "K": 10, "DST": 10})
    s = sim.player_team_scores(c, est, 8, 2, rng=np.random.default_rng(0), mean_se=0.0)
    # 9 starters at 10 points each; the 7-man bench must not contribute
    assert s.mean() == pytest.approx(90.0, abs=0.01)


def test_flex_takes_the_best_leftover_across_eligible_positions():
    c = load(REAL)
    est = stub(c, {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0})
    tid = c.teams[0]["team_id"]
    tes = [k for k, v in est.items() if k[0] == tid and v["pos"] == "TE"]
    assert len(tes) >= 2, "need a spare TE to test the flex"
    for k in tes[:2]:
        est[k]["mean"] = 50.0                      # TE1 starts at TE, TE2 must take the flex
    s = sim.player_team_scores(c, est, 4, 1, rng=np.random.default_rng(0), mean_se=0.0)
    assert s[:, 0, 0].mean() == pytest.approx(100.0, abs=0.01)


def test_depth_beyond_the_flex_is_worth_nothing():
    c = load(REAL)
    base = stub(c, {p: 0 for p in ("QB", "RB", "WR", "TE", "K", "DST")})
    tid = c.teams[0]["team_id"]
    wrs = [k for k, v in base.items() if k[0] == tid and v["pos"] == "WR"]
    if len(wrs) < 4:
        pytest.skip("team does not carry four receivers")
    a = {k: dict(v) for k, v in base.items()}
    for k in wrs[:3]:                              # WR1, WR2 start; WR3 takes the flex
        a[k]["mean"] = 30.0
    b = {k: dict(v) for k, v in a.items()}
    b[wrs[3]]["mean"] = 30.0                       # a fourth 30-point receiver adds nothing
    sa = sim.player_team_scores(c, a, 4, 1, rng=np.random.default_rng(1), mean_se=0.0)[:, 0, 0].mean()
    sb = sim.player_team_scores(c, b, 4, 1, rng=np.random.default_rng(1), mean_se=0.0)[:, 0, 0].mean()
    assert sb == pytest.approx(sa, abs=0.01), "a fourth starter-quality WR has nowhere to play"


def test_an_absent_player_scores_nothing_and_the_next_man_plays():
    c = load(REAL)
    est = stub(c, {p: 0 for p in ("QB", "RB", "WR", "TE", "K", "DST")})
    tid = c.teams[0]["team_id"]
    qbs = [k for k, v in est.items() if k[0] == tid and v["pos"] == "QB"]
    est[qbs[0]].update(mean=40.0, p_play=0.0)      # the starter never plays
    s = sim.player_team_scores(c, est, 200, 1, rng=np.random.default_rng(2), mean_se=0.0)
    assert s[:, 0, 0].mean() == pytest.approx(0.0, abs=0.01)


# ---------- estimates ----------

def test_every_rostered_player_gets_an_estimate():
    c = load(REAL)
    est = P.ros_estimates(c)
    assert len(est) == sum(len(t["roster"]) for t in c.teams)
    assert all(v["sd"] > 0 and 0 <= v["p_play"] <= 1 for v in est.values())


def test_sources_cover_the_fallback_chain():
    est = P.ros_estimates(load(REAL))
    srcs = {v["source"] for v in est.values()}
    assert "blend" in srcs and "kdst_flat" in srcs
    assert srcs <= {"blend", "current_only", "prior_only", "replacement", "kdst_flat"}


def test_a_player_with_no_snaps_this_year_is_marked_likely_hurt():
    est = P.ros_estimates(load(REAL))
    absent = [v for v in est.values() if v["source"] == "prior_only"]
    if not absent:
        pytest.skip("nobody on a roster is currently absent")
    assert all(v["p_play"] == P.P_PLAY_ABSENT for v in absent)


def test_availability_rises_with_a_players_level():
    """One flat figure benched real starters a fifth of the time; it has to follow the player."""
    est = P.ros_estimates(load(REAL))
    skill = [v for v in est.values() if v["pos"] in P.SKILL and v["source"] != "prior_only"]
    big = max(skill, key=lambda v: v["mean"]); small = min(skill, key=lambda v: v["mean"])
    assert big["p_play"] > small["p_play"]
    assert big["p_play"] <= P.P_PLAY_CAP, "nobody is certain to play"


def test_a_lineup_is_named_before_the_week_not_after():
    """Starting the best AVAILABLE player by projection, never the one who turned out best.

    Sorting realized points is hindsight, and it pays a team for bench depth it could not have
    known to start -- it valued a benched third quarterback at 26 season points.
    """
    c = load(REAL)
    est = stub(c, {p: 0 for p in ("QB", "RB", "WR", "TE", "K", "DST")})
    tid = c.teams[0]["team_id"]
    qbs = [k for k, v in est.items() if k[0] == tid and v["pos"] == "QB"]
    if len(qbs) < 2:
        pytest.skip("team carries one quarterback")
    est[qbs[0]].update(mean=20.0, sd=1e-9, p_play=1.0)     # always starts on projection
    est[qbs[1]].update(mean=1.0, sd=40.0, p_play=1.0)      # wild, but never the named starter
    s = sim.player_team_scores(c, est, 4000, 1, rng=np.random.default_rng(3), mean_se=0.0)
    assert s[:, 0, 0].mean() == pytest.approx(20.0, abs=0.5), \
        "a high-variance bench QB must not be retroactively started"


def test_variance_grows_with_the_player():
    c = load(REAL)
    est = P.ros_estimates(c)
    skill = [v for v in est.values() if v["pos"] in P.SKILL]
    big = max(skill, key=lambda v: v["mean"]); small = min(skill, key=lambda v: v["mean"])
    assert big["sd"] > small["sd"], "absolute spread must grow with scoring level"
    assert big["sd"] / max(big["mean"], .1) < small["sd"] / max(small["mean"], .1), \
        "relative spread must shrink -- this is why a bad team wants stars"


def test_shrinkage_pulls_toward_the_positional_mean():
    c = load(REAL)
    raw = P.ros_estimates(c, mean_shrink=1.0)
    cal = P.ros_estimates(c, mean_shrink=0.7)
    spread_raw = np.std([v["mean"] for v in raw.values() if v["pos"] == "WR"])
    spread_cal = np.std([v["mean"] for v in cal.values() if v["pos"] == "WR"])
    assert spread_cal < spread_raw


def test_calibrated_dispersion_matches_the_measured_league():
    """Between- and within-team spread must match what 490 team-weeks say, not just look sane."""
    c = load(REAL)
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    s = sim.player_team_scores(c, P.ros_estimates(c), 4000, need, rng=np.random.default_rng(0))
    between = s.mean(axis=(0, 2)).std()
    within = (s - s.mean(axis=(0, 2), keepdims=True)).std()
    assert 7.5 <= between <= 10.0, f"between-team sd {between:.2f}, measured target 8.7"
    assert 19.5 <= within <= 24.0, f"within-team sd {within:.2f}, measured target 21.7"


# ---------- uncertainty about a team, not just within a week ----------

def test_parameter_uncertainty_is_a_season_long_offset_not_weekly_noise():
    """A team's true strength is fixed within a season; we just do not know it."""
    c = load(REAL)
    est = P.ros_estimates(c)
    s = sim.player_team_scores(c, est, 3000, 6, rng=np.random.default_rng(0), mean_se=25.0)
    # a per-season offset correlates a team's weeks with each other; per-week noise would not
    a, b = s[:, 0, 0], s[:, 0, 1]
    assert np.corrcoef(a, b)[0, 1] > 0.3, "weeks of the same simulated season must move together"


def test_uncertainty_flattens_the_title_race():
    c = load(REAL)
    est = P.ros_estimates(c)
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    sure = sim.run(c, sim.player_team_scores(c, est, 6000, need, rng=np.random.default_rng(0),
                                             mean_se=0.0), played_weeks=[1])
    humble = sim.run(c, sim.player_team_scores(c, est, 6000, need, rng=np.random.default_rng(0),
                                               mean_se=10.0), played_weeks=[1])
    top_sure = max(v["p_title"] for v in sure["teams"].values())
    top_humble = max(v["p_title"] for v in humble["teams"].values())
    assert top_humble < top_sure, "admitting we might be wrong must reduce the favourite's odds"


def test_default_uncertainty_is_the_measured_disagreement():
    assert 4.0 <= sim.MEAN_SE <= 6.5, "two independent views of these rosters disagree by ~5.1 pts"


def test_threshold_bonuses_are_part_of_league_scoring(monkeypatch):
    """A 100-yard bonus is what the league pays; omitting it undervalues whoever earns it."""
    import pandas as pd
    c = load(REAL)
    raw = dict(c.raw)
    raw["scoring"] = {**raw["scoring"], "bonuses": [{"stat": "rec_yds", "threshold": 100,
                                                     "points": 3}]}
    from gm.config import LeagueConfig
    with_bonus = LeagueConfig(raw)
    frame = pd.DataFrame([{"player_id": "x", "player_display_name": "A", "position": "WR",
                           "team": "SF", "season": 2026, "week": 1, "receiving_yards": 120.0,
                           "receptions": 5.0}])
    monkeypatch.setattr(P.pd, "read_parquet", lambda *a, **k: frame)
    plain = P.weekly_in_league_scoring(c, (2026,)).pts.iloc[0]
    bonused = P.weekly_in_league_scoring(with_bonus, (2026,)).pts.iloc[0]
    assert bonused == pytest.approx(plain + 3.0)
