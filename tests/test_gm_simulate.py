"""Bracket and seeding tests, using scores chosen so the right answer is known in advance.

A simulator handed a broken bracket does not crash, it returns plausible probabilities. So these
drive it with deterministic scores where the champion is decidable by hand, which is the only way
a bye or a reseeding mistake shows up.
"""
import os, sys
import numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm.config import load
from gm import simulate as sim

REAL = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")


def mini(n_teams=8, reg_weeks=2, playoff_teams=6, playoff_weeks=(3, 4, 5)):
    """A minimal valid league: round-robin regular season, then a bracket."""
    ids = [str(i) for i in range(n_teams)]
    rot, fixed, ms = ids[1:], ids[0], []
    for wk in range(1, reg_weeks + 1):
        o = [fixed] + rot
        for a, b in zip(o[: n_teams // 2], o[: n_teams // 2 - 1: -1]):
            ms.append({"week": wk, "home": a, "away": b})
        rot = rot[1:] + rot[:1]
    return load({
        "schema_version": 1, "league_id": "mini", "season": 2026,
        "roster": {"starters": [{"slot": "QB", "count": 1}], "flex_eligibility": {}},
        "scoring": {"per_stat": {"pass_yds": 0.04}},
        "schedule": {"regular_season_weeks": list(range(1, reg_weeks + 1)),
                     "playoff_weeks": list(playoff_weeks),
                     "championship_weeks": [playoff_weeks[-1]],
                     "playoff_teams": playoff_teams,
                     "first_round_byes": (1 << (playoff_teams - 1).bit_length()) - playoff_teams},
        "teams": [{"team_id": i, "name": f"T{i}", "record": {"w": 0, "l": 0, "t": 0},
                   "points_for": 0.0, "points_against": 0.0, "roster": []} for i in ids],
        "matchups": ms,
    })


def flat(cfg, n_cols, per_team, n_sims=1):
    """Every team scores the same amount every week: outcomes become deterministic."""
    return np.tile(np.asarray(per_team, float)[None, :, None], (n_sims, 1, n_cols))


# ---------- bracket construction ----------

def test_bracket_order_is_the_standard_seeding():
    assert sim.bracket_order(2) == [1, 2]
    assert sim.bracket_order(4) == [1, 4, 2, 3]
    assert sim.bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    assert sim.bracket_order(16)[:4] == [1, 16, 8, 9]


def test_every_seed_appears_once():
    for size in (2, 4, 8, 16, 32):
        assert sorted(sim.bracket_order(size)) == list(range(1, size + 1))


# ---------- the strongest team must win ----------

def test_best_team_always_wins_when_scores_are_deterministic():
    c = mini()
    need, _ = sim.weeks_needed(c)
    s = flat(c, need, [100 - 5 * i for i in range(8)], n_sims=50)
    r = sim.run(c, s)
    assert r["teams"]["0"]["p_title"] == 1.0
    assert r["teams"]["0"]["p_playoffs"] == 1.0


def test_worst_teams_never_reach_the_playoffs():
    c = mini()
    need, _ = sim.weeks_needed(c)
    r = sim.run(c, flat(c, need, [100 - 5 * i for i in range(8)], n_sims=50))
    assert r["teams"]["7"]["p_playoffs"] == 0.0
    assert r["teams"]["6"]["p_playoffs"] == 0.0


# ---------- byes ----------

def test_top_seeds_skip_the_first_playoff_round():
    """Seed 1 scores zero in the first playoff week. With a bye it does not matter."""
    c = mini(playoff_teams=6, playoff_weeks=(3, 4, 5))
    need, future = sim.weeks_needed(c)
    s = flat(c, need, [100 - 5 * i for i in range(8)], n_sims=20)
    s[:, 0, len(future)] = 0.0            # team 0 (seed 1) is shut out in playoff round 1
    r = sim.run(c, s)
    assert r["teams"]["0"]["p_title"] == 1.0, "a bye means round one cannot eliminate the top seed"


def test_a_team_without_a_bye_is_eliminated_by_a_shutout():
    c = mini(playoff_teams=6, playoff_weeks=(3, 4, 5))
    need, future = sim.weeks_needed(c)
    s = flat(c, need, [100 - 5 * i for i in range(8)], n_sims=20)
    s[:, 2, len(future)] = 0.0            # team 2 is seed 3, which plays in round one
    r = sim.run(c, s)
    assert r["teams"]["2"]["p_title"] == 0.0
    assert r["teams"]["2"]["p_playoffs"] == 1.0, "it still made the playoffs, it just lost"


def test_bye_count_matches_the_bracket():
    assert mini(playoff_teams=6).raw["schedule"]["first_round_byes"] == 2
    assert mini(playoff_teams=4, playoff_weeks=(3, 4)).raw["schedule"]["first_round_byes"] == 0


# ---------- seeding ----------

def test_record_outranks_points_for():
    """A 1-0 team with few points seeds above a 0-1 team with many -- the live standings rule."""
    c = mini(n_teams=4, reg_weeks=1, playoff_teams=2, playoff_weeks=(2,))
    c.raw["teams"][0]["record"] = {"w": 1, "l": 0, "t": 0}; c.raw["teams"][0]["points_for"] = 10.0
    c.raw["teams"][1]["record"] = {"w": 0, "l": 1, "t": 0}; c.raw["teams"][1]["points_for"] = 999.0
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    r = sim.run(c, flat(c, need, [0, 0, 0, 0], n_sims=10), played_weeks=[1])
    assert r["teams"]["0"]["exp_seed"] < r["teams"]["1"]["exp_seed"]


def test_points_for_breaks_a_tied_record():
    c = mini(n_teams=4, reg_weeks=1, playoff_teams=2, playoff_weeks=(2,))
    for i, pf in ((0, 50.0), (1, 500.0)):
        c.raw["teams"][i]["record"] = {"w": 1, "l": 0, "t": 0}; c.raw["teams"][i]["points_for"] = pf
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    r = sim.run(c, flat(c, need, [0, 0, 0, 0], n_sims=10), played_weeks=[1])
    assert r["teams"]["1"]["exp_seed"] < r["teams"]["0"]["exp_seed"]


# ---------- probabilities have to add up ----------

def test_probabilities_are_coherent_on_the_real_league():
    c = load(REAL)
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    s = sim.shrunk_team_scores(c, 4000, need, rng=np.random.default_rng(3))
    r = sim.run(c, s, played_weeks=[1])
    assert sum(v["p_title"] for v in r["teams"].values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(v["p_playoffs"] for v in r["teams"].values()) == pytest.approx(6.0, abs=1e-9)
    assert all(0 <= v["p_title"] <= v["p_playoffs"] <= 1 for v in r["teams"].values())


def test_every_sim_produces_exactly_one_champion_from_the_playoff_field():
    c = load(REAL)
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    r = sim.run(c, sim.shrunk_team_scores(c, 2000, need, rng=np.random.default_rng(5)),
                played_weeks=[1])
    champ, made = r["champion"], r["made"]
    assert champ.min() >= 0, "no simulation may end with an empty chair as champion"
    assert made[np.arange(len(champ)), champ].all(), "the champion must have made the playoffs"


def test_seeds_are_a_permutation_in_every_sim():
    c = load(REAL)
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    r = sim.run(c, sim.shrunk_team_scores(c, 500, need, rng=np.random.default_rng(1)),
                played_weeks=[1])
    s = np.sort(r["seeds"], axis=1)
    assert (s == np.arange(len(c.teams))[None, :]).all()


# ---------- shape contract ----------

def test_wrong_number_of_week_columns_is_rejected():
    c = mini()
    need, _ = sim.weeks_needed(c)
    with pytest.raises(ValueError, match="week columns"):
        sim.run(c, flat(c, need - 1, [1] * 8, n_sims=3))


def test_wrong_number_of_teams_is_rejected():
    c = mini()
    need, _ = sim.weeks_needed(c)
    with pytest.raises(ValueError, match="teams"):
        sim.run(c, np.zeros((3, 7, need)))


def test_played_weeks_shortens_what_must_be_supplied():
    c = mini(reg_weeks=4, playoff_weeks=(5, 6, 7))
    assert sim.weeks_needed(c)[0] == 4 + 3
    assert sim.weeks_needed(c, played_weeks=[1, 2])[0] == 2 + 3


def test_same_seed_gives_the_same_answer():
    c = load(REAL)
    need, _ = sim.weeks_needed(c, played_weeks=[1])
    a = sim.run(c, sim.shrunk_team_scores(c, 1000, need, rng=np.random.default_rng(11)), played_weeks=[1])
    b = sim.run(c, sim.shrunk_team_scores(c, 1000, need, rng=np.random.default_rng(11)), played_weeks=[1])
    assert a["teams"] == b["teams"]
