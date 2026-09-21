"""The validator's job is to reject configs that would make a simulator lie rather than crash."""
import copy, json, os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm.config import load, validate, LeagueConfigError, SCHEMA_VERSION

EXAMPLE = os.path.join(ROOT, "gm", "leagues", "example_keeper.json")


@pytest.fixture
def raw():
    return json.load(open(EXAMPLE))


def bad(raw, msg_fragment):
    with pytest.raises(LeagueConfigError) as e:
        validate(raw)
    assert msg_fragment in str(e.value), f"expected {msg_fragment!r} in {e.value!r}"


# ---------- the shipped example must stay valid ----------

def test_example_league_is_valid():
    c = load(EXAMPLE)
    assert c.starting_size == 9 and len(c.teams) == 10


# ---------- scoring ----------

def test_scoring_matches_the_pipelines_stat_names():
    c = load(EXAMPLE)
    # 8 rec + 11.2 rec_yds + 6 rec_td + 3 hundred-yard bonus
    assert c.score({"rec": 8, "rec_yds": 112, "rec_tds": 1}) == pytest.approx(28.2)


def test_scoring_ignores_unknown_stats_and_missing_ones():
    c = load(EXAMPLE)
    assert c.score({"rec": 2, "nonsense": 999}) == pytest.approx(2.0)
    assert c.score({}) == 0.0


def test_bonus_only_applies_at_the_threshold():
    c = load(EXAMPLE)
    assert c.score({"rush_yds": 99}) == pytest.approx(9.9)
    assert c.score({"rush_yds": 100}) == pytest.approx(13.0)


# ---------- the keeper clock ----------

def test_keeper_clock_expires_and_cost_escalates():
    c = load(EXAMPLE)
    fresh = {"player_id": "x", "keeper_cost": {"round": 9}, "times_kept": 0}
    late = {"player_id": "y", "keeper_cost": {"round": 4}, "times_kept": 2}
    done = {"player_id": "z", "keeper_cost": {"round": 2}, "times_kept": 3}
    assert c.keeper_cost(fresh) == {"round": 8}
    assert c.keeper_cost(late) == {"round": 3}
    assert c.keeper_eligible(done) is False and c.keeper_cost(done) is None


def test_keeper_round_never_goes_below_one():
    c = load(EXAMPLE)
    assert c.keeper_cost({"player_id": "a", "keeper_cost": {"round": 1}, "times_kept": 0}) == {"round": 1}


def test_expiring_next_year_finds_the_last_keeper_year(raw):
    c = load(raw)
    got = {x["player_id"] for x in c.expiring_next_year()}
    assert got == {"00-0036355"}, "only the player at max_times_kept - 1 expires next season"


def test_auction_escalation(raw):
    raw["keepers"].update({"cost_model": "auction", "auction_escalation_pct": 20})
    raw["teams"][0]["roster"] = [{"player_id": "p", "keeper_cost": {"price": 30}, "times_kept": 1}]
    raw["teams"][1]["roster"] = []
    c = load(raw)
    assert c.keeper_cost(c.team("t1")["roster"][0]) == {"price": 36.0}


def test_times_kept_beyond_the_limit_is_rejected(raw):
    raw["teams"][0]["roster"][0]["times_kept"] = 4
    bad(raw, "times_kept 4 outside 0..3")


# ---------- the checks that stop silent nonsense ----------

def test_playoff_teams_cannot_exceed_the_league(raw):
    raw["schedule"]["playoff_teams"] = 99
    bad(raw, "playoff_teams: 99 with only 10 teams")


def test_bracket_with_an_odd_first_round_is_rejected(raw):
    raw["schedule"].update({"playoff_teams": 6, "first_round_byes": 1})
    bad(raw, "odd first round")


def test_regular_and_playoff_weeks_cannot_overlap(raw):
    raw["schedule"]["playoff_weeks"] = [14, 15, 16]
    bad(raw, "both regular season and playoff")


def test_championship_week_must_be_a_playoff_week(raw):
    raw["schedule"]["championship_weeks"] = [18]
    bad(raw, "championship_weeks")


def test_a_player_cannot_be_on_two_rosters(raw):
    raw["teams"][1]["roster"] = [dict(raw["teams"][0]["roster"][0])]
    bad(raw, "is on both")


def test_duplicate_team_ids_rejected(raw):
    raw["teams"][1]["team_id"] = raw["teams"][0]["team_id"]
    bad(raw, "team_id values must be unique")


def test_a_team_playing_twice_in_a_week_is_rejected(raw):
    raw["matchups"].append({"week": 1, "home": "t1", "away": "t3"})
    bad(raw, "playing more than once")


def test_team_cannot_play_itself(raw):
    raw["matchups"][0]["away"] = raw["matchups"][0]["home"]
    bad(raw, "plays itself")


def test_unknown_team_in_matchups(raw):
    raw["matchups"][0]["home"] = "nobody"
    bad(raw, "unknown team_id")


def test_faab_remaining_above_budget_rejected(raw):
    raw["teams"][0]["faab_remaining"] = 500
    bad(raw, "faab_remaining: 500.0 outside 0..100.0")


def test_trade_deadline_must_be_in_the_regular_season(raw):
    raw["acquisitions"]["trade_deadline_week"] = 16
    bad(raw, "trade_deadline_week: 16")


def test_flex_slot_must_be_defined(raw):
    raw["roster"]["starters"].append({"slot": "SUPERFLEX", "count": 1})
    bad(raw, "neither a position nor defined in flex_eligibility")


def test_flex_eligibility_rejects_unknown_positions(raw):
    raw["roster"]["flex_eligibility"]["FLEX"] = ["RB", "PUNTER"]
    bad(raw, "unknown position(s) ['PUNTER']")


def test_schema_version_is_checked(raw):
    raw["schema_version"] = 99
    bad(raw, "schema_version")


def test_missing_section_names_itself(raw):
    del raw["scoring"]
    bad(raw, "missing required section 'scoring'")


def test_unknown_team_lookup_raises(raw):
    c = load(raw)
    with pytest.raises(LeagueConfigError):
        c.team("nope")


# ---------- standings-conditional keeper inflation ----------

def with_inflation(raw, conditions=None):
    raw["keepers"]["inflation"] = {
        "base": 5,
        "conditions": conditions if conditions is not None else [
            {"name": "missed_playoffs", "delta": -2},
            {"name": "missed_playoffs_and_drafted_champion", "delta": -3},
        ],
    }
    return load(raw)


def test_inflation_matches_the_league_rule(raw):
    c = with_inflation(raw)
    assert c.keeper_inflation(made_playoffs=True) == 5.0
    assert c.keeper_inflation(made_playoffs=False) == 3.0
    assert c.keeper_inflation(made_playoffs=False, drafted_champion=True) == 0.0


def test_drafting_the_champion_only_helps_a_non_playoff_team(raw):
    c = with_inflation(raw)
    assert c.keeper_inflation(made_playoffs=True, drafted_champion=True) == 5.0


def test_no_inflation_block_means_no_bump():
    assert load(EXAMPLE).keeper_inflation(made_playoffs=False) == 0.0


def test_unknown_inflation_condition_is_rejected(raw):
    raw["keepers"]["inflation"] = {"base": 5, "conditions": [{"name": "vibes", "delta": -1}]}
    bad(raw, "condition 'vibes' not in")


def test_inflation_condition_needs_a_delta(raw):
    raw["keepers"]["inflation"] = {"base": 5, "conditions": [{"name": "missed_playoffs"}]}
    bad(raw, "needs a 'delta'")


def test_inflation_needs_a_base(raw):
    raw["keepers"]["inflation"] = {"conditions": []}
    bad(raw, "needs a 'base' bump")


# ---------- the real league ----------

REAL = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")


@pytest.mark.skipif(not os.path.exists(REAL), reason="kuhn_2026.json not built yet")
def test_real_league_loads_and_matches_espn():
    c = load(REAL)
    assert len(c.teams) == 12
    assert c.starting_size == 9 and c.raw["roster"]["bench"] == 7
    assert c.regular_weeks == list(range(1, 15)) and c.playoff_weeks == [15, 16, 17]
    assert c.raw["schedule"]["playoff_teams"] == 6 and c.raw["schedule"]["first_round_byes"] == 2
    assert len(c.raw["matchups"]) == 84, "12 teams x 14 weeks / 2"


@pytest.mark.skipif(not os.path.exists(REAL), reason="kuhn_2026.json not built yet")
def test_real_league_scoring_is_not_the_nflverse_default():
    """The league pays 6 for a passing TD and -1 for an interception; the grader uses 4 and -2."""
    c = load(REAL)
    ps = c.raw["scoring"]["per_stat"]
    assert ps["pass_tds"] == 6.0 and ps["pass_int"] == -1.0
    assert ps["rec"] == 1.0 and ps["fumbles_lost"] == -1.0
    # a 300/3/1 quarterback line is worth 6 more here than under the grader's scoring
    assert c.score({"pass_yds": 300, "pass_tds": 3, "pass_int": 1}) == pytest.approx(29.0)


@pytest.mark.skipif(not os.path.exists(REAL), reason="kuhn_2026.json not built yet")
def test_real_league_carries_the_keeper_clock():
    c = load(REAL)
    kept = [e for t in c.teams for e in t["roster"] if e["times_kept"]]
    assert kept, "times_kept should be derived from draft history, not left at zero"
    assert all(0 <= e["times_kept"] <= 3 for t in c.teams for e in t["roster"])
