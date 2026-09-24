"""What an arrow means, under each rule, from the code a fan sends to the number that gets graded.

Rule u1 changed the meaning of a press mid-season. The risk is a code locked in under one rule being
scored under the other -- a 10% boost read as +10 yards, or half a touchdown read as +5% -- so these
pin both rules end to end, including a code that carries no rule at all.
"""
import os, sys, json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fan_rules


def recorder():
    """The recorder's functions without its CLI, which runs at import and would read argv (and, given
    codes, write the long table). Everything above `codes = []` is definitions only."""
    src = open(os.path.join(ROOT, "scripts", "fan_adjust_record.py"), encoding="utf-8").read()
    head = src.split("\ncodes = []", 1)[0]
    ns = {"__file__": os.path.join(ROOT, "scripts", "fan_adjust_record.py"), "__name__": "fan_adjust_record_defs"}
    exec(compile(head, "fan_adjust_record.py", "exec"), ns)
    return ns


def test_legacy_rule_is_the_default():
    for v in (None, "", float("nan"), "pct10", "nonsense"):
        assert fan_rules.rule_of(v) == "pct10"
    assert fan_rules.rule_of("u1") == "u1"


def test_units_rule_moves_by_set_amounts_and_stops_at_zero():
    assert fan_rules.adjusted("u1", "rec_tds", 0.6, 2) == pytest.approx(1.6)      # one more TD = 2 presses
    assert fan_rules.adjusted("u1", "rec_yds", 82.35, -3) == pytest.approx(52.35)
    assert fan_rules.adjusted("u1", "pass_yds", 240, 1) == pytest.approx(265)
    assert fan_rules.adjusted("u1", "pass_int", 0.7, -5) == 0.0
    assert fan_rules.min_presses("u1", "rec_tds", 0.6) == -2                      # 0.6 -> 0.1 -> 0
    assert fan_rules.min_presses("u1", "rec_tds", 0.5) == -1                      # exactly one step
    assert fan_rules.min_presses("u1", "rush_yds", 0.0) == 0


def test_percentage_rule_is_unchanged():
    assert fan_rules.adjusted("pct10", "rec_tds", 0.6, 3) == pytest.approx(0.78)
    assert fan_rules.adjusted("pct10", "rec_yds", 80, -10) == 0.0
    assert fan_rules.min_presses("pct10", "rec_yds", 80) == -10


@pytest.fixture(scope="module")
def bake():
    """A frozen bake from the repo and one player-stat in it to press on."""
    import glob
    fp = sorted(glob.glob(os.path.join(ROOT, "docs", "fan", "proj_2026_wk3_*.json")))[-1]
    b = json.load(open(fp, encoding="utf-8"))
    si = b["stat_keys"].index("rec_tds")
    pi = next(p["i"] for p in b["players"] if (p["stats"].get("rec_tds") or 0) >= 0.3)
    return b, pi, si


def test_recorder_scores_each_code_under_its_own_rule(bake):
    b, pi, si = bake
    rows_for = recorder()["rows_for"]
    base = b["players"][pi]["stats"]["rec_tds"]
    code = {"n": "tester", "t": "2026-09-24T12:00Z", "b": b["bake_id"], "a": [[pi, si, 2]]}
    _, _, new = rows_for(2026, 3, dict(code, r="u1"))
    _, _, old = rows_for(2026, 3, code)                                            # no "r": a code from before u1
    assert new[0]["rule"] == "u1" and new[0]["adjusted"] == pytest.approx(round(base + 1.0, 2))
    assert new[0]["delta"] == pytest.approx(1.0)
    assert old[0]["rule"] == "pct10" and old[0]["adjusted"] == pytest.approx(round(base * 1.2, 2))
    assert old[0]["pct"] == pytest.approx(20.0)


def test_recorder_floors_presses_past_zero(bake):
    b, pi, si = bake
    rows_for = recorder()["rows_for"]
    base = b["players"][pi]["stats"]["rec_tds"]
    _, _, rows = rows_for(2026, 3, {"n": "t", "t": "x", "b": b["bake_id"], "r": "u1", "a": [[pi, si, -40]]})
    assert rows[0]["arrows"] == fan_rules.min_presses("u1", "rec_tds", base)
    assert rows[0]["adjusted"] == 0.0


def _sets():
    """One fan, week 3. Locks in Thursday afternoon, comes back Saturday and locks in again.
    TNF kicks off Fri 00:15Z; the Sunday games at 17:00Z."""
    import pandas as pd
    tnf, sun = pd.Timestamp("2026-09-25T00:15Z"), pd.Timestamp("2026-09-27T17:00Z")
    thu, sat = "2026-09-24T18:00Z", "2026-09-26T15:00Z"
    rows = [  # (submitted_at, game kickoff, what)
        (thu, tnf, "thu set, TNF arrow"),
        (thu, sun, "thu set, Sunday arrow he later changed"),
        (thu, sun, "thu set, Sunday arrow he later removed"),
        (sat, tnf, "sat set, TNF arrow (the page still showed it)"),
        (sat, sun, "sat set, Sunday arrow"),
        (sat, pd.NaT, "sat set, game with no kickoff on file"),
    ]
    return pd.DataFrame([dict(season=2026, week=3, fan="A", submitted_at=s, _kick=k, what=w) for s, k, w in rows])


def test_coming_back_midweek_keeps_the_early_games():
    rows = _sets()
    live, late, sup = fan_rules.live_arrows(rows)
    got = dict(zip(rows.what, zip(live, late, sup)))
    assert got["thu set, TNF arrow"] == (True, False, False)                 # the arrow the old rule lost
    assert got["sat set, Sunday arrow"] == (True, False, False)
    assert got["sat set, TNF arrow (the page still showed it)"] == (False, True, False)
    assert got["thu set, Sunday arrow he later changed"] == (False, False, True)
    assert got["thu set, Sunday arrow he later removed"] == (False, False, True)   # removed stays removed
    assert got["sat set, game with no kickoff on file"] == (True, False, False)


def test_every_arrow_is_exactly_one_of_live_late_superseded():
    live, late, sup = fan_rules.live_arrows(_sets())
    assert ((live.astype(int) + late.astype(int) + sup.astype(int)) == 1).all()


def test_fans_are_kept_apart():
    import pandas as pd
    rows = _sets()
    other = rows.assign(fan="B", submitted_at="2026-09-26T20:00Z")          # B's one set is later than all of A's
    live, _, _ = fan_rules.live_arrows(pd.concat([rows, other], ignore_index=True))
    assert live[:len(rows)].tolist() == fan_rules.live_arrows(rows)[0].tolist()
