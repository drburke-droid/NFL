"""DK spells some players differently from the FFA file; reconcile must bridge that without
inventing matches. The live failure was Kenny/Kenneth Gainwell: DK had priced him, the exact-name
join dropped the line, and `no_line` then reported him as a player the books would not price.
"""
import importlib.util
import os
import re
import sys

import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("name_match", os.path.join(REPO, "scripts", "name_match.py"))
nm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nm)


def norm(s):                                  # same normalization sabersim_weekly applies
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


@pytest.mark.parametrize("a,b", [
    ("josh", "joshua"), ("kenny", "kenneth"), ("cam", "cameron"),
    ("dan", "daniel"), ("ben", "benjamin"), ("josh", "josh"),
])
def test_same_person_accepts_one_name_written_two_ways(a, b):
    assert nm.same_person(a, b) and nm.same_person(b, a)


@pytest.mark.parametrize("a,b", [
    ("cameron", "keleki"), ("dohnte", "jakobi"), ("keshawn", "caleb"),
    ("mike", "michael"),                      # a known miss: strict beats guessing
    ("kyren", "kyle"), ("javonte", "jameson"),
])
def test_same_person_rejects_different_people(a, b):
    assert not nm.same_person(a, b) and not nm.same_person(b, a)


def test_reconcile_bridges_a_nickname():
    alias, amb = nm.reconcile([("g1", "kenny gainwell")], {"g1": {"TB", "CLE"}},
                              [("kenneth gainwell", "TB")])
    assert alias == {"kenny gainwell": "kenneth gainwell"} and amb == []


def test_reconcile_refuses_a_different_player_with_the_same_surname():
    alias, amb = nm.reconcile([("g1", "cameron latu")], {"g1": {"BUF", "DET"}},
                              [("keleki latu", "BUF")])
    assert alias == {} and amb == []


def test_reconcile_stays_inside_the_game():
    """Same surname, same first name, but the slate player is in a different game."""
    alias, _ = nm.reconcile([("g1", "kenny gainwell")], {"g1": {"TB", "CLE"}},
                            [("kenneth gainwell", "PHI")])
    assert alias == {}


def test_reconcile_reports_ambiguity_instead_of_picking():
    """Two different slate players both plausibly "cam smith" — refuse rather than guess."""
    alias, amb = nm.reconcile([("g1", "cam smith")], {"g1": {"TB", "CLE"}},
                              [("cameron smith", "TB"), ("camden smith", "CLE")])
    assert alias == {} and amb == ["cam smith"]


def test_identical_slate_names_are_not_ambiguous():
    """The same normalized name on both teams collapses to one candidate: the downstream join is
    on that name anyway, so aliasing to it changes nothing."""
    alias, amb = nm.reconcile([("g1", "josh allen")], {"g1": {"BUF", "JAX"}},
                              [("joshua allen", "BUF"), ("joshua allen", "JAX")])
    assert alias == {"josh allen": "joshua allen"} and amb == []


def test_reconcile_leaves_exact_matches_alone():
    alias, _ = nm.reconcile([("g1", "josh palmer")], {"g1": {"BUF", "DET"}},
                            [("josh palmer", "BUF")])
    assert alias == {}


def test_against_the_committed_week_2_feeds():
    """Real DK names vs the real FFA file, with every team treated as one game — the most
    adversarial framing, since it removes the same-game guard and leans entirely on the
    first-name test and the uniqueness check."""
    ffa = pd.read_csv(os.path.join(REPO, "data", "ffanalytics", "FFAn_weekly", "raw_stats_2026_wk2.csv"),
                      na_values=["NA"])
    ffa = ffa[ffa.position.isin(["QB", "RB", "WR", "TE"])].dropna(subset=["player"])
    players = [(norm(p), "ALL") for p in ffa.player]

    dk = pd.read_csv(os.path.join(REPO, "data", "props_frames", "snapshots", "props_2026.csv"))
    dk = dk[dk.week == 2].dropna(subset=["player"])
    names = sorted({norm(p) for p in dk.player})
    alias, amb = nm.reconcile([("g", n) for n in names], {"g": {"ALL"}}, players)

    assert alias.get("kenny gainwell") == "kenneth gainwell"
    assert alias.get("joshua palmer") == "josh palmer"
    for bad in ("cameron latu", "dohnte meyers", "keshawn williams"):
        assert bad not in alias, f"{bad} should not have been matched"
    for dk_name, ffa_name in alias.items():        # an alias never crosses surnames
        assert dk_name.split()[-1] == ffa_name.split()[-1]
