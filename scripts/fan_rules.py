"""What a Fan Picks arrow means, under each rule a code can carry.

The page's code says which rule its presses were made under ("r"), and the recorder and the grader
both turn presses into a number through `adjusted()` here, so the two can never disagree about it.

  pct10  (codes with no "r", through 2026-09-24)  each press = 10% of the baseline; -10 = zero
  u1     (from 2026-09-24)                        each press = a set amount in the stat's own units

u1 exists because a percentage made small stats unreachable: "one more touchdown" on a 0.6-TD
projection was seventeen presses. Its steps are the calls fans actually make -- half a touchdown or
interception, one catch, ten rushing or receiving yards, twenty-five passing yards -- which also
makes most single presses worth about one DraftKings point (a TD half-step is 2-3). docs/fan.html
carries the same table; tests/test_dk_scoring.py fails if the two drift.
"""
STEP_U = {"pass_yds": 25.0, "pass_tds": 0.5, "pass_int": 0.5, "rush_yds": 10.0, "rush_tds": 0.5,
          "rec": 1.0, "rec_yds": 10.0, "rec_tds": 0.5}
RULES = ("pct10", "u1")
LEGACY = "pct10"


def rule_of(value):
    """A code's or a row's rule; anything missing or unknown is the legacy percentage rule."""
    return value if isinstance(value, str) and value in RULES else LEGACY


def adjusted(rule, stat, base, n):
    """The fan's number for one stat: `base` moved by `n` presses under `rule`, never below zero."""
    if rule == "u1":
        return max(0.0, float(base) + STEP_U[stat] * int(n))
    return max(0.0, float(base) * (1 + 0.1 * int(n)))


def min_presses(rule, stat, base):
    """The most ▼ presses that still mean something: -10 under pct10, down to zero under u1."""
    if rule == "u1":
        import math
        return -math.ceil(float(base or 0) / STEP_U[stat] - 1e-9)
    return -10


def live_arrows(rows):
    """Which recorded arrows count, as three boolean masks over `rows`: (live, late, superseded).

    For each game, a fan's live arrows are the ones in the LAST set he locked in before that game
    kicked off. Fans come back during the week: lock in Thursday afternoon, return Saturday to
    change the Sunday players, lock in again. Keeping only the latest set for the whole week (the
    rule until 2026-09-24) threw away every Thursday-night arrow in that case, because the Saturday
    set replaced Thursday's and its own Thursday arrows came after kickoff. Per game, Thursday's
    game keeps Thursday's set and Sunday's games take Saturday's.

    rows needs season, week, fan, submitted_at (ISO) and _kick (the arrow's game's UTC kickoff, NaT
    when unknown -- such an arrow just follows the fan's latest set). A set that leaves a game out
    still counts as the fan's word on it: arrows he removed stay removed.
    late        placed after its game kicked off (a result the fan could already have seen)
    superseded  replaced by a later set locked in before that game
    """
    import pandas as pd
    sub = pd.to_datetime(rows["submitted_at"], utc=True, errors="coerce")
    kick = pd.to_datetime(rows["_kick"], utc=True, errors="coerce") if "_kick" in rows else pd.Series(pd.NaT, index=rows.index)
    late = kick.notna() & sub.notna() & (sub > kick)
    current = pd.Series(False, index=rows.index)
    for _, idx in rows.groupby(["season", "week", "fan"]).groups.items():
        times = sorted(set(sub[idx].dropna()))
        latest_str = rows.loc[idx, "submitted_at"].astype(str).max()
        for i in idx:
            if pd.isna(sub[i]):                       # unparseable stamp: the old string-latest rule
                current[i] = str(rows.at[i, "submitted_at"]) == latest_str
                continue
            before = [t for t in times if pd.isna(kick[i]) or t <= kick[i]]
            current[i] = bool(before) and sub[i] == before[-1]
    live = current & ~late
    return live, late, ~late & ~current
