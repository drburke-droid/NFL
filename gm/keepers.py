"""What a roster is worth NEXT season, so a trade can be judged on more than this year's title.

The search so far asks only whether a trade raises P(title) now. In a keeper league that is the
wrong question for half the teams: a side sitting on 1% title odds should be selling this season
for next, and every such trade looks like a loss to a win-now objective. This supplies the other
term.

It does not recompute keeper economics -- scripts/predict_keepers.py already does that, against
the league's real rules, several of which are not guessable:

  * cost = last year's corrected basis + the CURRENT OWNER's inflation bump, and that bump depends
    on where the owner finished (+$5, +$3 if they missed the playoffs, +$0 if they also drafted the
    champion). So a player's keeper cost CHANGES when he is traded, because he inherits the new
    owner's bump. A contender pays more to keep the same player than a cellar team does.
  * acquisitionType decides the basis. TRADE carries the drafted salary to the new owner; ADD --
    any mid-season waiver pickup, including a drafted player who was dropped and re-added --
    resets it to $1. That means a roster cut is not merely a lost player: whoever claims him next
    gets a $1 keeper.
  * at most three keepers per team, so the fourth-best surplus on a roster is worth nothing, in
    exactly the way a fourth startable receiver is worth nothing.

That last rule is why this cannot be a per-player number added up. A roster's keeper value is the
sum of its best three surpluses, so what a player adds depends entirely on what is already there.
"""
import json
import os

MAX_KEEPERS = 3


def load_board(path=None, root=None):
    """The keeper board predict_keepers.py emits, as {team_id: {"bump": int, "players": {...}}}."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.path.join(root, "outputs", "draft_tool", "keepers_2026.js")
    s = open(path).read()
    d = json.loads(s[s.index("{"):s.rindex(";")])
    out = {}
    for t in d["teams"]:
        out[str(t["id"])] = {
            "bump": t["bump2026"],
            "players": {c["name"]: c for c in t["candidates"]},
        }
    return out


def surplus_under(entry, bump):
    """A player's keeper surplus if the owner's inflation bump were `bump`.

    cost = basis (or $1 for a waiver pickup) + the owner's bump, so moving a player to a team with
    a different finish changes what keeping him costs without changing what he is worth.
    """
    base = 1 if entry.get("waiver") else entry.get("basis", 0)
    return entry["value"] - (base + bump)


def team_keeper_value(board, team_id, names, bump=None, max_keepers=MAX_KEEPERS):
    """Best `max_keepers` surpluses on a roster. Players off the board contribute nothing."""
    t = board.get(str(team_id))
    if not t:
        return 0.0
    bump = t["bump"] if bump is None else bump
    vals = []
    for n in names:
        e = t["players"].get(n)
        if e is None:
            continue
        vals.append(surplus_under(e, bump))
    return float(sum(sorted(vals, reverse=True)[:max_keepers]))


def board_lookup(board, name):
    """Find a player's entry on whichever roster currently holds him."""
    for tid, t in board.items():
        if name in t["players"]:
            return tid, t["players"][name]
    return None, None


def keeper_delta(board, rosters_before, rosters_after, teams):
    """{team_id: change in keeper value} for a trade, honouring the new owner's bump.

    rosters_* are {team_id: [player names]}. A player who appears in neither after-roster was cut,
    and his keeper value is simply gone -- for the team that held him.
    """
    out = {}
    for t in teams:
        t = str(t)
        bump = board.get(t, {}).get("bump", 0)
        before = team_keeper_value(board, t, rosters_before.get(t, []))
        # a player arriving from elsewhere brings his basis but pays THIS owner's bump
        vals = []
        for n in rosters_after.get(t, []):
            src, e = board_lookup(board, n)
            if e is None:
                continue
            vals.append(surplus_under(e, bump))
        after = float(sum(sorted(vals, reverse=True)[:MAX_KEEPERS]))
        out[t] = after - before
    return out
