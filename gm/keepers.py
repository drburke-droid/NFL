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
import re

import numpy as np

MAX_KEEPERS = 3
NEXT_BUMP = 5          # the owner's inflation bump next year is not known until the standings settle
YOUNG = 24             # age at which the year-over-year drift flips sign (below: rises; above: fades)
# Year-over-year change in a player's league-scored ppg, players with 6+ games both seasons, 2012-25
# nflverse: (mean drift, sd) by position and whether he was 24 or younger. Young players drift up
# by half a point and everyone else down by the same, and the spread is four points at RB/WR,
# three at TE, five at QB. That spread is what gives a cheap young player option value: the dollar
# curve is convex, so the expected dollars of an uncertain player exceed the dollars of his
# expected level.
YOY = {("QB", True): (0.69, 5.27), ("QB", False): (-0.46, 5.15),
       ("RB", True): (0.30, 4.45), ("RB", False): (-0.87, 4.01),
       ("WR", True): (0.58, 3.58), ("WR", False): (-0.88, 3.61),
       ("TE", True): (0.46, 3.00), ("TE", False): (-0.45, 2.81)}


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


# ----------------------------------------------------------------- next season, from this one
def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def draft_tool_players(root=None):
    """The draft tool's player file (docs/data.js): preseason projection and age by name."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = open(os.path.join(root, "docs", "data.js"), encoding="utf-8").read()
    P = json.loads(s.split("const PLAYERS = ")[1].rsplit(";", 1)[0])
    return {(_norm(p["name"]), p["position"]): p for p in P}


def _isotonic(x, y):
    """Pool-adjacent-violators: the closest non-decreasing fit of y on sorted x."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks = [[ys[i], 1] for i in range(len(ys))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m, n = blocks[i], blocks[i + 1]
            merged = [(m[0] * m[1] + n[0] * n[1]) / (m[1] + n[1]), m[1] + n[1]]
            blocks[i:i + 2] = [merged]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.concatenate([[b[0]] * b[1] for b in blocks])
    return xs, fitted


def dollar_curve(board=None, players=None):
    """{pos: (ppg grid, dollars)}: what this league pays at auction for a level of weekly scoring.

    Fitted from the keeper board's calibrated values against the draft tool's projected ppg for
    the same players, made monotone. The shape is the point: RB 8 ppg is $1, 11 is $6, 14 is $29.
    """
    board = board or load_board()
    players = players or draft_tool_players()
    pairs = {}
    for t in board.values():
        for name, e in t["players"].items():
            p = players.get((_norm(name), e["pos"]))
            if p and p.get("proj_ppg") is not None:
                pairs.setdefault(e["pos"], []).append((float(p["proj_ppg"]), float(e["value"])))
    return {pos: _isotonic(*zip(*v)) for pos, v in pairs.items() if len(v) >= 8}


def dollars_at(curve, pos, ppg):
    if pos not in curve:
        return 1.0
    xs, ys = curve[pos]
    return float(np.interp(ppg, xs, ys, left=1.0, right=ys[-1]))


def expected_dollars(curve, pos, ppg, age=None):
    """Expected auction value next season for a player at `ppg` now: the dollar curve averaged over
    the measured year-over-year distribution of his level. Convexity is what makes a $1 dart worth
    more than the dollars at his expected level."""
    drift, sd = YOY.get((pos, bool(age is not None and age <= YOUNG)), (0.0, 4.0))
    grid = ppg + drift + sd * np.linspace(-2.5, 2.5, 41)
    w = np.exp(-0.5 * np.linspace(-2.5, 2.5, 41) ** 2)
    return float(np.sum(w * np.array([dollars_at(curve, pos, g) for g in grid])) / w.sum())


def expected_bump(cfg, p_playoffs):
    """The owner's inflation bump next year, in expectation over whether he makes the playoffs.

    The league pays +$5 on a keeper's price for a playoff team and +$3 for one that misses (+$0
    for the non-playoff owner who drafted the champion, which is not predictable and is ignored).
    A team at 35% to make it expects about $3.70, and that is what a $1 stash costs him to keep.
    """
    inf = cfg.raw.get("keepers", {}).get("inflation") or {}
    base = float(inf.get("base", NEXT_BUMP))
    missed = base + float(next((c["delta"] for c in inf.get("conditions", [])
                                if c.get("name") == "missed_playoffs"), -2))
    return base * p_playoffs + missed * (1.0 - p_playoffs)


def next_season_board(cfg, est, curve=None, players=None, bump=NEXT_BUMP):
    """A keeper board for NEXT season built from the current rosters and the fitted player means.

    keepers_2026.js was the board for keeping INTO this season -- costs from the 2025 draft, values
    from the 2026 preseason -- and that decision is made. What a trade changes is next year's:
      cost   = this year's basis (the auction price; $1 for any waiver pickup, per league rule)
               + the owner's bump (assumed NEXT_BUMP until the standings settle)
      value  = expected auction dollars at the player's healthy level (the fitted rest-of-season
               mean, which does not know he is hurt) after a measured year of drift and spread
    Same shape as load_board() so the trade search reads it unchanged. Injured stashes are the
    case this exists for: their weekly value is near zero, their keeper value is not.
    `bump` is a number for every team or {team_id: bump} (see expected_bump).
    """
    curve = curve or dollar_curve()
    players = players or draft_tool_players()
    out = {}
    for t in cfg.teams:
        tid = str(t["team_id"])
        b = float(bump.get(tid, NEXT_BUMP)) if isinstance(bump, dict) else float(bump)
        out[tid] = {"bump": b, "players": {}}
        for (tt, pid), v in est.items():
            if tt != tid or v["pos"] not in ("QB", "RB", "WR", "TE"):
                continue
            waiver = str(v.get("acquired") or "").upper() == "ADD"
            basis = 1.0 if waiver else float(v.get("keeper_price") or 0.0)
            p = players.get((_norm(v["name"]), v["pos"]))
            age = (p.get("age") + 1) if p and p.get("age") else None
            healthy = float(v.get("raw_mean", v["mean"]))
            value = expected_dollars(curve, v["pos"], healthy, age)
            cost = basis + b
            out[tid]["players"][v["name"]] = {
                "name": v["name"], "pos": v["pos"], "value": round(value, 1), "basis": basis,
                "waiver": waiver, "cost": cost, "surplus": round(value - cost, 1),
                "healthy_ppg": round(healthy, 1), "age": age, "injury": v.get("injury"),
                "times_kept": v.get("times_kept", 0)}
    return out
