"""What a player's NAME is worth to the other managers, as opposed to what he will score.

The mutual-gain search (gm.trades.find_trades) keeps a trade only if the simulator says both sides
gain. Managers don't run simulators. Most judge a trade by who they recognise, and a name keeps
its price long after the production behind it has moved. This module prices that perception, so
the search can look for trades that help you on the simulator while looking like a win to the
other side.

  brand     ESPN's 2026 sitewide average auction salary (outputs/draft_tool/espn_salary_2026.js,
            from ESPN Live Draft Trends): the public price tag every ESPN manager saw on the player
            at the draft. $0 for a player ESPN never priced -- no name to trade on.
  fandom    the one valuation bias measured in THIS league (outputs/draft_tool/owner_bias_2026.js,
            mined from 2023-25 winning bids): some owners pay 1.05-1.3x for their favourite NFL
            team's players. Positional overpay and player loyalty were tested and are not real
            here, so they are not modelled.
  premium   brand minus the price the market pays for this projection: a curve fitted across
            every rostered player ESPN priced, $ against points a week over the waiver wire at
            his position (r = 0.78 on 2026-09-26). Positive = the name costs more than the
            production, so sell. Negative = production without the name, so buy.

A perceived gain is a model of acceptance, not a prediction of it. The report shows it beside what
the simulator says the other side actually gets, so you can see how lopsided each proposal is.
"""
import json
import os
import re

from gm.players import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALARY = os.path.join(ROOT, "outputs", "draft_tool", "espn_salary_2026.js")
BIAS = os.path.join(ROOT, "outputs", "draft_tool", "owner_bias_2026.js")
LEAGUE_FILE = os.path.join(ROOT, "outputs", "espn_league.json")
SKILL = ("QB", "RB", "WR", "TE")


def _js_value(path, const):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"const %s = (.*?);\s*$" % re.escape(const), txt, re.S | re.M)
    if not m:
        raise ValueError(f"{const} not found in {path}")
    return json.loads(m.group(1))


def load_salaries(path=SALARY):
    """{normalised name: ESPN average auction $}"""
    return {norm(n): float(v) for n, _pos, v in _js_value(path, "ESPN_SALARY_RAW")}


def load_fandom(bias_path=BIAS, league_path=LEAGUE_FILE):
    """{team_id (str): {NFL team: bid multiplier}} for the owners with a measured team bias."""
    bias = _js_value(bias_path, "OWNER_BIAS")
    league = json.load(open(league_path, encoding="utf-8"))
    out = {}
    for t in league.get("teams", []):
        b = bias.get(t.get("owner") or "")
        if b and b.get("fan"):
            out[str(t["id"])] = {k: float(x) for k, x in b["fan"].items()}
    return out


def brand(v, sal):
    """A player's name value in $: his ESPN price, 0 if ESPN never priced him."""
    return float(sal.get(norm(v["name"]), 0.0))


def perceived(team_id, receive, send, sal, fandom):
    """How much better a trade LOOKS to `team_id`, in brand $: names received minus names sent,
    each scaled by that owner's fandom for the player's NFL team."""
    fan = fandom.get(str(team_id), {})
    val = lambda v: brand(v, sal) * fan.get(v.get("nfl_team") or "", 1.0)
    return sum(val(v) for v in receive) - sum(val(v) for v in send)


def ppw(v):
    return v["mean"] * v["p_play"]


def market_curve(est, sal, waiver):
    """$ the public market pays per point a week over the waiver wire: (intercept, slope), fitted
    on rostered skill players ESPN priced who project above waiver."""
    import numpy as np
    xs, ys = [], []
    for v in est.values():
        if v["pos"] in SKILL and norm(v["name"]) in sal:
            over = ppw(v) - waiver.get(v["pos"], 0.0)
            if over > 0:
                xs.append(over); ys.append(sal[norm(v["name"])])
    if len(xs) < 10:
        raise ValueError("too few priced players to fit the market curve")
    slope, icpt = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    return float(icpt), float(slope), r, len(xs)


def premium(v, sal, waiver, curve):
    """brand $ minus what the market pays for this projection (the curve, floored at $1)."""
    icpt, slope = curve[0], curve[1]
    fair = max(1.0, icpt + slope * max(ppw(v) - waiver.get(v["pos"], 0.0), 0.0))
    return brand(v, sal) - fair, fair
