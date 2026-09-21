"""Search the league for trades worth proposing, rather than grading one somebody already typed in.

A calculator that evaluates a trade you thought of is the small half of the problem. The useful
half is that, knowing every roster and where every team sits on the win curve, the trades that
should happen are computable -- and most of them are ones neither manager would think to ask for,
because they involve a player who is surplus to one roster and a starter on another.

Two stages, because a full simulation per candidate is far too slow. Roughly 2,800 one-for-one
swaps exist between one roster and the other eleven, and each exact evaluation runs two seasons.
So stage one ranks every swap by the change in each side's expected starting lineup, which is a
sort over sixteen numbers and costs nothing, and stage two runs the real simulation on the
survivors. The cheap stage is only used to ORDER candidates; nothing is reported from it.

The acceptance filter is what makes the output a list of proposals rather than a wishlist: a trade
is only kept if the simulation says BOTH teams' title odds rise. That is possible far more often
than a value chart suggests, because a roster is worth the best nine it can field -- a third
quarterback in a one-quarterback league is worth nothing to his owner and a starter to someone
else, and the trade creates value rather than moving it.

The honest limit: this models whether a manager SHOULD accept, not whether they will. It also
assumes both sides value this season only, which is wrong in a keeper league -- a rebuilding team
should trade a win-now piece for a keeper-eligible young one even when this year's title odds
fall. Keeper value is in the config and not yet in this search.
"""
import numpy as np

from . import simulate as sim
from .simulate import roster_capacity


def expected_lineup(cfg, means_by_pos):
    """Expected points from the best startable lineup. The cheap stage-one score."""
    flex = cfg.raw["roster"].get("flex_eligibility", {})
    total, used = 0.0, {}
    for slot, count in cfg.starters:
        if slot in flex:
            continue
        vals = sorted(means_by_pos.get(slot, []), reverse=True)
        total += sum(vals[:count])
        used[slot] = min(count, len(vals))
    for slot, count in cfg.starters:
        if slot not in flex:
            continue
        pool = []
        for pos in flex[slot]:
            pool += sorted(means_by_pos.get(pos, []), reverse=True)[used.get(pos, 0):]
        total += sum(sorted(pool, reverse=True)[:count])
    return total


def _roster_means(est, team_id, out_ids=(), in_players=(), capacity=None):
    """Expected weekly contributions by position, after a swap and any forced cut.

    Rosters are full, so a team receiving more than it sends must drop somebody, and stage one has
    to see that cost or it will rank uneven trades as free depth.
    """
    drop = set(out_ids)
    kept = [v for (t, pid), v in est.items() if t == team_id and pid not in drop]
    kept = kept + list(in_players)
    if capacity is not None and len(kept) > capacity:
        kept.sort(key=lambda v: v["mean"] * v["p_play"])
        kept = kept[len(kept) - capacity:]
    out = {}
    for v in kept:
        out.setdefault(v["pos"], []).append(v["mean"] * v["p_play"])
    return out


def _packages(players, max_size, pool=None):
    """Every package of up to max_size players a roster might send.

    Drawn from the `pool` best players by expected contribution rather than the whole roster. The
    bottom of a full roster is filler that neither side can start, so including it multiplies the
    search without adding a trade anyone would make. The pool is NOT restricted to low-value
    players: consolidation means giving up real starters, so the search has to be able to offer
    them.
    """
    ids = sorted(players, key=lambda i: -players[i]["mean"] * players[i]["p_play"])
    if pool:
        ids = ids[:pool]
    out = [(i,) for i in ids]
    n = len(ids)
    if max_size >= 2:
        out += [(ids[a], ids[b]) for a in range(n) for b in range(a + 1, n)]
    if max_size >= 3:
        out += [(ids[a], ids[b], ids[c]) for a in range(n) for b in range(a + 1, n)
                for c in range(b + 1, n)]
    return out


def find_trades(cfg, est, my_team, shortlist=25, top_n=8, n_sims=8000, seed=0, played_weeks=(),
                max_package=3, max_combined=4, pool=12, max_per_team=3):
    """Trades that raise BOTH teams' title odds, best first.

    Packages of up to max_package a side. Two-for-one matters more than it sounds: a one-for-one
    mostly moves value rather than creating it -- of 2,816 such swaps available to this roster,
    738 helped one side and 564 the other but exactly ONE helped both. Consolidation is where the
    mutual gain lives, because a roster is worth the best nine it can field, so a team with four
    startable receivers and a hole at back genuinely gains by sending two for one.

    shortlist : how many stage-one candidates to evaluate exactly
    top_n     : how many surviving proposals to return
    """
    est = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    mine = {p: v for (t, p), v in est.items() if t == my_team}
    cap = roster_capacity(cfg)
    base_me = expected_lineup(cfg, _roster_means(est, my_team, capacity=cap))
    my_pkgs = _packages(mine, max_package, pool)

    cands = []
    for other in [t["team_id"] for t in cfg.teams if t["team_id"] != my_team]:
        theirs = {p: v for (t, p), v in est.items() if t == other}
        base_them = expected_lineup(cfg, _roster_means(est, other, capacity=cap))
        their_pkgs = _packages(theirs, max_package, pool)
        for gp in my_pkgs:
            give = [mine[i] for i in gp]
            for rp in their_pkgs:
                if len(gp) + len(rp) > max_combined:
                    continue                      # the far tail of shapes costs more than it finds
                get = [theirs[i] for i in rp]
                d_me = expected_lineup(cfg, _roster_means(est, my_team, gp, get, cap)) - base_me
                if d_me <= 0:
                    continue
                d_them = expected_lineup(cfg, _roster_means(est, other, rp, give, cap)) - base_them
                if d_them > 0:
                    cands.append((d_me + d_them, d_me, d_them, other, gp, rp, give, get))
    cands.sort(reverse=True, key=lambda c: c[0])

    out = []
    for _, d_me, d_them, other, gp, rp, give, get in cands[:shortlist]:
        moves = [(i, my_team, other) for i in gp] + [(i, other, my_team) for i in rp]
        r = sim.evaluate_trade(cfg, est, moves, n_sims=n_sims, seed=seed, played_weeks=played_weeks)
        me, them = r["delta"][my_team], r["delta"][other]
        if me["p_title"]["delta"] > 0 and them["p_title"]["delta"] > 0:
            out.append({
                "give": [v["name"] for v in give], "get": [v["name"] for v in get],
                "with_team": other,
                "my_p_title": me["p_title"]["delta"], "their_p_title": them["p_title"]["delta"],
                "my_p_playoffs": me["p_playoffs"]["delta"],
                "my_points": me["exp_points_for"]["delta"],
                "their_points": them["exp_points_for"]["delta"],
                "stage1_me": d_me, "stage1_them": d_them})
    out.sort(reverse=True, key=lambda d: d["my_p_title"])
    # One counterparty usually dominates -- here every surviving trade is with the team carrying
    # three quarterbacks. That is a real finding, but a list of eight variations on one deal is
    # less useful than a few genuine alternatives, so cap how many any single partner may fill.
    seen, kept = {}, []
    for d in out:
        n = seen.get(d["with_team"], 0)
        if n < max_per_team:
            seen[d["with_team"]] = n + 1
            kept.append(d)
    return {"proposals": kept[:top_n], "considered": len(cands),
            "simulated": min(shortlist, len(cands)), "partners": len(seen)}
