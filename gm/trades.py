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

from . import keepers as kp
from . import simulate as sim
from .simulate import roster_capacity

# A $200 budget buys a starting lineup worth about 124 points a week in this league, so this is
# what a dollar of keeper surplus is worth in next season's weekly points.
DOLLARS_TO_POINTS = 124.0 / 200.0


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


def _title_slope(cfg, est, team_id, n_sims=4000, seed=0, played_weeks=None, bump=1.0):
    """Title probability gained per extra point a week, for THIS team where it currently sits.

    The win curve is the whole reason a trade is not worth the same to both sides, so converting
    next season's dollars into title odds with one league-wide rate would throw away the thing the
    simulator exists to compute. Measured by nudging the roster and re-running.
    """
    need, _ = sim.weeks_needed(cfg, played_weeks)
    lifted = {k: ({**v, "mean": v["mean"] + bump / max(cfg.starting_size, 1)} if k[0] == team_id else v)
              for k, v in est.items()}
    rng = lambda: __import__("numpy").random.default_rng(seed)
    a = sim.run(cfg, sim.player_team_scores(cfg, est, n_sims, need, rng=rng(), common_seed=seed),
                played_weeks)["teams"][team_id]["p_title"]
    b = sim.run(cfg, sim.player_team_scores(cfg, lifted, n_sims, need, rng=rng(), common_seed=seed),
                played_weeks)["teams"][team_id]["p_title"]
    return max(b - a, 0.0) / bump


def _keeper_gain(board, team_id, roster, out_ids, in_players, cap, est):
    """Change in a roster's best-three keeper surplus, cheaply, for stage-one ranking."""
    if board is None:
        return 0.0
    before = [v["name"] for v in roster.values()]
    kept = [v for pid, v in roster.items() if pid not in set(out_ids)] + list(in_players)
    if len(kept) > cap:
        kept = sorted(kept, key=lambda v: v["mean"] * v["p_play"])[len(kept) - cap:]
    return (kp.team_keeper_value(board, team_id, [v["name"] for v in kept])
            - kp.team_keeper_value(board, team_id, before))


def find_trades(cfg, est, my_team, shortlist=25, top_n=8, n_sims=8000, seed=0, played_weeks=None,
                max_package=3, max_combined=4, pool=12, max_per_team=3,
                board=None, keeper_discount=0.0):
    """Trades that raise BOTH teams' title odds, best first.

    Packages of up to max_package a side. Two-for-one matters more than it sounds: a one-for-one
    mostly moves value rather than creating it -- of 2,816 such swaps available to this roster,
    738 helped one side and 564 the other but exactly ONE helped both. Consolidation is where the
    mutual gain lives, because a roster is worth the best nine it can field, so a team with four
    startable receivers and a hole at back genuinely gains by sending two for one.

    shortlist     : how many stage-one candidates to evaluate exactly
    top_n         : how many surviving proposals to return
    board           : the keeper board from gm.keepers.load_board()
    keeper_discount : how much next season counts against this one. 0 is pure win-now and the
                      default; 1 treats a dollar of keeper surplus as worth its full value in
                      next season's lineup. Keeper deltas are reported either way -- the knob
                      only decides whether they enter the ranking.

    With a board, the acceptance test loosens in the way a keeper league actually works: a trade
    also survives if a side's title odds fall but its keeper value rises by enough. That is what
    lets a seller sell, and a win-now-only objective can never find it.
    """
    est = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    stage1_keepers = DOLLARS_TO_POINTS * keeper_discount if board is not None else 0.0
    # Dollars have to become title probability at THIS team's position on the win curve, not the
    # league's average slope. A point a week is worth about two percentage points to a team on the
    # bubble and almost nothing to one at 1% or 90%, so a single exchange rate would misprice
    # exactly the teams that most need to sell. The slope is measured per team by nudging its
    # roster a point a week and re-simulating -- two extra runs, once, not per candidate.
    slope = {}
    if keeper_discount and board is not None:
        for t in [my_team] + [x["team_id"] for x in cfg.teams if x["team_id"] != my_team]:
            slope[t] = _title_slope(cfg, est, t, n_sims=n_sims, seed=seed,
                                    played_weeks=played_weeks)
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
                d_them = expected_lineup(cfg, _roster_means(est, other, rp, give, cap)) - base_them
                k_me = k_them = 0.0
                if stage1_keepers:
                    k_me = _keeper_gain(board, my_team, mine, gp, get, cap, est)
                    k_them = _keeper_gain(board, other, theirs, rp, give, cap, est)
                # Stage one must score on the same objective stage two will, or a seller's trade
                # -- worse this year, better next -- is discarded before it is ever simulated.
                s_me = d_me + stage1_keepers * k_me
                s_them = d_them + stage1_keepers * k_them
                if s_me > 0 and s_them > 0:
                    cands.append((s_me + s_them, d_me, d_them, other, gp, rp, give, get))
    cands.sort(reverse=True, key=lambda c: c[0])

    out = []
    for _, d_me, d_them, other, gp, rp, give, get in cands[:shortlist]:
        moves = [(i, my_team, other) for i in gp] + [(i, other, my_team) for i in rp]
        r = sim.evaluate_trade(cfg, est, moves, n_sims=n_sims, seed=seed, played_weeks=played_weeks)
        me, them = r["delta"][my_team], r["delta"][other]
        k_me = k_them = 0.0
        if board is not None:
            after = sim.apply_trade(est, moves, cap)
            rb = {t: [v["name"] for (x, _), v in est.items() if x == t] for t in (my_team, other)}
            ra = {t: [v["name"] for (x, _), v in after.items() if x == t] for t in (my_team, other)}
            kd = kp.keeper_delta(board, rb, ra, [my_team, other])
            k_me, k_them = kd.get(str(my_team), 0.0), kd.get(str(other), 0.0)
        # each side's dollars converted at its OWN slope
        w_me = DOLLARS_TO_POINTS * keeper_discount * slope.get(my_team, 0.0)
        w_them = DOLLARS_TO_POINTS * keeper_discount * slope.get(other, 0.0)
        score_me = me["p_title"]["delta"] + w_me * k_me
        score_them = them["p_title"]["delta"] + w_them * k_them
        if score_me > 0 and score_them > 0:
            out.append({
                "keeper_me": k_me, "keeper_them": k_them,
                "score_me": score_me, "score_them": score_them,
                "give": [v["name"] for v in give], "get": [v["name"] for v in get],
                "with_team": other,
                "my_p_title": me["p_title"]["delta"], "their_p_title": them["p_title"]["delta"],
                "my_p_playoffs": me["p_playoffs"]["delta"],
                "my_points": me["exp_points_for"]["delta"],
                "their_points": them["exp_points_for"]["delta"],
                "stage1_me": d_me, "stage1_them": d_them})
    out.sort(reverse=True, key=lambda d: d["score_me"])
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
