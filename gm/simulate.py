"""Rest-of-season Monte Carlo: play the remaining weeks many times and count the titles.

This is the number every other question reduces to. "Is this trade fair" is unanswerable; "does
this raise my chance of winning the league" is not, and answering it needs the standings, the
schedule and the bracket rather than a value chart. Simulating is what makes all three enter at
once: a team's title odds depend on where it sits now, who it still has to play, and how the
bracket resolves.

What that buys you over a value chart is the WIN CURVE. A marginal point is worth a great deal to
a team on the playoff bubble and nearly nothing to one at 1-9 or 9-1, so the same trade is
genuinely good for one side and bad for the other -- a fact about the standings, not a matter of
taste. Run the season twice, once per roster, and the change in P(title) is the verdict.

The engine takes weekly team scores as an ARRAY rather than computing them. How a team's week is
generated -- a player-level projection model, a shrunk points-for estimate, a stub chosen to make
a test's answer obvious -- is a different question from what those scores imply for a season.
Keeping them apart is what lets the bracket logic be tested exactly, and lets the projection model
be replaced without touching any of this.

Byes fall out of the bracket rather than being special-cased: a 6-team playoff fills an 8-slot
bracket, seeds 7 and 8 do not exist, and whoever is drawn against an empty chair advances. With
reseeding off, as in Kuhn and Friends, the bracket is fixed after the first round, so the 4/5
winner meets the 1 seed even when the 3/6 winner finished below them.
"""
import zlib

import numpy as np

# Measured on Kuhn and Friends, 490 regular-season team-weeks across 2023-2025.
# A team's week varies by 20.5 points around its own season average; the spread of true team
# strength, once sampling noise is removed from the spread of season averages, is only 8.2. That
# ratio is why a team's record early on says so little: week-to-week noise is 2.5x the real
# difference between teams.
WEEKLY_SD = 20.5
# How many games of league-average pull to apply to an observed average. Two estimates agree the
# answer is not small: variance components (within^2 / between^2) gives 6.25, and regressing
# rest-of-season scoring on to-date scoring across weeks 2-11 gives a median of 7.7. Both say a
# team's scoring average is still mostly noise at the point most trades get discussed.
PRIOR_GAMES = 7.0


def bracket_order(size):
    """Standard single-elimination seeding order: [1, 2] -> [1, 4, 2, 3] -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1, 2]
    while len(order) < size:
        n = len(order) * 2
        nxt = []
        for s in order:
            nxt += [s, n + 1 - s]
        order = nxt
    return order


def completed_weeks(cfg):
    """Regular weeks already baked into the config's records and points-for.

    A config carrying live standings has those games counted twice unless the simulator is told
    to skip them -- it would replay week 1 and add the result on top of a points_for that already
    includes it. Games played is the honest source: every team having played one game means the
    first week is done.
    """
    played = {t["record"]["w"] + t["record"]["l"] + t["record"]["t"] for t in cfg.teams}
    n = max(played) if played else 0
    return cfg.regular_weeks[:n]


def weeks_needed(cfg, played_weeks=None):
    """How many score columns run() expects: remaining regular weeks, then one per playoff round.

    played_weeks defaults to whatever the standings already contain, so the common call cannot
    silently double-count. Pass an explicit list to override.
    """
    done = set(completed_weeks(cfg) if played_weeks is None else played_weeks)
    future = [w for w in cfg.regular_weeks if w not in done]
    return len(future) + len(cfg.playoff_weeks), future


def week_columns(cfg, played_weeks=None):
    """The league week each score column stands for: remaining regular weeks, then the playoffs.

    This is what lets a bye land in the right column. A column is not "some future week"; it is
    week 11, and every Packer on the roster sits it out.
    """
    _, future = weeks_needed(cfg, played_weeks)
    return list(future) + list(cfg.playoff_weeks)


def run(cfg, weekly_scores, played_weeks=None):
    """Simulate the rest of the season.

    cfg           : a LeagueConfig; its teams carry the record and points-for so far
    weekly_scores : (n_sims, n_teams, n_cols) in cfg team order, where n_cols is
                    weeks_needed(cfg, played_weeks)[0] -- remaining regular weeks then playoffs
    played_weeks  : regular-season weeks already reflected in the config's records

    Returns per-team P(playoffs) and P(title), plus the raw seed and champion draws.
    """
    order = [t["team_id"] for t in cfg.teams]
    idx = {t: i for i, t in enumerate(order)}
    n_sims, n_teams, n_cols = weekly_scores.shape
    if n_teams != len(order):
        raise ValueError(f"weekly_scores has {n_teams} teams, config has {len(order)}")
    need, future = weeks_needed(cfg, played_weeks)
    if n_cols != need:
        raise ValueError(f"weekly_scores has {n_cols} week columns, need {need} "
                         f"({len(future)} regular + {len(cfg.playoff_weeks)} playoff)")

    wins = np.zeros((n_sims, n_teams)); ties = np.zeros((n_sims, n_teams))
    pf = np.zeros((n_sims, n_teams))
    for t in cfg.teams:                                  # the season so far, from the config
        i = idx[t["team_id"]]
        wins[:, i] += t["record"]["w"]; ties[:, i] += t["record"]["t"]
        pf[:, i] += t["points_for"]

    by_week = {}
    for m in cfg.raw["matchups"]:
        by_week.setdefault(m["week"], []).append((idx[m["home"]], idx[m["away"]]))

    for k, wk in enumerate(future):
        s = weekly_scores[:, :, k]
        pf += s
        for h, a in by_week.get(wk, []):
            hw, aw = s[:, h] > s[:, a], s[:, a] > s[:, h]
            wins[:, h] += hw; wins[:, a] += aw
            tie = ~hw & ~aw
            ties[:, h] += tie; ties[:, a] += tie

    # seeding: record first, points-for as the tiebreak. np.lexsort takes the LAST key as primary.
    # axis=1 sorts teams WITHIN each sim; axis=0 would sort sims against each other.
    seeds = np.lexsort((-pf, -(wins + 0.5 * ties)), axis=1)       # (n_sims, n_teams), best first

    n_po = int(cfg.raw["schedule"]["playoff_teams"])
    made = np.zeros((n_sims, n_teams), dtype=bool)
    np.put_along_axis(made, seeds[:, :n_po], True, axis=1)

    size = 1 << (n_po - 1).bit_length()
    alive = np.full((n_sims, size), -1, dtype=int)
    for pos, s in enumerate(bracket_order(size)):
        if s <= n_po:
            alive[:, pos] = seeds[:, s - 1]              # -1 stays put: an empty chair

    for rnd in range(len(cfg.playoff_weeks)):
        s = weekly_scores[:, :, len(future) + rnd]
        nxt = np.full((n_sims, alive.shape[1] // 2), -1, dtype=int)
        for j in range(nxt.shape[1]):
            A, B = alive[:, 2 * j], alive[:, 2 * j + 1]
            a_pts = np.where(A >= 0, np.take_along_axis(s, np.clip(A, 0, None)[:, None], 1)[:, 0], -np.inf)
            b_pts = np.where(B >= 0, np.take_along_axis(s, np.clip(B, 0, None)[:, None], 1)[:, 0], -np.inf)
            nxt[:, j] = np.where(a_pts >= b_pts, A, B)
        alive = nxt
    champion = alive[:, 0]

    seat = np.argsort(seeds, axis=1)                     # team -> its seed-1
    teams_out = {}
    for t, i in idx.items():
        teams_out[t] = {
            "p_playoffs": float(made[:, i].mean()),
            "p_title": float((champion == i).mean()),
            "exp_wins": float(wins[:, i].mean()),
            "exp_points_for": float(pf[:, i].mean()),
            "exp_seed": float((seat[:, i] + 1).mean()),
        }
    return {"teams": teams_out, "n_sims": n_sims, "seeds": seeds, "champion": champion,
            "made": made, "weeks_simulated": future, "playoff_weeks": list(cfg.playoff_weeks)}


def shrunk_team_scores(cfg, n_sims, n_cols, sd=WEEKLY_SD, prior_games=PRIOR_GAMES, rng=None):
    """A team-level score generator: each team's mean shrunk toward the league mean.

    Both constants are measured from this league's own 490 regular-season team-weeks (2023-2025),
    not chosen. It is still a team-level model and knows nothing about who is on a roster, so a
    player-level generator should replace it -- but it is no longer guessing at its own inputs.
    """
    rng = rng or np.random.default_rng(0)
    played = np.array([max(1, t["record"]["w"] + t["record"]["l"] + t["record"]["t"])
                       for t in cfg.teams], float)
    obs = np.array([t["points_for"] for t in cfg.teams], float) / played
    mu = (obs * played + obs.mean() * prior_games) / (played + prior_games)
    return rng.normal(mu[None, :, None], sd, size=(n_sims, len(mu), n_cols))


SCORE_FLOOR = -3.0   # a real fantasy week can go slightly negative, but not far
# How wrong a team's estimated mean might be, in points per week. Two independent views of these
# same rosters -- estimates built from 2026 form, and ESPN's 2025 season totals -- disagree with
# sd 7.2 per team, implying about 5.1 of error each if they err independently. That is comparable
# to the 8.7 spread between the teams themselves, so treating an estimate as exact makes the
# leader look far more certain than the evidence supports. Drawn ONCE PER SIMULATED SEASON, not
# per week: within a season a team has a fixed true strength that we happen not to know.
# It is a lower bound -- two views sharing a bias would agree while both being wrong.
MEAN_SE = 5.1


def _player_rng(common_seed, player_id):
    """A stream that belongs to the PLAYER, not to the roster he happens to be on.

    This is what makes a trade measurable. P(title) for a mid-table team is a few percent, and
    with 20,000 seasons the Monte Carlo error on it is larger than most trades are worth, so two
    independent runs would mostly measure their own noise. Seeding each player's weeks from his
    own id means every player who did not move draws exactly the same season in both worlds, and
    the difference that survives is the trade.
    """
    return np.random.default_rng([common_seed, zlib.crc32(str(player_id).encode())])


def player_team_scores(cfg, est, n_sims, n_cols, rng=None, floor=SCORE_FLOOR,
                       mean_se=MEAN_SE, common_seed=None, weeks=None):
    """Weekly team scores built from a roster, so a trade can be run through the season.

    Each player either plays (Bernoulli on p_play) and draws from his own distribution, or scores
    nothing. The lineup is then set optimally, which matters more than it sounds: a roster is worth
    the best nine it can field in a given week, not the sum of its parts, so depth at a position is
    worth much less than the same points spread across positions. That is why a trade of two good
    backs for one great one can lose value even when the totals match.

    Two things a roster meets in a real season are in here too. With `weeks` (the league week of
    each column, from week_columns), a player sits out his NFL bye, so depth is worth exactly what
    it covers: a roster with five backs off in week 11 feels it, one with a spare receiver does
    not. And every position has a virtual waiver player (the "FA" entries the estimator adds):
    the best free agent fills any slot the roster cannot, so the only tight end's bye costs the
    gap to a streamer, not the whole slot -- and a bench player below waiver level is worth
    nothing, because the manager would pick up the free agent instead.

    The optimal lineup is a sort, not a search. Fill each dedicated slot with the best at that
    position; the flex then takes the best player left over, which can only be the next one down at
    some eligible position. With every slot contributing to one sum, greedy is exact.
    """
    rng = rng or np.random.default_rng(0)
    order = [t["team_id"] for t in cfg.teams]
    # one draw per team per simulated season: how wrong we are about this roster, held fixed
    # across the weeks of that season
    offset = rng.normal(0.0, mean_se, size=(n_sims, len(order))) if mean_se else None
    flex_elig = cfg.raw["roster"].get("flex_eligibility", {})
    dedicated = [(s, c) for s, c in cfg.starters if s not in flex_elig]
    flexes = [(s, c) for s, c in cfg.starters if s in flex_elig]
    out = np.zeros((n_sims, len(order), n_cols))

    waiver = {pos: {**v, "key": f"FA:{pos}", "bye": None} for (t, pos), v in est.items() if t == "FA"}
    for ti, tid in enumerate(order):
        roster = [{**v, "key": v.get("key", pid)} for (t, pid), v in est.items()
                  if t == tid]
        by_pos = {}
        for v in roster:
            by_pos.setdefault(v["pos"], []).append(v)
        for pos, v in waiver.items():
            by_pos.setdefault(pos, []).append(v)

        drawn = {}
        for pos, players in by_pos.items():
            if common_seed is None:
                mu = np.array([p["mean"] for p in players])[None, None, :]
                sd = np.array([p["sd"] for p in players])[None, None, :]
                pp = np.array([p["p_play"] for p in players])[None, None, :]
                s = np.maximum(rng.normal(mu, sd, size=(n_sims, n_cols, len(players))), floor)
                played_mask = rng.random((n_sims, n_cols, len(players))) < pp
                s = s * played_mask
            else:
                cols, masks = [], []
                for pl in players:
                    r = _player_rng(common_seed, pl["key"])
                    x = np.maximum(r.normal(pl["mean"], pl["sd"], size=(n_sims, n_cols)), floor)
                    m = r.random((n_sims, n_cols)) < pl["p_play"]
                    cols.append(x * m); masks.append(m)
                s = np.stack(cols, axis=2); played_mask = np.stack(masks, axis=2)
            if weeks is not None:
                on_bye = np.array([[pl.get("bye") == w for pl in players] for w in weeks])
                if on_bye.any():
                    s = s * ~on_bye[None]
                    played_mask = played_mask & ~on_bye[None]
            # A lineup is named BEFORE the week, so it is ordered by expectation and filtered by
            # availability -- not by what the scores turned out to be. Sorting realized points
            # would be hindsight, and it quietly pays teams for bench depth they could never have
            # known to start: it valued a benched third quarterback at 26 season points.
            rank = np.argsort([-pl["mean"] for pl in players], kind="stable")
            s = s[:, :, rank]
            avail = np.take(played_mask, rank, axis=2)
            first = np.argsort(~avail, axis=2, kind="stable")   # available players, best-mean first
            drawn[pos] = np.take_along_axis(s, first, axis=2)

        total = np.zeros((n_sims, n_cols))
        used = {}
        for slot, count in dedicated:
            a = drawn.get(slot)
            if a is None:
                continue
            take = min(count, a.shape[2])
            total += a[:, :, :take].sum(axis=2)
            used[slot] = take

        for slot, count in flexes:
            # pool everything left over at each eligible position and take the best `count`.
            # Exact for any number of flex slots, where charging the pick to a guessed position
            # is not.
            pool = [drawn[pos][:, :, used.get(pos, 0):]
                    for pos in flex_elig.get(slot, []) if pos in drawn
                    and used.get(pos, 0) < drawn[pos].shape[2]]
            if not pool:
                continue
            left = np.concatenate(pool, axis=2)
            take = min(count, left.shape[2])
            total += (-np.sort(-left, axis=2))[:, :, :take].sum(axis=2)

        if offset is not None:
            # keyed to the team, so a trade does not reshuffle how wrong we are about either side
            total = total + offset[:, ti][:, None]
        out[:, ti, :] = total
    return out


def roster_capacity(cfg):
    """Starters plus bench. Every team in a settled league sits exactly on it."""
    return cfg.starting_size + int(cfg.raw["roster"].get("bench", 0))


def apply_trade(est, moves, capacity=None, starters=None):
    """Move players between teams; with a capacity, cut the surplus a full roster cannot hold.

    moves: [(player_id, from_team, to_team), ...]. Rosters in this league are full, so an uneven
    trade forces the receiving side to drop somebody, and the drop is a real cost of the trade.
    The player dropped is the one a manager would drop: lowest expected weekly contribution that
    does not empty a dedicated starting slot (see forced_cuts).
    """
    out = dict(est)
    for pid, a, b in moves:
        key = (a, str(pid))
        if key not in out:
            raise KeyError(f"player {pid!r} is not on team {a!r}")
        out[(b, str(pid))] = out.pop(key)
    if capacity is None:
        return out
    by_team = {}
    for (t, pid), v in out.items():
        by_team.setdefault(t, []).append((pid, v))
    for t, players in by_team.items():
        for v in forced_cuts([w for _, w in players], capacity, starters):
            pid = next(pid for pid, w in players if w is v)
            del out[(t, pid)]
    return out


def dedicated_starters(cfg):
    """Positions with a slot of their own and how many: {"QB": 1, "K": 1, "DST": 1, ...}."""
    flex = cfg.raw["roster"].get("flex_eligibility", {})
    return {slot: count for slot, count in cfg.starters if slot not in flex}


def forced_cuts(players, capacity, starters=None):
    """Which players a full roster drops to make room: the lowest expected contribution, except
    that nobody is cut who is the last holder of a dedicated starting slot.

    A manager taking on a third receiver drops a bench back, not his only defence. The lowest
    mean on a roster is often the kicker or defence (flat 8.85 / 5.59), and cutting one leaves a
    slot that scores nothing every week -- which made receiving a star for free look like a loss
    once bench backs were priced above the defence.
    """
    if capacity is None or len(players) <= capacity:
        return []
    starters = starters or {}
    # "hold" is what a player is worth beyond this season -- keeper surplus in weekly points, set by
    # the trade search from the keeper board -- so a $1 stash with next-year value outlasts a
    # journeyman with the same weekly mean and none.
    order = sorted(players, key=lambda v: v["mean"] * v["p_play"] + v.get("hold", 0.0))
    counts = {}
    for v in players:
        counts[v["pos"]] = counts.get(v["pos"], 0) + 1
    n_cut = len(players) - capacity
    cuts = []
    for v in order:
        if len(cuts) == n_cut:
            break
        if counts.get(v["pos"], 0) - 1 < starters.get(v["pos"], 0):
            continue
        counts[v["pos"]] -= 1
        cuts.append(v)
    for v in order:                                   # nothing else to give: cut the lowest anyway
        if len(cuts) == n_cut:
            break
        if not any(v is c for c in cuts):
            cuts.append(v)
    return cuts


def evaluate_trade(cfg, est, moves, n_sims=20000, seed=0, played_weeks=None, mean_se=MEAN_SE):
    """Run the season with and without a trade and report what it changes.

    Both worlds share player draws, team offsets and the schedule, so the difference is the trade
    and not Monte Carlo noise. Reported per team involved, because the same trade is genuinely good
    for one side and bad for the other -- that is the win curve, not a matter of opinion.
    """
    est = {(t, str(p)): {**v, "key": str(p)} for (t, p), v in est.items()}
    need, _ = weeks_needed(cfg, played_weeks)
    weeks = week_columns(cfg, played_weeks)
    before = run(cfg, player_team_scores(cfg, est, n_sims, need, rng=np.random.default_rng(seed),
                                         mean_se=mean_se, common_seed=seed, weeks=weeks), played_weeks)
    after = run(cfg, player_team_scores(cfg, apply_trade(est, moves, roster_capacity(cfg),
                                                         dedicated_starters(cfg)),
                                        n_sims, need, rng=np.random.default_rng(seed),
                                        mean_se=mean_se, common_seed=seed, weeks=weeks), played_weeks)
    involved = sorted({m[1] for m in moves} | {m[2] for m in moves})
    delta = {}
    for t in involved:
        b, a = before["teams"][t], after["teams"][t]
        delta[t] = {k: {"before": b[k], "after": a[k], "delta": a[k] - b[k]}
                    for k in ("p_title", "p_playoffs", "exp_wins", "exp_points_for")}
    return {"delta": delta, "before": before["teams"], "after": after["teams"], "moves": moves}
