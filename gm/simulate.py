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


def weeks_needed(cfg, played_weeks=()):
    """How many score columns run() expects: remaining regular weeks, then one per playoff round."""
    future = [w for w in cfg.regular_weeks if w not in set(played_weeks)]
    return len(future) + len(cfg.playoff_weeks), future


def run(cfg, weekly_scores, played_weeks=()):
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


def player_team_scores(cfg, est, n_sims, n_cols, rng=None, floor=SCORE_FLOOR):
    """Weekly team scores built from a roster, so a trade can be run through the season.

    Each player either plays (Bernoulli on p_play) and draws from his own distribution, or scores
    nothing. The lineup is then set optimally, which matters more than it sounds: a roster is worth
    the best nine it can field in a given week, not the sum of its parts, so depth at a position is
    worth much less than the same points spread across positions. That is why a trade of two good
    backs for one great one can lose value even when the totals match.

    The optimal lineup is a sort, not a search. Fill each dedicated slot with the best at that
    position; the flex then takes the best player left over, which can only be the next one down at
    some eligible position. With every slot contributing to one sum, greedy is exact.
    """
    rng = rng or np.random.default_rng(0)
    order = [t["team_id"] for t in cfg.teams]
    flex_elig = cfg.raw["roster"].get("flex_eligibility", {})
    dedicated = [(s, c) for s, c in cfg.starters if s not in flex_elig]
    flexes = [(s, c) for s, c in cfg.starters if s in flex_elig]
    out = np.zeros((n_sims, len(order), n_cols))

    for ti, tid in enumerate(order):
        roster = [v for (t, _), v in est.items() if t == tid]
        by_pos = {}
        for v in roster:
            by_pos.setdefault(v["pos"], []).append(v)

        drawn = {}
        for pos, players in by_pos.items():
            mu = np.array([p["mean"] for p in players])[None, None, :]
            sd = np.array([p["sd"] for p in players])[None, None, :]
            pp = np.array([p["p_play"] for p in players])[None, None, :]
            s = rng.normal(mu, sd, size=(n_sims, n_cols, len(players)))
            s = np.maximum(s, floor) * (rng.random((n_sims, n_cols, len(players))) < pp)
            drawn[pos] = -np.sort(-s, axis=2)            # best first

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

        out[:, ti, :] = total
    return out
