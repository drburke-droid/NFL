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


def shrunk_team_scores(cfg, n_sims, n_cols, sd=28.0, prior_games=4.0, rng=None):
    """A stand-in score generator so the engine runs end to end. Not a good model.

    With one or two games played a team's scoring average is close to pure noise, so it is pulled
    hard toward the league mean -- prior_games is how many games of league-average pull to apply.
    The weekly standard deviation is a flat default and is the weakest input in the whole pipeline;
    replacing this function with a player-level model is the point of the exercise.
    """
    rng = rng or np.random.default_rng(0)
    played = np.array([max(1, t["record"]["w"] + t["record"]["l"] + t["record"]["t"])
                       for t in cfg.teams], float)
    obs = np.array([t["points_for"] for t in cfg.teams], float) / played
    mu = (obs * played + obs.mean() * prior_games) / (played + prior_games)
    return rng.normal(mu[None, :, None], sd, size=(n_sims, len(mu), n_cols))
