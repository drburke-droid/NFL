"""Per-player rest-of-season estimates, in the league's own scoring.

The simulator's team-level generator knows nothing about who is on a roster, so it cannot price a
trade. This produces what it needs instead: for every rostered player, a weekly mean, a weekly
standard deviation and a probability of playing.

Three things here are measured rather than assumed.

Scoring is the league's, not the projection pipeline's. Kuhn and Friends pays 6 for a passing
touchdown and -1 for an interception where nflverse PPR pays 4 and -2, so nflverse stat lines are
re-scored through the league config. A quarterback is worth about 6 points a week more here than
the accuracy page thinks, and using the wrong scoring would misvalue every quarterback trade in
the same direction.

The weekly spread grows with the player, so a single number will not do: measured across 1,154
player-seasons of 2023-25 with at least eight games, sd = 0.383 x ppg + 2.23. A 22-point-a-week
player swings by 9.4, a 2-point player by 2.5 -- but in RELATIVE terms it inverts, 0.41 against
1.18. That is the statistical reason a bad team should want stars and a good team should want
floor, and it falls out rather than being asserted.

The blend of this year and last follows the shrinkage curve measured on eleven seasons of
nflverse: weight on to-date scoring rises from 0.46 at week 3 to about 0.72 by week 13 and never
reaches 1. games/(games + 4) reproduces that curve to within a point or two, so that is the form
used.

What this does NOT do: it does not use the weekly projection model, even though the rest-of-season
analysis found a single next-week projection beats to-date scoring outright (MAE 2.645 vs 2.746 on
2025). That needs the walk-forward pipeline run per player per week, and the honest note is that
this module is the fallback the analysis named, not the better thing it found.
"""
import re
import numpy as np, pandas as pd

NFLVERSE = ("https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
            "stats_player_week_{season}.parquet")

# nflverse column -> the stat name the league config scores
STAT_MAP = {"passing_yards": "pass_yds", "passing_tds": "pass_tds",
            "passing_interceptions": "pass_int", "rushing_yards": "rush_yds",
            "rushing_tds": "rush_tds", "receptions": "rec", "receiving_yards": "rec_yds",
            "receiving_tds": "rec_tds", "fumbles_lost": "fumbles_lost"}

SD_SLOPE, SD_INTERCEPT = 0.383, 2.229   # measured, 1154 player-seasons 2023-25, league scoring
PRIOR_GAMES = 4.0                       # reproduces the measured 0.46 -> 0.72 shrinkage curve
# Availability rises steeply with a player's level, so one number for everyone is wrong in both
# directions: 90% of players projected under 2 points score nothing in a given week, and none
# projected above 17 do. A flat 0.81 benched real starters a fifth of the time, which cost the
# simulated league 12 points a week once lineups stopped being set with hindsight.
P_PLAY = 0.81                           # the pooled figure, kept as the fallback
P_PLAY_BASE, P_PLAY_SLOPE, P_PLAY_CAP = 0.75, 0.030, 0.97
P_PLAY_ABSENT = 0.45                    # a player with no snaps at all this season is hurt, not gone
KDST_PPG = {"K": 8.85, "DST": 5.59}     # ESPN season totals / 17; K and DST are not differentiated
# Re-solved after lineups stopped being set with hindsight, which had inflated every team by ~12
# points a week and so made the earlier 0.70 too aggressive.
# Player means built from a couple of games are noisy, and a lineup optimiser compounds that: it
# keeps whichever estimates happen to be high, so a roster's summed mean is biased upward. Left
# raw, simulated team means spread with sd 11.7 where this league's measured spread is 8.7 (8.2
# over 2023-25, scaled to 2026's higher scoring). Pulling each player 30% toward his positional
# mean lands both moments: between-team 8.70 against 8.68, within-team 21.8 against 21.7.
MEAN_SHRINK = 0.80
TEAM_FIX = {"LVR": "LV", "JAC": "JAX", "LAR": "LA", "WSH": "WAS", "ARZ": "ARI"}
SKILL = ("QB", "RB", "WR", "TE")


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def weekly_in_league_scoring(cfg, seasons):
    """nflverse weekly stat lines re-scored through this league's rules."""
    frames = []
    for yr in seasons:
        a = pd.read_parquet(NFLVERSE.format(season=yr))
        a = a[a.season == yr]
        keep = ["player_id", "player_display_name", "position", "team", "season", "week"]
        cols = keep + [c for c in STAT_MAP if c in a.columns] + \
               [c for c in a.columns if "fumbles_lost" in c]
        a = a[[c for c in dict.fromkeys(cols) if c in a.columns]].copy()
        a["fumbles_lost"] = a[[c for c in a.columns if "fumbles_lost" in c]].sum(axis=1)
        frames.append(a)
    d = pd.concat(frames, ignore_index=True).fillna(0)
    line = {v: d[k] for k, v in STAT_MAP.items() if k in d.columns}
    ps = cfg.raw["scoring"]["per_stat"]
    d["pts"] = sum(line[k] * ps.get(k, 0) for k in line)
    # Threshold bonuses are part of the league's scoring, so a 100-yard game has to be worth what
    # the league pays for it. Omitting them would quietly undervalue exactly the players who earn
    # them. (Kuhn and Friends has none; other configs do.)
    for b in cfg.raw["scoring"].get("bonuses", []):
        if b["stat"] in line:
            d["pts"] = d["pts"] + (line[b["stat"]] >= float(b["threshold"])) * float(b["points"])
    d["key"] = d.player_display_name.map(norm)
    return d[d.position.isin(SKILL)]


def replacement_level(weekly, cfg):
    """Points a freely available player at each position is worth -- the floor a rookie gets.

    Ranked by season average and cut at the last starter the league can field: a 12-team league
    starting one quarterback has a QB12 replacement, two backs plus a flex a deeper one.
    """
    n_teams = len(cfg.teams)
    slots = dict(cfg.starters)
    depth = {p: n_teams * (slots.get(p, 0) + (1 if p in ("RB", "WR", "TE") else 0)) for p in SKILL}
    last = weekly.season.max()
    out = {}
    for p in SKILL:
        avg = (weekly[(weekly.position == p) & (weekly.season == last)]
               .groupby("player_id").pts.agg(["mean", "size"]))
        avg = avg[avg["size"] >= 6]["mean"].sort_values(ascending=False)
        i = min(max(depth[p] - 1, 0), max(len(avg) - 1, 0))
        out[p] = float(avg.iloc[i]) if len(avg) else 0.0
    return out


def ros_estimates(cfg, current_season=None, prior_season=None, weekly=None,
                  mean_shrink=MEAN_SHRINK):
    """{(team_id, player_id): {mean, sd, p_play, source}} for every rostered player.

    The fallback chain is explicit because every step is a different kind of ignorance:
      blend        both this year and last -> shrink one toward the other
      current_only rookie or new arrival with snaps -> shrink toward replacement level
      prior_only   no snaps at all this year -> last year's rate, and probably injured
      replacement  never played -> the freely available alternative
    """
    current_season = current_season or cfg.raw["season"]
    prior_season = prior_season or current_season - 1
    if weekly is None:
        weekly = weekly_in_league_scoring(cfg, (prior_season, current_season))

    cur = weekly[weekly.season == current_season].groupby("key").pts.agg(["mean", "size"])
    pri = weekly[weekly.season == prior_season].groupby("key").pts.agg(["mean", "size"])
    repl = replacement_level(weekly, cfg)

    est = {}
    for t in cfg.teams:
        for e in t["roster"]:
            pos, k = e["pos"], norm(e["name"])
            if pos not in SKILL:
                mean, sd, p, src = KDST_PPG.get(pos, 5.0), None, 1.0, "kdst_flat"
            else:
                c = cur.loc[k] if k in cur.index else None
                p0 = pri.loc[k] if k in pri.index else None
                p, src = P_PLAY, "blend"
                if c is not None and p0 is not None:
                    n = float(c["size"])
                    mean = (c["mean"] * n + p0["mean"] * PRIOR_GAMES) / (n + PRIOR_GAMES)
                elif c is not None:
                    n = float(c["size"])
                    mean = (c["mean"] * n + repl[pos] * PRIOR_GAMES) / (n + PRIOR_GAMES)
                    src = "current_only"
                elif p0 is not None:
                    mean, p, src = float(p0["mean"]), P_PLAY_ABSENT, "prior_only"
                else:
                    mean, src = repl[pos], "replacement"
                sd = None
            if sd is None:
                sd = SD_SLOPE * max(mean, 0.0) + SD_INTERCEPT
            est[(t["team_id"], e["player_id"])] = {
                "name": e["name"], "pos": pos, "mean": float(mean), "sd": float(sd),
                "p_play": float(p), "source": src, "raw_mean": float(mean)}

    if mean_shrink < 1.0:
        pos_mean = {}
        for v in est.values():
            pos_mean.setdefault(v["pos"], []).append(v["raw_mean"])
        pos_mean = {k: float(np.mean(v)) for k, v in pos_mean.items()}
        for v in est.values():
            if v["pos"] in SKILL:                  # K and DST are already flat
                base = pos_mean[v["pos"]]
                v["mean"] = base + mean_shrink * (v["raw_mean"] - base)
                v["sd"] = SD_SLOPE * max(v["mean"], 0.0) + SD_INTERCEPT

    # availability follows the final mean, so it is set after any shrinkage. A player with no
    # snaps at all this season keeps his lowered figure -- being hurt is why he has none.
    for v in est.values():
        if v["pos"] in SKILL and v["source"] != "prior_only":
            v["p_play"] = min(P_PLAY_CAP, P_PLAY_BASE + P_PLAY_SLOPE * max(v["mean"], 0.0))
    return est
