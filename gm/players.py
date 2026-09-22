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

The MEAN is a fitted model, not a blend. The first version blended this season with last at
games/(games + 4) and nothing else; scored walk-forward on 2019-25 (scripts/ros_player_study.py)
it ran 0.54 points high on everyone and 1.42 high on the top quarter of each position -- it
believed hot starts. The replacement is a per-position ridge on the full history since 2012: this
season (mean, last four, best game, games), the prior two seasons, the career, and the FFA
projection for the coming week, fitted by scripts/ros_player_study.py --fit-production and stored
in gm/ros_model.json so the runtime needs only numpy. Walk-forward MAE 3.11 against 3.31 for the
blend, rank correlation 0.58 against 0.56, bias within 0.05 on the players who get traded. Age
and archetype were tested in the same study and added nothing, which matches every prior study
in this repo. The blend survives as model="blend" for comparison.
"""
import glob, json, os, re
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
FFA_DIR = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
MODEL = os.path.join(ROOT, "gm", "ros_model.json")
SCHEDULE = os.path.join(ROOT, "data", "schedule_{season}.csv")
NFLVERSE = ("https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
            "stats_player_week_{season}.parquet")
FIRST_SEASON = 2012

# nflverse column -> the stat name the league config scores
STAT_MAP = {"passing_yards": "pass_yds", "passing_tds": "pass_tds",
            "passing_interceptions": "pass_int", "rushing_yards": "rush_yds",
            "rushing_tds": "rush_tds", "receptions": "rec", "receiving_yards": "rec_yds",
            "receiving_tds": "rec_tds", "fumbles_lost": "fumbles_lost"}
# FFA raw-stat column -> the same stat names
FFA_MAP = {"pass_yds": "pass_yds", "pass_tds": "pass_tds", "pass_int": "pass_int", "rush_yds": "rush_yds",
           "rush_tds": "rush_tds", "fumbles_lost": "fumbles_lost", "rec": "rec", "rec_yds": "rec_yds",
           "rec_tds": "rec_tds"}

SD_SLOPE, SD_INTERCEPT = 0.383, 2.229   # measured, 1154 player-seasons 2023-25, league scoring
PRIOR_GAMES = 4.0                       # the blend's prior weight; reproduces the 0.46 -> 0.72 curve
# Availability rises steeply with a player's level, so one number for everyone is wrong in both
# directions: 90% of players projected under 2 points score nothing in a given week, and none
# projected above 17 do. A flat 0.81 benched real starters a fifth of the time, which cost the
# simulated league 12 points a week once lineups stopped being set with hindsight.
P_PLAY = 0.81                           # the pooled figure, kept as the fallback
P_PLAY_BASE, P_PLAY_SLOPE, P_PLAY_CAP = 0.75, 0.030, 0.97
P_PLAY_ABSENT = 0.45                    # a player with no snaps at all this season is hurt, not gone
KDST_PPG = {"K": 8.85, "DST": 5.59}     # ESPN season totals / 17; K and DST are not differentiated
# The waiver wire is a real roster spot. A slot nobody on the roster can fill -- the only tight end
# on bye, two backs hurt the same week -- is filled by the best free agent, not left empty, so
# every team carries one virtual "waiver" player per position at the level of the best unrostered
# players. Five deep because the single best free agent is often a data artefact (a backup who
# played twice) and a manager picks from what is actually there on Tuesday.
FA_DEPTH = 5
# Player means built from a couple of games are noisy, and a lineup optimiser compounds that: it
# keeps whichever estimates happen to be high, so a roster's summed mean is biased upward. Left
# raw, the blend's simulated team means spread with sd 11.7 where this league's measured spread is
# 8.7 (8.2 over 2023-25, scaled to 2026's higher scoring). Pulling each player 30% toward his
# positional mean lands both moments for the blend.
# The ridge needs none of it. A fitted estimate is already the regressed value, so the spread of
# estimates is SUPPOSED to be narrower than the spread of true strength: what is left is the
# uncertainty about each team, which the simulator adds separately (simulate.MEAN_SE, 5.1 a
# week). Measured on the 2026 rosters the ridge spreads team means by 6.7; combined with the 5.1
# that is 8.4 against the 8.7 target. The blend at 0.80 spread them by 8.8 on its own, so with the
# uncertainty on top it was over-dispersed at about 10.2. Shrinking the ridge further (0.9 -> 6.0,
# 0.8 -> 5.4) only moves it away from the target.
MEAN_SHRINK = 0.80
MEAN_SHRINK_RIDGE = 1.00
TEAM_FIX = {"LVR": "LV", "JAC": "JAX", "LAR": "LA", "WSH": "WAS", "ARZ": "ARI"}
SKILL = ("QB", "RB", "WR", "TE")

# Feature names the ridge is fitted on. The order is the contract with gm/ros_model.json.
HIST_COLS = ["ytd_mean", "ytd_games", "last4_mean", "ytd_max", "p1_mean", "p1_games", "p2_mean", "p2_games",
             "car_mean", "car_games", "car_seasons", "week"]
SHRINK_COLS = ["ytd_w", "ytd_wt", "p1_w", "p1_wt"]     # the blend's own shape, for a linear model to use
PROJ_COLS = ["ffa_next", "ffa_sd"]
MODEL_COLS = HIST_COLS + SHRINK_COLS + PROJ_COLS

_WEEKLY_CACHE = {}


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def _score(cfg, line):
    """Points under the league's rules for a dict of stat-name -> Series."""
    ps = cfg.raw["scoring"]["per_stat"]
    pts = sum(line[k] * ps.get(k, 0) for k in line)
    # Threshold bonuses are part of the league's scoring, so a 100-yard game has to be worth what
    # the league pays for it. Omitting them would quietly undervalue exactly the players who earn
    # them. (Kuhn and Friends has none; other configs do.)
    for b in cfg.raw["scoring"].get("bonuses", []):
        if b["stat"] in line:
            pts = pts + (line[b["stat"]] >= float(b["threshold"])) * float(b["points"])
    return pts


def _read_season(yr, settled=False):
    """One season of nflverse weekly rows.

    A settled season is read from data/nflverse_cache and downloaded into it the first time (the
    directory is gitignored, so a fresh clone fetches ~25 MB once). The current season and the
    one before are downloaded fresh on every run, as they always were, because they change
    weekly; the cache is only the fallback when the download fails.
    """
    local = os.path.join(CACHE, f"stats_player_week_{yr}.parquet")
    if settled and os.path.exists(local):
        return pd.read_parquet(local)
    try:
        a = pd.read_parquet(NFLVERSE.format(season=yr))
    except Exception:
        if os.path.exists(local):
            return pd.read_parquet(local)
        raise
    if settled:
        os.makedirs(CACHE, exist_ok=True)
        a.to_parquet(local)
    return a


def weekly_in_league_scoring(cfg, seasons, current_season=None):
    """nflverse weekly stat lines re-scored through this league's rules. Regular season only."""
    current_season = current_season or max(seasons)
    frames = []
    for yr in seasons:
        a = _read_season(yr, settled=yr < current_season - 1)
        a = a[a.season == yr]
        if "season_type" in a.columns:
            a = a[a.season_type == "REG"]
        keep = ["player_id", "player_display_name", "position", "team", "season", "week"]
        cols = keep + [c for c in STAT_MAP if c in a.columns and c != "fumbles_lost"]
        a = a[[c for c in dict.fromkeys(cols) if c in a.columns]].assign(
            # nflverse carries the three components AND their total; summing every column whose
            # name contains "fumbles_lost" counted each fumble twice.
            fumbles_lost=(a["fumbles_lost_total"] if "fumbles_lost_total" in a.columns else
                          a[[c for c in a.columns if c.endswith("_fumbles_lost")]].sum(axis=1)
                          if any(c.endswith("_fumbles_lost") for c in a.columns) else 0.0))
        frames.append(a)
    d = pd.concat(frames, ignore_index=True).fillna(0)
    d["pts"] = _score(cfg, {v: d[k] for k, v in STAT_MAP.items() if k in d.columns})
    d["key"] = d.player_display_name.map(norm)
    d = d[d.position.isin(SKILL)]
    return d.drop_duplicates(["player_id", "season", "week"])


def _weekly_all(cfg, current_season):
    """Every season from FIRST_SEASON to the current one, cached per scoring config."""
    key = (json.dumps(cfg.raw["scoring"], sort_keys=True), current_season)
    if key not in _WEEKLY_CACHE:
        _WEEKLY_CACHE[key] = weekly_in_league_scoring(cfg, range(FIRST_SEASON, current_season + 1),
                                                      current_season)
    return _WEEKLY_CACHE[key]


def bye_weeks(season):
    """{nfl team: bye week} from the season's schedule file; empty if the file is not there."""
    p = SCHEDULE.format(season=season)
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p, usecols=["game_type", "week", "away_team", "home_team"])
    s = s[s.game_type == "REG"]
    teams = set(s.away_team) | set(s.home_team)
    out = {}
    for w, g in s.groupby("week"):
        for t in teams - set(g.away_team) - set(g.home_team):
            out[t] = int(w)
    return out


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


# ----------------------------------------------------------------------------- features
def ffa_file_scored(cfg, path):
    """One FFA weekly raw-stats file as (key, position, ffa_next, ffa_sd) in league scoring."""
    ps = cfg.raw["scoring"]["per_stat"]
    d = pd.read_csv(path, low_memory=False)
    d = d[d.position.isin(SKILL)].copy()
    d["ffa_next"] = sum(d[c].fillna(0) * ps.get(v, 0) for c, v in FFA_MAP.items() if c in d.columns)
    d["ffa_sd"] = np.sqrt(sum((d[c + "_sd"].fillna(0) * ps.get(v, 0)) ** 2 for c, v in FFA_MAP.items()
                              if c + "_sd" in d.columns))
    d["key"] = d.player.map(norm)
    return d[["key", "position", "ffa_next", "ffa_sd"]].drop_duplicates(["key", "position"])


def ffa_next_week(cfg, season, week):
    """FFA's projection for week+1 -- or, before that file is uploaded, the current week's.

    Returns (frame, stale). The stand-in matters: a Tuesday trade is priced on the projection made
    for the week just played, and the flag says so.
    """
    for w, stale in ((week + 1, False), (week, True)):
        p = os.path.join(FFA_DIR, f"raw_stats_{season}_wk{w}.csv")
        if os.path.exists(p):
            return ffa_file_scored(cfg, p), stale
    return None, True


def history_features(weekly, season, week, ffa=None):
    """Per-player features known at the end of `week` of `season`, indexed by nflverse id.

    Everything the ridge uses: this season to date, the two prior seasons, the career, and the FFA
    projection for the coming week (joined by name). Shared by the study that fits the model and
    the estimator that applies it, so the two cannot drift apart.
    """
    ws = weekly[weekly.season == season]
    ytd = ws[ws.week <= week]
    hist = weekly[weekly.season < season]
    tot = hist.groupby(["player_id", "season"]).pts.agg(s_mean="mean", s_games="size").reset_index()
    p1 = tot[tot.season == season - 1].set_index("player_id")
    p2 = tot[tot.season == season - 2].set_index("player_id")
    car = tot.groupby("player_id").apply(
        lambda g: pd.Series({"car_mean": np.average(g.s_mean, weights=g.s_games),
                             "car_games": float(g.s_games.sum()), "car_seasons": float(len(g))}),
        include_groups=False) if len(tot) else pd.DataFrame(columns=["car_mean", "car_games", "car_seasons"])
    ident = (weekly.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
             .set_index("player_id")[["position", "player_display_name", "key"]])
    f = ident.copy()
    f = f.join(ytd.groupby("player_id").pts.agg(ytd_mean="mean", ytd_games="size"))
    f = f.join(ytd[ytd.week > week - 4].groupby("player_id").pts.mean().rename("last4_mean"))
    f = f.join(ytd.groupby("player_id").pts.max().rename("ytd_max"))
    f = f.join(p1[["s_mean", "s_games"]].rename(columns={"s_mean": "p1_mean", "s_games": "p1_games"}))
    f = f.join(p2[["s_mean", "s_games"]].rename(columns={"s_mean": "p2_mean", "s_games": "p2_games"}))
    f = f.join(car)
    f["week"] = float(week)
    f["ytd_wt"] = f.ytd_games.fillna(0) / (f.ytd_games.fillna(0) + PRIOR_GAMES)
    f["ytd_w"] = f.ytd_mean.fillna(0) * f.ytd_wt
    f["p1_wt"] = f.p1_games.fillna(0) / (f.p1_games.fillna(0) + PRIOR_GAMES)
    f["p1_w"] = f.p1_mean.fillna(0) * f.p1_wt
    if ffa is not None and len(ffa):
        f = f.reset_index().merge(ffa, on=["key", "position"], how="left").set_index("player_id")
    else:
        f["ffa_next"], f["ffa_sd"] = np.nan, np.nan
    # a player is in the frame only if he has some history: this season, or any season before
    has = f.ytd_games.notna() | f.p1_games.notna() | f.car_games.notna()
    return f[has]


def ridge_design(feats, spec):
    """The design matrix the stored ridge expects: medians for gaps, then a flag per gappy column."""
    X = feats[spec["cols"]].astype(float).copy()
    for c in spec["cols"]:
        X[c] = X[c].fillna(spec["medians"][c])
    for c in spec["na_cols"]:
        X[c + "_na"] = feats[c].isna().astype(float)
    return X


def load_model(path=MODEL):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ridge_predict(model, feats):
    """Fitted rest-of-season mean for every row of a history_features frame."""
    out = pd.Series(np.nan, index=feats.index)
    for p, spec in model["positions"].items():
        rows = feats[feats.position == p]
        if rows.empty:
            continue
        X = ridge_design(rows, spec)
        out.loc[rows.index] = X.values @ np.array(spec["coef"]) + spec["intercept"]
    return out


# ----------------------------------------------------------------------------- estimates
def ros_estimates(cfg, current_season=None, prior_season=None, weekly=None,
                  mean_shrink=None, model="ridge"):
    """{(team_id, player_id): {mean, sd, p_play, source, model}} for every rostered player.

    `source` names what data the player had, which is a different question from how it was
    combined:
      blend        this year and last          (blend: shrink one toward the other)
      current_only rookie or new arrival       (blend: shrink toward replacement level)
      prior_only   no snaps at all this year   (last year's rate, and probably injured)
      replacement  never played                (the freely available alternative)
    model="ridge" (the default when gm/ros_model.json exists) fits the mean from the full history
    and the FFA projection; model="blend" is the original games/(games+4) rule.
    """
    current_season = current_season or cfg.raw["season"]
    prior_season = prior_season or current_season - 1
    use_ridge = model == "ridge" and os.path.exists(MODEL)
    if weekly is None:
        weekly = (_weekly_all(cfg, current_season) if use_ridge
                  else weekly_in_league_scoring(cfg, (prior_season, current_season)))
    if mean_shrink is None:
        mean_shrink = MEAN_SHRINK_RIDGE if use_ridge else MEAN_SHRINK

    cur = weekly[weekly.season == current_season].groupby("key").pts.agg(["mean", "size"])
    pri = weekly[weekly.season == prior_season].groupby("key").pts.agg(["mean", "size"])
    repl = replacement_level(weekly, cfg)
    byes = bye_weeks(current_season)
    rostered = {(norm(e["name"]), e["pos"]) for t in cfg.teams for e in t["roster"]}
    waiver = dict(KDST_PPG)
    waiver.update({p: repl[p] for p in SKILL})

    fitted = {}
    if use_ridge:
        played = weekly[weekly.season == current_season].week
        week = int(played.max()) if len(played) else 0
        ffa, stale = ffa_next_week(cfg, current_season, week)
        feats = history_features(weekly, current_season, week, ffa)
        pred = ridge_predict(load_model(), feats)
        # by name, the id that played most recently -- the roster carries ESPN ids, not nflverse
        latest = feats.reset_index().sort_values(["ytd_games", "p1_games"], na_position="first")
        has_ffa = {}
        for _, r in latest.iterrows():
            fitted[(r["key"], r["position"])] = float(pred[r["player_id"]])
            has_ffa[(r["key"], r["position"])] = bool(pd.notna(r["ffa_next"]))
        skill_rostered = [k for k in rostered if k[1] in SKILL]
        ffa_cover = float(np.mean([has_ffa.get(k, False) for k in skill_rostered])) if skill_rostered else 0.0
        # the best free agents at each position, by the same model, with at least one game played
        fa = feats[feats.ytd_games.notna() & np.isfinite(pred)].assign(pred=pred)
        fa = fa[[(k, p) not in rostered for k, p in zip(fa.key, fa.position)]]
        for p in SKILL:
            top = fa[fa.position == p].pred.sort_values(ascending=False).head(FA_DEPTH)
            if len(top):
                waiver[p] = float(max(top.mean(), 0.0))
        ros_estimates.last_ridge = {"week": week, "ffa_stale": stale, "ffa_coverage": ffa_cover,
                                    "fitted": sum(k in fitted for k in skill_rostered),
                                    "rostered": len(skill_rostered), "waiver": dict(waiver)}

    est = {}
    for t in cfg.teams:
        for e in t["roster"]:
            pos, k = e["pos"], norm(e["name"])
            if pos not in SKILL:
                mean, sd, p, src = KDST_PPG.get(pos, 5.0), None, 1.0, "kdst_flat"
                how = "flat"
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
                how = "blend"
                if use_ridge and (k, pos) in fitted and np.isfinite(fitted[(k, pos)]):
                    mean, how = max(fitted[(k, pos)], 0.0), "ridge"
                sd = None
            if sd is None:
                sd = SD_SLOPE * max(mean, 0.0) + SD_INTERCEPT
            nfl = TEAM_FIX.get(e.get("nfl_team"), e.get("nfl_team"))
            est[(t["team_id"], e["player_id"])] = {
                "name": e["name"], "pos": pos, "mean": float(mean), "sd": float(sd),
                "p_play": float(p), "source": src, "model": how, "raw_mean": float(mean),
                "nfl_team": nfl, "bye": byes.get(nfl)}

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
    # the waiver wire, one virtual player per position, keyed to the pseudo-team "FA"
    for pos, m in waiver.items():
        p = 1.0 if pos in KDST_PPG else min(P_PLAY_CAP, P_PLAY_BASE + P_PLAY_SLOPE * m)
        est[("FA", pos)] = {"name": f"waiver {pos}", "pos": pos, "mean": float(m),
                            "sd": float(SD_SLOPE * m + SD_INTERCEPT), "p_play": float(p),
                            "source": "waiver", "model": "waiver", "raw_mean": float(m),
                            "nfl_team": None, "bye": None}
    return est


ros_estimates.last_ridge = None
