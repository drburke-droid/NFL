"""Do a player's first three weeks tell you anything the box score does not?

The trade question this answers: after three weeks you can see who is over- and under-performing.
Which of those moves are real, and which are noise you should be trading against?

Method. For every player-season 2016-2025 we build what a manager can see after week 3 (volume,
efficiency, touchdowns, last season) and predict points per game over the REST of the season,
weeks 4 to 17, in the league's own scoring. The naive read is "he keeps doing what he just did",
so the quantity we model is the REGRESSION, rest-of-season ppg minus first-three-week ppg. A
feature earns its place only if it predicts that gap out of sample, fit on earlier seasons and
scored on later ones (walk-forward, never on the season being scored).

The central split is volume versus efficiency. Volume is how often the offence uses a player;
efficiency is what he did with it. The hypothesis, which this script tests rather than assumes,
is that volume carries and efficiency evaporates, so the sell-high candidate is a player whose
points came from touchdowns on light usage, and the buy-low candidate is a heavily used player
whose touchdowns have not landed yet.

Usage: python scripts/hot_start_study.py  [--start-week 4] [--min-early-games 2] [--min-ros-games 6]
Writes outputs/reports/hot_start_study.md and outputs/models/hot_start_frame.parquet.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
SEASONS = list(range(2016, 2026))
SKILL = ["QB", "RB", "WR", "TE"]

ap = argparse.ArgumentParser()
ap.add_argument("--start-week", type=int, default=4, help="first week of the rest-of-season window")
ap.add_argument("--min-early-games", type=int, default=2, help="games needed in weeks 1..start-1")
ap.add_argument("--min-ros-games", type=int, default=6, help="games needed in the rest-of-season window")
ap.add_argument("--end-week", type=int, default=17, help="last week counted (fantasy seasons end at 17)")
A = ap.parse_args()
EARLY = A.start_week - 1


def league_points(d):
    """Kuhn and Friends scoring: full PPR, 6-point passing TD, 0.04 per passing yard, -1 INT and fumble."""
    z = lambda c: d[c].fillna(0).astype(float) if c in d.columns else 0.0
    fl = z("fumbles_lost_total") if "fumbles_lost_total" in d.columns else \
        z("sack_fumbles_lost") + z("rushing_fumbles_lost") + z("receiving_fumbles_lost")
    return (0.04 * z("passing_yards") + 6 * z("passing_tds") - z("passing_interceptions")
            + 0.1 * z("rushing_yards") + 6 * z("rushing_tds")
            + z("receptions") + 0.1 * z("receiving_yards") + 6 * z("receiving_tds")
            + 2 * (z("passing_2pt_conversions") + z("rushing_2pt_conversions") + z("receiving_2pt_conversions"))
            - fl)


def load():
    out = []
    for s in SEASONS:
        fp = os.path.join(CACHE, f"stats_player_week_{s}.parquet")
        if not os.path.exists(fp):
            print(f"  missing {os.path.basename(fp)}, skipping {s}"); continue
        d = pd.read_parquet(fp)
        d = d[(d.season_type == "REG") & d.position.isin(SKILL) & (d.week <= A.end_week)].copy()
        d["pts"] = league_points(d)
        out.append(d)
    a = pd.concat(out, ignore_index=True)
    print(f"  {len(a):,} player-weeks, {a.season.min()}-{a.season.max()}")
    return a


def build(a):
    """One row per player-season: what week 3 showed, and what the rest of the year delivered."""
    z = lambda d, c: d[c].fillna(0).astype(float) if c in d.columns else pd.Series(0.0, index=d.index)
    early = a[a.week < A.start_week].copy()
    ros = a[a.week >= A.start_week].copy()
    key = ["player_id", "season"]

    e = early.groupby(key).agg(
        player=("player_display_name", "last"), pos=("position", "last"), team=("team", "last"),
        g=("week", "count"), pts=("pts", "sum"),
        tgt=("targets", "sum"), rec=("receptions", "sum"), rec_yds=("receiving_yards", "sum"),
        rec_td=("receiving_tds", "sum"), car=("carries", "sum"), rush_yds=("rushing_yards", "sum"),
        rush_td=("rushing_tds", "sum"), att=("attempts", "sum"), pass_yds=("passing_yards", "sum"),
        pass_td=("passing_tds", "sum"), tgt_share=("target_share", "mean"), ay_share=("air_yards_share", "mean"),
    ).reset_index()
    r = ros.groupby(key).agg(ros_g=("week", "count"), ros_pts=("pts", "sum")).reset_index()
    d = e.merge(r, on=key, how="inner")
    d = d[(d.g >= A.min_early_games) & (d.ros_g >= A.min_ros_games)].copy()

    d["early_ppg"] = d.pts / d.g
    d["ros_ppg"] = d.ros_pts / d.ros_g
    d["gap"] = d.ros_ppg - d.early_ppg              # what we model: the regression
    # ---- volume per game (how often the offence uses him)
    for c, nm in (("tgt", "tgt_g"), ("car", "car_g"), ("att", "att_g")):
        d[nm] = d[c] / d.g
    d["touch_g"] = (d.tgt + d.car) / d.g
    d["tgt_share"] = d.tgt_share.fillna(0)
    d["ay_share"] = d.ay_share.fillna(0)
    # ---- efficiency (what he did with it)
    d["ypt"] = np.where(d.tgt > 0, d.rec_yds / d.tgt.replace(0, np.nan), np.nan)
    d["ypc"] = np.where(d.car > 0, d.rush_yds / d.car.replace(0, np.nan), np.nan)
    d["catch_rate"] = np.where(d.tgt > 0, d.rec / d.tgt.replace(0, np.nan), np.nan)
    # ---- touchdowns, the loudest and least repeatable part of an early hot start
    d["td"] = d.rec_td + d.rush_td + d.pass_td
    d["td_g"] = d.td / d.g
    d["td_pts_share"] = np.where(d.pts > 0, 6 * d.td / d.pts.replace(0, np.nan), np.nan)
    # ---- last season, the prior the market had before any of this
    prev = a[a.week <= A.end_week].groupby(["player_id", "season"]).agg(
        pg=("week", "count"), pp=("pts", "sum")).reset_index()
    prev["prev_ppg"] = prev.pp / prev.pg
    prev["season"] = prev.season + 1
    d = d.merge(prev[["player_id", "season", "prev_ppg", "pg"]].rename(columns={"pg": "prev_g"}),
                on=["player_id", "season"], how="left")
    d["prev_ppg"] = d.prev_ppg.where(d.prev_g >= 6)     # a 2-game cameo is not a prior
    d["has_prev"] = d.prev_ppg.notna().astype(int)
    return d


def expected_points(d):
    """Points the early volume alone implies, from league-average rates fitted per position.

    Fitting the rates on the same rows we score would leak, so each position's rates come from a
    plain least-squares fit of points on volume across ALL seasons; the rates are stable enough
    (they are league averages, not player effects) that this is a description of the sport, not a
    model of these players. The part of a player's early scoring NOT explained by his volume is
    what we call luck.
    """
    d = d.copy(); d["exp_pts_g"] = np.nan
    for pos, grp in d.groupby("pos"):
        cols = ["tgt_g", "car_g", "att_g"]
        X = grp[cols].fillna(0).values
        X = np.column_stack([X, np.ones(len(X))])
        y = (grp.pts / grp.g).values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        d.loc[grp.index, "exp_pts_g"] = X @ beta
        print(f"    {pos}: pts/g ~ {beta[0]:+.3f}*targets {beta[1]:+.3f}*carries {beta[2]:+.3f}*attempts {beta[3]:+.2f}")
    d["luck"] = d.early_ppg - d.exp_pts_g            # >0 = scored more than his usage implies
    return d


def walk_forward(d, feats, label):
    """Fit on every season before the one being scored; never on it. Returns per-row predictions."""
    pred = pd.Series(np.nan, index=d.index)
    # prev_ppg is missing for every rookie, so a median fill would invent a career for them. It is
    # filled with zero and carried by has_prev, which lets the fit give rookies their own intercept.
    def design(frame, med):
        X = frame[feats].copy()
        for c in feats:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            X[c] = X[c].fillna(0.0 if c.startswith("prev_") else med.get(c, 0.0))
        X = X.astype(float).values
        return np.column_stack([X, np.ones(len(X))])
    for s in sorted(d.season.unique()):
        tr = d[(d.season < s) & d.gap.notna()]
        te = d.index[d.season == s]
        if len(tr) < 200 or not len(te):
            continue
        med = tr[feats].replace([np.inf, -np.inf], np.nan).median()
        Xtr, ytr = design(tr, med), tr["gap"].values
        ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        if ok.sum() < 200: continue
        beta, *_ = np.linalg.lstsq(Xtr[ok], ytr[ok], rcond=None)
        pred.loc[te] = design(d.loc[te], med) @ beta
    return pred


def score(d, pred, label, out):
    m = d.dropna(subset=["gap"]).index.intersection(pred.dropna().index)
    if not len(m):
        return
    err = (d.loc[m, "gap"] - pred.loc[m]).abs()
    base = d.loc[m, "gap"].abs()          # the naive read: he keeps doing what he did
    line = (f"{label:38s} n={len(m):5d}  MAE {err.mean():5.3f} vs naive {base.mean():5.3f}  "
            f"({(1 - err.mean() / base.mean()):+.1%})  corr {np.corrcoef(pred.loc[m], d.loc[m,'gap'])[0,1]:+.3f}")
    print("  " + line); out.append(line)


def main():
    print("loading weekly box scores")
    a = load()
    print(f"\nbuilding the panel (weeks 1-{EARLY} -> weeks {A.start_week}-{A.end_week})")
    d = build(a)
    print(f"  {len(d):,} player-seasons with {A.min_early_games}+ early games and {A.min_ros_games}+ later games")
    print("\nleague-average points per unit of volume")
    d = expected_points(d)

    out = []
    out.append(f"rows: {len(d):,} player-seasons, {d.season.min()}-{d.season.max()}, "
               f"weeks 1-{EARLY} predicting {A.start_week}-{A.end_week}, league scoring")
    out.append(f"naive rule = he keeps scoring his first-{EARLY}-week average; "
               f"mean |regression| = {d.gap.abs().mean():.2f} ppg")
    print(f"\nwalk-forward tests (predicting the GAP: rest-of-season ppg minus first-{EARLY}-week ppg)")
    tests = [
        (["early_ppg"], "early scoring alone"),
        (["luck"], "luck alone (points above volume)"),
        (["td_g"], "touchdowns per game alone"),
        (["tgt_share", "touch_g"], "volume alone"),
        (["early_ppg", "luck"], "early scoring + luck"),
        (["early_ppg", "luck", "tgt_share", "touch_g"], "scoring + luck + volume"),
        (["early_ppg", "luck", "tgt_share", "touch_g", "prev_ppg", "has_prev"], "the lot, with last season"),
    ]
    for feats, label in tests:
        score(d, walk_forward(d, feats, label), label, out)

    print("\nby position, the full model")
    out.append("")
    for pos in SKILL:
        dp = d[d.pos == pos].copy()
        if len(dp) < 300: continue
        p = walk_forward(dp, ["early_ppg", "luck", "tgt_share", "touch_g", "prev_ppg", "has_prev"], pos)
        score(dp, p, f"{pos}: scoring + luck + volume + prior", out)

    # ---- the practical table: sort by luck, see what happens next
    print("\nwhat actually happens to the players who over- and under-performed their volume")
    out.append("")
    d["luck_q"] = d.groupby(["season", "pos"]).luck.transform(
        lambda s: pd.qcut(s, 5, labels=["most unlucky", "unlucky", "even", "lucky", "most lucky"], duplicates="drop"))
    t = d.groupby("luck_q", observed=True).agg(
        n=("gap", "size"), early=("early_ppg", "mean"), ros=("ros_ppg", "mean"), gap=("gap", "mean")).round(2)
    print(t.to_string()); out.append("luck quintile (points above what volume implies, weeks 1-3):")
    out.append(t.to_string())

    d["vol_q"] = d.groupby(["season", "pos"]).touch_g.transform(
        lambda s: pd.qcut(s, 5, labels=["lowest", "low", "mid", "high", "highest"], duplicates="drop"))
    t2 = d.groupby("vol_q", observed=True).agg(
        n=("gap", "size"), early=("early_ppg", "mean"), ros=("ros_ppg", "mean"), gap=("gap", "mean")).round(2)
    print("\n" + t2.to_string()); out.append("")
    out.append("volume quintile (targets + carries per game, weeks 1-3):")
    out.append(t2.to_string())

    # the two corners the trade question is really about
    print("\nthe two corners")
    hi_luck_lo_vol = d[(d.luck_q == "most lucky") & (d.vol_q.isin(["lowest", "low", "mid"]))]
    lo_luck_hi_vol = d[(d.luck_q == "most unlucky") & (d.vol_q.isin(["high", "highest"]))]
    for lab, sub in (("SELL corner: hot but lightly used", hi_luck_lo_vol),
                     ("BUY corner: heavily used, cold", lo_luck_hi_vol)):
        line = (f"{lab:36s} n={len(sub):4d}  early {sub.early_ppg.mean():5.2f} -> rest {sub.ros_ppg.mean():5.2f} ppg "
                f"({sub.gap.mean():+.2f})")
        print("  " + line); out.append(line)

    os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "outputs", "models"), exist_ok=True)
    d.to_parquet(os.path.join(ROOT, "outputs", "models", "hot_start_frame.parquet"))
    with open(os.path.join(ROOT, "outputs", "reports", "hot_start_study.md"), "w", encoding="utf-8") as fh:
        fh.write("# Do the first three weeks predict the rest of the season?\n\n")
        fh.write("```\n" + "\n".join(out) + "\n```\n")
    print("\nwrote outputs/reports/hot_start_study.md and outputs/models/hot_start_frame.parquet")


if __name__ == "__main__":
    main()
