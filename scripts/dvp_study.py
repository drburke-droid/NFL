"""Is a flat points-per-game projection good enough, or does the opponent matter?

Two questions, both answered on 2016-2025 nflverse weekly data in the league's own scoring.

1. DOES EARLY DEFENCE-VERSUS-POSITION PERSIST? After three weeks you can rank every defence by the
   points it has allowed to each position. If that ranking is real it should predict what the same
   defence allows over the rest of the season. If it is noise it will not. The complication is that
   three weeks of "points allowed" is mostly a statement about which offences a defence happened to
   face, so this measures both the raw number and an opponent-adjusted one, where each performance
   is scored against what that offence's players do on average all year.

2. DOES IT IMPROVE A PLAYER PROJECTION? The practical test. Predict a player's actual points in a
   given week from his own rate alone, then from his rate plus his opponent's defence-versus-position
   figure, and see whether the second is closer. Walk-forward: the adjustment is fitted only on
   seasons before the one being scored. An earlier repo study found this adds nothing on top of the
   FFA consensus, which already prices matchup; the baseline here is deliberately weaker, a flat
   player rate, which is what scripts/hot_start_board.py uses.

Usage: python scripts/dvp_study.py [--early-weeks 3]
Writes outputs/reports/dvp_study.md
"""
import os, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
SKILL = ["QB", "RB", "WR", "TE"]
ap = argparse.ArgumentParser()
ap.add_argument("--early-weeks", type=int, default=3, help="weeks of defensive form the manager can see")
ap.add_argument("--end-week", type=int, default=17)
A = ap.parse_args()


def league_points(d):
    z = lambda c: d[c].fillna(0).astype(float) if c in d.columns else 0.0
    fl = z("fumbles_lost_total") if "fumbles_lost_total" in d.columns else \
        z("sack_fumbles_lost") + z("rushing_fumbles_lost") + z("receiving_fumbles_lost")
    return (0.04 * z("passing_yards") + 6 * z("passing_tds") - z("passing_interceptions")
            + 0.1 * z("rushing_yards") + 6 * z("rushing_tds")
            + z("receptions") + 0.1 * z("receiving_yards") + 6 * z("receiving_tds") - fl)


def load():
    out = []
    for s in range(2016, 2026):
        fp = os.path.join(CACHE, f"stats_player_week_{s}.parquet")
        if not os.path.exists(fp): continue
        d = pd.read_parquet(fp)
        d = d[(d.season_type == "REG") & d.position.isin(SKILL) & (d.week <= A.end_week)].copy()
        d["pts"] = league_points(d)
        out.append(d[["season", "week", "player_id", "player_display_name", "position", "team", "opponent_team", "pts"]])
    a = pd.concat(out, ignore_index=True)
    print(f"  {len(a):,} player-weeks, {a.season.min()}-{a.season.max()}")
    return a


def team_pos_week(a):
    """Points a defence allowed to each position each week, and what the attacking side usually scores."""
    g = a.groupby(["season", "week", "opponent_team", "position"], as_index=False).pts.sum()
    g = g.rename(columns={"opponent_team": "defence", "pts": "allowed"})
    off = a.groupby(["season", "week", "team", "position"], as_index=False).pts.sum().rename(
        columns={"team": "offence", "pts": "scored"})
    # what this offence's position group averages across the whole season: the strength it brought
    off_season = off.groupby(["season", "offence", "position"], as_index=False).scored.mean().rename(
        columns={"scored": "off_season_avg"})
    key = a.groupby(["season", "week", "opponent_team", "position"], as_index=False).team.first().rename(
        columns={"opponent_team": "defence", "team": "offence"})
    g = g.merge(key, on=["season", "week", "defence", "position"], how="left")
    g = g.merge(off_season, on=["season", "offence", "position"], how="left")
    lg = g.groupby(["season", "position"], as_index=False).allowed.mean().rename(columns={"allowed": "lg_avg"})
    g = g.merge(lg, on=["season", "position"], how="left")
    g["raw"] = g.allowed - g.lg_avg                       # points allowed above league average
    g["adj"] = g.allowed - g.off_season_avg               # ... above what that offence usually scores
    return g


def persistence(g, out):
    """Does the first few weeks of defence-versus-position predict the rest of the season?"""
    early = g[g.week <= A.early_weeks].groupby(["season", "defence", "position"], as_index=False).agg(
        raw_e=("raw", "mean"), adj_e=("adj", "mean"), n_e=("raw", "size"))
    late = g[g.week > A.early_weeks].groupby(["season", "defence", "position"], as_index=False).agg(
        raw_l=("raw", "mean"), adj_l=("adj", "mean"), n_l=("raw", "size"))
    m = early.merge(late, on=["season", "defence", "position"]).query("n_e >= 2 and n_l >= 6")
    out.append(f"persistence of defence-vs-position, first {A.early_weeks} weeks -> rest of season "
               f"(n={len(m)} defence-seasons-positions)")
    for pos in SKILL + ["ALL"]:
        d = m if pos == "ALL" else m[m.position == pos]
        if len(d) < 40: continue
        r_raw = np.corrcoef(d.raw_e, d.raw_l)[0, 1]
        r_adj = np.corrcoef(d.adj_e, d.adj_l)[0, 1]
        line = f"  {pos:4s} n={len(d):4d}   raw r={r_raw:+.3f}   opponent-adjusted r={r_adj:+.3f}"
        print(line); out.append(line)
    # how much of a real spread is there to exploit at all?
    out.append("")
    sd = g[g.week > A.early_weeks].groupby(["season", "defence", "position"]).raw.mean().groupby(
        level=[0, 2]).std().groupby(level=1).mean()
    out.append("spread of true rest-of-season defence-vs-position (sd of points allowed vs average):")
    for pos, v in sd.items():
        out.append(f"  {pos:4s} {v:5.2f} pts/game")
    print("\nspread across defences, rest of season (sd, pts/game):")
    print("  " + "  ".join(f"{p} {v:.2f}" for p, v in sd.items()))
    return m


def does_it_help(a, g, out):
    """The practical test: player's own rate, with and without an opponent adjustment."""
    # the manager's view after the early weeks: the player's own rate, and each defence's form
    early_pl = a[a.week <= A.early_weeks].groupby(["season", "player_id"], as_index=False).agg(
        pos=("position", "last"), rate=("pts", "mean"), n=("pts", "size"))
    early_pl = early_pl[early_pl.n >= 2]
    early_def = g[g.week <= A.early_weeks].groupby(["season", "defence", "position"], as_index=False).agg(
        raw_e=("raw", "mean"), adj_e=("adj", "mean"))
    later = a[a.week > A.early_weeks].merge(early_pl[["season", "player_id", "rate"]], on=["season", "player_id"])
    later = later.merge(early_def.rename(columns={"defence": "opponent_team", "position": "position"}),
                        on=["season", "opponent_team", "position"], how="left")
    later = later.dropna(subset=["rate", "raw_e", "adj_e"])
    out.append("")
    out.append(f"predicting a player's ACTUAL weekly points, weeks {A.early_weeks+1}-{A.end_week} "
               f"(n={len(later):,} player-weeks)")
    res = []
    for s in sorted(later.season.unique()):
        tr, te = later[later.season < s], later[later.season == s]
        if len(tr) < 5000 or not len(te): continue
        base_tr = tr.rate.values
        for lab, col in (("raw", "raw_e"), ("opponent-adjusted", "adj_e")):
            X = np.column_stack([base_tr, tr[col].values, np.ones(len(tr))])
            beta, *_ = np.linalg.lstsq(X, tr.pts.values, rcond=None)
            Xte = np.column_stack([te.rate.values, te[col].values, np.ones(len(te))])
            res.append(dict(season=s, kind=lab, mae=np.abs(te.pts.values - Xte @ beta).mean(),
                            weight=beta[1], n=len(te)))
        # baseline: the player's rate alone, refitted the same way so the comparison is fair
        Xb = np.column_stack([base_tr, np.ones(len(tr))])
        bb, *_ = np.linalg.lstsq(Xb, tr.pts.values, rcond=None)
        res.append(dict(season=s, kind="player rate alone", mae=np.abs(
            te.pts.values - np.column_stack([te.rate.values, np.ones(len(te))]) @ bb).mean(),
            weight=np.nan, n=len(te)))
    R = pd.DataFrame(res)
    piv = R.pivot_table(index="kind", values="mae", aggfunc="mean")
    base = piv.loc["player rate alone", "mae"]
    for kind in ("player rate alone", "raw", "opponent-adjusted"):
        if kind not in piv.index: continue
        v = piv.loc[kind, "mae"]
        w = R[R.kind == kind].weight.mean()
        line = (f"  {kind:22s} MAE {v:6.4f}" + ("" if kind == "player rate alone" else
                f"   vs baseline {(1 - v/base):+.3%}   avg weight on the matchup term {w:+.3f}"))
        print(line); out.append(line)
    return R


def main():
    print("loading")
    a = load()
    print(f"\nbuilding defence-vs-position (first {A.early_weeks} weeks as the signal)")
    g = team_pos_week(a)
    print(f"  {len(g):,} defence-week-position rows")
    out = []
    print("\n--- 1. does early defence-vs-position persist? ---")
    persistence(g, out)
    print("\n--- 2. does it improve a player projection? ---")
    does_it_help(a, g, out)
    os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "reports", "dvp_study.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Does the opponent matter, and can you see it after {A.early_weeks} weeks?\n\n```\n"
                 + "\n".join(out) + "\n```\n")
    print("\nwrote outputs/reports/dvp_study.md")


if __name__ == "__main__":
    main()
