"""How much does a player's ACTUAL scoring move when he faces a soft or hard defence?

No projections anywhere. Everything is box-score actuals in the league's own scoring:
  - a player's year-to-date points per game, from the weeks before the game in question
  - the defence he is about to face, ranked by the points it has allowed that position
  - what he then actually scored

The answer is the mean of (actual points - his own year-to-date rate), bucketed by how soft the
defence was. If matchup matters that difference should climb steadily from the toughest bucket to the
softest, and the size of the climb is the most you could ever gain by streaming for matchup.

Two views of the defence, because they answer different questions:
  KNOWABLE  ranked on what the defence had allowed BEFORE this game, which is all a manager has
  ORACLE    ranked on the defence's whole-season figure, which nobody can know in advance and is
            therefore the ceiling on any matchup strategy, however good the measurement gets

Usage: python scripts/dvp_buckets.py [--min-ytd-games 3]
Writes outputs/reports/dvp_buckets.md
"""
import os, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
SKILL = ["QB", "RB", "WR", "TE"]
ap = argparse.ArgumentParser()
ap.add_argument("--min-ytd-games", type=int, default=3, help="games of year-to-date rate needed before a game counts")
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
    a = pd.concat(out, ignore_index=True).sort_values(["season", "player_id", "week"])
    print(f"  {len(a):,} player-weeks, {a.season.min()}-{a.season.max()}")
    return a


def main():
    print("loading actuals")
    a = load()

    # ---- a player's year-to-date rate, using only the weeks BEFORE the game in question
    grp = a.groupby(["season", "player_id"])
    a["ytd_pts"] = grp.pts.transform(lambda s: s.shift().expanding().sum())
    a["ytd_g"] = grp.pts.transform(lambda s: s.shift().expanding().count())
    a["ytd_ppg"] = a.ytd_pts / a.ytd_g

    # ---- the defence, as points allowed to that position
    dw = a.groupby(["season", "week", "opponent_team", "position"], as_index=False).pts.sum().rename(
        columns={"opponent_team": "defence", "pts": "allowed"})
    lg = dw.groupby(["season", "position"], as_index=False).allowed.mean().rename(columns={"allowed": "lg"})
    dw = dw.merge(lg, on=["season", "position"]).sort_values(["season", "defence", "position", "week"])
    dw["dev"] = dw.allowed - dw.lg
    dg = dw.groupby(["season", "defence", "position"])
    # KNOWABLE: what this defence had allowed before this week
    dw["known"] = dg.dev.transform(lambda s: s.shift().expanding().mean())
    dw["known_n"] = dg.dev.transform(lambda s: s.shift().expanding().count())
    # ORACLE: the whole season, including this game
    dw["oracle"] = dg.dev.transform("mean")

    m = a.merge(dw[["season", "week", "defence", "position", "known", "known_n", "oracle"]],
                left_on=["season", "week", "opponent_team", "position"],
                right_on=["season", "week", "defence", "position"], how="left")
    m = m[(m.ytd_g >= A.min_ytd_games) & m.ytd_ppg.notna()]
    m["gap"] = m.pts - m.ytd_ppg          # how far he beat or missed his own year-to-date rate
    print(f"  {len(m):,} games with {A.min_ytd_games}+ games of year-to-date rate behind them")

    out = [f"{len(m):,} player-games, 2016-2025, league scoring. No projections used: the baseline is the "
           f"player's own year-to-date points per game from the weeks BEFORE each game.",
           "'gap' = his actual points that week minus his own year-to-date rate.", ""]

    for col, lab, note in (("known", "KNOWABLE", "defence ranked on what it had allowed before this game"),
                           ("oracle", "ORACLE", "defence ranked on its whole season: unknowable, the ceiling")):
        d = m[m[col].notna() & (m.known_n >= 2)] if col == "known" else m[m[col].notna()]
        d = d.copy()
        d["bucket"] = d.groupby(["season", "position"])[col].transform(
            lambda s: pd.qcut(s, 5, labels=["toughest", "hard", "average", "soft", "softest"], duplicates="drop"))
        t = d.groupby("bucket", observed=True).agg(
            games=("gap", "size"), ytd_rate=("ytd_ppg", "mean"), actual=("pts", "mean"), gap=("gap", "mean")).round(2)
        spread = t.gap.iloc[-1] - t.gap.iloc[0]
        print(f"\n--- {lab}: {note}")
        print(t.to_string())
        print(f"    softest minus toughest: {spread:+.2f} pts per game")
        out += [f"{lab} ({note}):", t.to_string(), f"  softest minus toughest: {spread:+.2f} pts/game", ""]

        # by position, since the earlier study showed receivers have the widest defensive spread
        print(f"    by position (softest minus toughest):")
        line = []
        for pos in SKILL:
            dp = d[d.position == pos]
            if len(dp) < 500: continue
            tp = dp.groupby("bucket", observed=True).gap.mean()
            if len(tp) < 5: continue
            line.append(f"{pos} {tp.iloc[-1] - tp.iloc[0]:+.2f}")
        print("      " + "   ".join(line)); out.append("  by position: " + "   ".join(line)); out.append("")

    # what a perfect streamer would gain: always play the softest bucket instead of an average one
    d = m[m.oracle.notna()].copy()
    d["bucket"] = d.groupby(["season", "position"]).oracle.transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
    best = d[d.bucket == 5].gap.mean(); mid = d[d.bucket == 3].gap.mean()
    line = (f"a manager who could always pick the softest-fifth matchup instead of an average one would gain "
            f"{best - mid:+.2f} pts per game per player, with perfect foreknowledge")
    print("\n" + line); out.append(line)
    line2 = ("for scale: the typical week-to-week swing in a player's own scoring is about "
             f"{m.gap.abs().mean():.2f} pts, and a bye week costs the whole starter")
    print(line2); out.append(line2)

    os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "reports", "dvp_buckets.md"), "w", encoding="utf-8") as fh:
        fh.write("# How much do actual points move with the defence faced?\n\n```\n" + "\n".join(out) + "\n```\n")
    print("\nwrote outputs/reports/dvp_buckets.md")


if __name__ == "__main__":
    main()
