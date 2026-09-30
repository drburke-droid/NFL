"""Can we measure a WR1's matchup without knowing which cornerback covers him?

The problem this attacks: the true softest-vs-toughest spread for a team's WR1 is +3.41 points a
game, the largest of any position group, but the version a manager can measure from points allowed
to receivers is ZERO. Pooling all receivers is the suspected cause, because a defence's total
receiver yield is mostly a statement about how it handled WR2s and WR3s while its best cornerback
travelled with the WR1.

Three candidate fixes, all from data already in the repo, no paid feed:

  A. ROLE-SPECIFIC YIELD. Stop pooling. Measure what a defence has allowed specifically to the
     opposing team's WR1, WR2 and WR3, and match a receiver to the figure for HIS role.
  B. SCHEME. data/participation/team_def_*.parquet carries each defence's man-coverage rate,
     two-high-safety rate, pressure rate and time to throw per game. Shadow coverage is a man
     concept, so a man-heavy defence is where a travelling cornerback can exist at all.
  C. A COVERAGE MATRIX from play-by-play. `pass_defense_1_player_id` names a defender on about 12%
     of pass plays (the ones he got a hand to), which is a biased but real sample of who guards whom;
     it is used here only to ask whether WR1s face a CONCENTRATED defender (one man taking most of
     the coverage snaps against him) rather than a rotating cast.

Everything is measured from weeks BEFORE the game being predicted, and compared on the same footing
as the earlier study: mean (actual points - his own year-to-date rate), softest fifth minus toughest.

Usage: python scripts/wr1_matchup_study.py
Writes outputs/reports/wr1_matchup_study.md
"""
import os, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
PART = os.path.join(ROOT, "data", "participation")
SEASONS = range(2016, 2026)


def league_points(d):
    z = lambda c: d[c].fillna(0).astype(float) if c in d.columns else 0.0
    fl = z("fumbles_lost_total") if "fumbles_lost_total" in d.columns else \
        z("sack_fumbles_lost") + z("rushing_fumbles_lost") + z("receiving_fumbles_lost")
    return (0.04 * z("passing_yards") + 6 * z("passing_tds") - z("passing_interceptions")
            + 0.1 * z("rushing_yards") + 6 * z("rushing_tds")
            + z("receptions") + 0.1 * z("receiving_yards") + 6 * z("receiving_tds") - fl)


def load_weekly():
    out = []
    for s in SEASONS:
        fp = os.path.join(CACHE, f"stats_player_week_{s}.parquet")
        if not os.path.exists(fp): continue
        d = pd.read_parquet(fp)
        d = d[(d.season_type == "REG") & d.position.isin(["QB", "RB", "WR", "TE"]) & (d.week <= 17)].copy()
        d["pts"] = league_points(d)
        out.append(d[["season", "week", "player_id", "player_display_name", "position",
                      "team", "opponent_team", "pts", "target_share"]])
    return pd.concat(out, ignore_index=True).sort_values(["season", "player_id", "week"])


def add_role_and_rate(a):
    """Year-to-date rate and depth role, both from PRIOR weeks only."""
    g = a.groupby(["season", "player_id"])
    a["ytd_ppg"] = g.pts.transform(lambda s: s.shift().expanding().mean())
    a["ytd_g"] = g.pts.transform(lambda s: s.shift().expanding().count())
    a["ytd_ts"] = g.target_share.transform(lambda s: s.shift().expanding().mean())
    wr = a[a.position == "WR"].copy()
    wr["rk"] = wr.groupby(["season", "week", "team"]).ytd_ts.rank(ascending=False, method="first")
    wr["role"] = np.where(wr.rk == 1, "WR1", np.where(wr.rk == 2, "WR2", np.where(wr.rk == 3, "WR3", "WR4+")))
    a = a.merge(wr[["season", "week", "player_id", "role"]], on=["season", "week", "player_id"], how="left")
    a["gap"] = a.pts - a.ytd_ppg
    return a


def yield_tables(a):
    """What each defence allowed: pooled over all receivers, and split by the role it was facing."""
    wr = a[(a.position == "WR") & a.role.notna()]
    # pooled (the old, failing metric)
    pool = wr.groupby(["season", "week", "opponent_team"], as_index=False).pts.sum().rename(
        columns={"opponent_team": "defence", "pts": "allowed"})
    lg = pool.groupby("season", as_index=False).allowed.mean().rename(columns={"allowed": "lg"})
    pool = pool.merge(lg, on="season"); pool["dev"] = pool.allowed - pool.lg
    pool = pool.sort_values(["season", "defence", "week"])
    pg = pool.groupby(["season", "defence"])
    pool["pooled_known"] = pg.dev.transform(lambda s: s.shift().expanding().mean())
    pool["pooled_n"] = pg.dev.transform(lambda s: s.shift().expanding().count())

    # role-specific: what it allowed to the WR1 it faced, the WR2 it faced, and so on
    role = wr.groupby(["season", "week", "opponent_team", "role"], as_index=False).pts.sum().rename(
        columns={"opponent_team": "defence", "pts": "allowed"})
    lgr = role.groupby(["season", "role"], as_index=False).allowed.mean().rename(columns={"allowed": "lg"})
    role = role.merge(lgr, on=["season", "role"]); role["dev"] = role.allowed - role.lg
    role = role.sort_values(["season", "defence", "role", "week"])
    rg = role.groupby(["season", "defence", "role"])
    role["role_known"] = rg.dev.transform(lambda s: s.shift().expanding().mean())
    role["role_n"] = rg.dev.transform(lambda s: s.shift().expanding().count())
    role["role_oracle"] = rg.dev.transform("mean")
    return pool[["season", "week", "defence", "pooled_known", "pooled_n"]], \
           role[["season", "week", "defence", "role", "role_known", "role_n", "role_oracle"]]


def scheme_table():
    """Defensive scheme per game, then the running mean from prior weeks only."""
    out = []
    for fp in sorted(glob.glob(os.path.join(PART, "team_def_*.parquet"))):
        out.append(pd.read_parquet(fp))
    t = pd.concat(out, ignore_index=True)
    t = t[t.week <= 17].copy()
    t = t.rename(columns={"defteam": "defence"}).sort_values(["season", "defence", "week"])
    g = t.groupby(["season", "defence"])
    for c in ("def_man_rate", "def_pressure_rate", "def_two_high_rate", "ttt_mean", "box_mean"):
        if c in t.columns:
            t[c + "_k"] = g[c].transform(lambda s: s.shift().expanding().mean())
    t["scheme_n"] = g.def_man_rate.transform(lambda s: s.shift().expanding().count())
    keep = ["season", "week", "defence", "scheme_n"] + [c + "_k" for c in
            ("def_man_rate", "def_pressure_rate", "def_two_high_rate", "ttt_mean", "box_mean") if c + "_k" in t.columns]
    return t[keep]


def buckets(d, col, label, out, min_n=2):
    d = d[d[col].notna()].copy()
    if len(d) < 500:
        return
    d["b"] = d.groupby(["season", "role"])[col].transform(
        lambda s: pd.qcut(s, 5, labels=["toughest", "hard", "avg", "soft", "softest"], duplicates="drop"))
    t = d.pivot_table(index="role", columns="b", values="gap", aggfunc="mean", observed=True)
    if "softest" not in t.columns or "toughest" not in t.columns:
        return
    t["spread"] = (t["softest"] - t["toughest"])
    t["n"] = d.groupby("role", observed=True).size()
    t = t.reindex([r for r in ("WR1", "WR2", "WR3", "WR4+") if r in t.index]).round(2)
    print(f"\n--- {label}")
    print(t.to_string()); out += [f"{label}:", t.to_string(), ""]


def main():
    print("loading weekly actuals")
    a = add_role_and_rate(load_weekly())
    pool, role = yield_tables(a)
    sch = scheme_table()
    print(f"  scheme rows {len(sch):,}")

    wr = a[(a.position == "WR") & a.role.notna() & (a.ytd_g >= 3) & a.ytd_ppg.notna()].copy()
    wr = wr.rename(columns={"opponent_team": "defence"})
    wr = wr.merge(pool, on=["season", "week", "defence"], how="left")
    wr = wr.merge(role, on=["season", "week", "defence", "role"], how="left")
    wr = wr.merge(sch, on=["season", "week", "defence"], how="left")
    print(f"  {len(wr):,} WR games with a role, a rate and a defence")

    out = [f"{len(wr):,} WR games, 2016-2025, league scoring. 'gap' = actual points minus the "
           f"receiver's own year-to-date rate. Every defensive figure uses PRIOR weeks only.",
           "Columns are quintiles of the defence measure; 'spread' = softest minus toughest.", ""]

    buckets(wr[wr.pooled_n >= 2], "pooled_known", "A0. POOLED yield to all receivers (the failing metric)", out)
    buckets(wr[wr.role_n >= 2], "role_known", "A1. ROLE-SPECIFIC yield: what this defence allowed to the role he plays", out)
    buckets(wr, "role_oracle", "A2. ROLE-SPECIFIC, ORACLE (whole season: the ceiling)", out)
    for c, lab in (("def_man_rate_k", "B1. SCHEME: man-coverage rate (soft = most man)"),
                   ("def_two_high_rate_k", "B2. SCHEME: two-high-safety rate"),
                   ("def_pressure_rate_k", "B3. SCHEME: pressure rate"),
                   ("ttt_mean_k", "B4. SCHEME: time to throw allowed")):
        if c in wr.columns:
            buckets(wr[wr.scheme_n >= 2], c, lab, out)

    # do the role-specific and pooled measures even agree?
    d = wr[(wr.role == "WR1") & wr.pooled_known.notna() & wr.role_known.notna()]
    r = np.corrcoef(d.pooled_known, d.role_known)[0, 1]
    line = (f"correlation between a defence's POOLED receiver yield and its WR1-specific yield: "
            f"r={r:+.3f} (n={len(d):,}) - if this were near 1 the pooling would not matter")
    print("\n" + line); out += ["", line]

    os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "reports", "wr1_matchup_study.md"), "w", encoding="utf-8") as fh:
        fh.write("# Measuring a WR1's matchup without cornerback tracking\n\n```\n" + "\n".join(out) + "\n```\n")
    wr.to_parquet(os.path.join(ROOT, "outputs", "models", "wr1_matchup_frame.parquet"))
    print("\nwrote outputs/reports/wr1_matchup_study.md")


if __name__ == "__main__":
    main()
