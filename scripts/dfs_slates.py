"""
Historical DraftKings main slates for optimizer backtests: salary + pregame projection + actual DK points.

One row per player on a Sunday main slate (games kicking off 13:00-16:59 ET; London 9:30 games and
TNF/SNF/MNF are not on it), 2016-25:
    season, week, game_id, player, pos, team, opp, salary, proj, actual, implied, spread
Sources: data/dk_salaries (salary + actual, RotoGuru 2016-21 / DFF 2022-25), the FFA weekly
consensus stat lines (data/ffanalytics/FFAn_weekly, the only pregame projection we hold for
every season), nflverse schedule + closing lines (data/sabersim/game_lines_2015_2026.parquet).

Projection = DraftKings points of the FFA stat line, bonuses as expectations (dk_scoring).
D/ST projection = FFA sacks/INT/safety/TD/blocks + a league-average fumble recovery + the
expected points-allowed tier with the opponent's implied total as the mean (sd 9.5, the spread
of NFL team scores around their implied totals). Players on the slate with no FFA line get
proj 0 (they are the punts nobody projects; the optimizer will not pick them).

    python scripts/dfs_slates.py        # -> outputs/dfs/slates.parquet
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import norm as N

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dk_scoring import projected_frame
from fill_dk_points import norm, initial_last, team_code, games, pa_points

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFA = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
OUT = os.path.join(ROOT, "outputs", "dfs")
FFA_TEAM = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "OAK", "SD": "SD", "STL": "STL"}
POS = {"Def": "DST", "DST": "DST", "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}
DST_FUMREC = 0.62                                   # league fumble recoveries per team-game (2016-25)
PA_SD = 9.5


def ffa_week(y, w):
    f = os.path.join(FFA, f"raw_stats_{y}_wk{w}.csv")
    if not os.path.exists(f): return None
    p = pd.read_csv(f)
    p = p[p.position.isin(["QB", "RB", "WR", "TE", "DST"])].copy()
    p["team"] = [team_code(FFA_TEAM.get(t, t), y) for t in p.team.fillna("FA")]
    p["proj"] = projected_frame(p) + 2 * p.two_pts.fillna(0) + 6 * p.return_tds.fillna(0)
    return p


def dst_proj(p, lines):
    d = p[p.position == "DST"].merge(lines[["team", "opp_implied"]], on="team", how="left")
    lam = d.opp_implied.fillna(22.0).values[:, None]
    grid = np.arange(0, 70)
    w = N.pdf(grid, lam, PA_SD); w /= w.sum(1, keepdims=True)
    e_pa = (w * pa_points(grid)).sum(1)
    z = lambda c: d[c].fillna(0)
    d["proj"] = z("dst_sacks") + 2 * (z("dst_int") + DST_FUMREC + z("dst_safety") + z("dst_blk")) + 6 * z("dst_td") + e_pa
    return d[["team", "proj"]]


def main():
    g = games()
    g = g[(g.game_type == "REG") & (g.weekday == "Sunday") & (g.gametime >= "13:00") & (g.gametime < "17:00")]
    main_games = pd.concat([g.assign(team=g.home_team, opp=g.away_team), g.assign(team=g.away_team, opp=g.home_team)])
    main_games = main_games[["season", "week", "game_id", "team", "opp"]]
    gl = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "game_lines_2015_2026.parquet"))
    gl = gl.rename(columns={"implied_team_total": "implied", "team_spread": "spread"})
    gl = gl.merge(gl[["season", "week", "team", "implied"]].rename(columns={"team": "opp", "implied": "opp_implied"}),
                  on=["season", "week", "opp"], how="left")
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "dk_salaries", "dk_salaries_*.csv"))):
        y = int(re.search(r"(\d{4})", f).group(1))
        if y < 2016 or y > 2025: continue
        s = pd.read_csv(f, dtype={"gid": str})
        s["pos"] = s.pos.map(POS); s = s[s.pos.notna()].copy()
        s["team"] = [team_code(t, y) for t in s.team]
        s = s.drop(columns=["opp"]).merge(main_games, on=["season", "week", "team"], how="inner")
        for w, sw in s.groupby("week"):
            p = ffa_week(y, w)
            if p is None: continue
            lines = gl[(gl.season == y) & (gl.week == w)]
            sk = sw[sw.pos != "DST"].copy()
            pp = p[p.position != "DST"]
            a = pp.assign(k=pp.player.map(norm)).drop_duplicates(["team", "k"], keep=False)[["team", "k", "proj"]]
            b = pp.assign(k=pp.player.map(initial_last)).drop_duplicates(["team", "k"], keep=False)[["team", "k", "proj"]]
            sk["k1"] = sk.player.map(norm); sk["k2"] = sk.player.map(initial_last)
            m1 = sk.merge(a, left_on=["team", "k1"], right_on=["team", "k"], how="left")["proj"].values
            m2 = sk.merge(b, left_on=["team", "k2"], right_on=["team", "k"], how="left")["proj"].values
            sk["proj"] = np.where(np.isnan(m1), m2, m1)
            ds = sw[sw.pos == "DST"].drop(columns=[c for c in ["proj"] if c in sw]).merge(dst_proj(p, lines), on="team", how="left")
            wk = pd.concat([sk, ds]).merge(lines[["team", "implied", "spread"]], on="team", how="left")
            out.append(wk)
    d = pd.concat(out, ignore_index=True)
    d = d.rename(columns={"dk_salary": "salary", "dk_points": "actual"})
    d["has_proj"] = d.proj.notna(); d["proj"] = d.proj.fillna(0.0)
    d = d[["season", "week", "game_id", "player", "pos", "team", "opp", "salary", "proj", "actual", "has_proj",
           "implied", "spread", "source"]]
    d = d[d.actual.notna() & (d.salary > 0)]
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(os.path.join(OUT, "slates.parquet"), index=False)
    s = d.groupby("season").agg(slates=("week", "nunique"), rows=("player", "size"), games=("game_id", "nunique"),
                                proj_cov=("has_proj", "mean"))
    print(s.to_string())
    pc = d[d.has_proj]
    print("\nproj vs actual by pos (rows with a projection): MAE, corr, bias")
    for pos, x in pc.groupby("pos"):
        print(f"  {pos:3s} n={len(x):6d}  MAE {np.mean(np.abs(x.proj - x.actual)):.2f}  r {np.corrcoef(x.proj, x.actual)[0,1]:.3f}"
              f"  bias {np.mean(x.actual - x.proj):+.2f}")
    print("\ncoverage of salary >= 4000 players by projection:",
          f"{d[d.salary >= 4000].has_proj.mean():.1%}")


if __name__ == "__main__":
    main()
