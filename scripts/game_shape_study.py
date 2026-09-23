"""How are a game's fantasy points shaped, and how much of that shape is knowable beforehand?

The question this answers: can games be sorted into types by the SHAPE of the fantasy points
their players accumulate -- a shootout paying everyone, a blowout paying the winner's back and
starving the loser's -- and if so, is the type predictable from the pregame market?

Measured on 2023-25 regular season (816 games, 1632 team-games), the answer has three parts:

1. There are NO discrete game types. K-means on the shape features peaks at silhouette 0.198
   at k=2 and decays (k=5: 0.157). The space is a continuum. Reporting five tidy clusters here
   would be naming arbitrary cuts through one cloud, so this script prints the silhouette scan
   and then works in continuous axes instead.

2. Four axes carry 84% of the variance, and two of them are the intuitive game types --
   recovered from the data without being told to look for them:
     PC1 (31%)  scoring environment       r=+0.46 with the combined game total
     PC2 (21%)  trailing-and-throwing     r=-0.50 with margin, -0.58 with carries
     PC3 (19%)  TE-centric offense        team identity, not game script (see 3)
     PC4 (12%)  QB-centric                weak context correlation
   Context features (margin, spread, totals) are deliberately EXCLUDED from the clustering and
   the PCA, so their correlation with the axes is a finding rather than an artefact.

3. Level is a property of the game; composition is a property of the team. Game context explains
   64% of how many fantasy points a team produces and 2-3% of how they are divided among its
   positions. Team-season identity is the reverse. That is why "game type" is the wrong frame for
   the position mix: the TE axis is which offense this is, not which kind of game.

The DFS-relevant consequence is a negative one, and the script prints it plainly: the pregame
spread and total explain 17% of a team's fantasy total and about 1% of its composition. You
cannot forecast a fingerprint better than you can forecast the game, and the market itself only
manages r=0.31 on the total and r=0.49 on the margin. The one clean monotone gradient worth
acting on is the favourite's running back: 17.8 points as an 8+ dog to 26.6 as an 8+ favourite,
rising in every bucket -- and even that is a level effect, since rb_sh moves only 0.266 -> 0.278.

Data: nflverse stats_player_week_{season} and the schedules release, downloaded and cached under
data/nflverse/. Needs no database and no API key, so it runs in the cloud console as well as on
a PC. Scoring is fantasy_points_ppr, the same target sabersim_grade.py uses, so the numbers here
are comparable to the accuracy reports.

With --db it also cross-tabs the axes against the 8 labels in the `game_scripts` table that
cluster_game_scripts.py writes (root of the repo, db/nfl_odds.db, seasons 2012-25). That table
is built from quarter-by-quarter scoring trajectories rather than fantasy points, so agreement
is a check that two independent routes find the same structure, not a tautology.

Usage:
    python scripts/game_shape_study.py                        # 2023-25, nflverse only
    python scripts/game_shape_study.py --seasons 2015-2025    # more history
    python scripts/game_shape_study.py --db                   # + cross-tab vs game_scripts
    python scripts/game_shape_study.py --out outputs/reports/game_shape_study.md
"""
import os, sys, argparse, sqlite3
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse")
BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
SKILL = ["QB", "RB", "WR", "TE"]
# shape features only -- no margin, no spread, no totals. See note 2 in the header.
FEATS = ["tot", "qb_sh", "rb_sh", "wr_sh", "te_sh", "top1", "n10", "pass_share"]


def seasons_arg(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(s)]


def cached(url, name):
    """nflverse parquets are small (<1 MB) but the release endpoint is slow; cache them."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        pd.read_parquet(url).to_parquet(p)
    return pd.read_parquet(p)


def load(seasons):
    sch = cached(BASE + "schedules/games.parquet", "games.parquet")
    frames = []
    for y in seasons:
        try:
            frames.append(cached(BASE + f"stats_player/stats_player_week_{y}.parquet", f"pw_{y}.parquet"))
        except Exception as e:
            print(f"  {y}: unavailable ({str(e)[:60]})", file=sys.stderr)
    if not frames:
        raise SystemExit("no player-week data could be loaded")
    return pd.concat(frames, ignore_index=True), sch


def team_game_shapes(pw, sch, seasons):
    """One row per (game, team): how many fantasy points, and how they were distributed.

    Pure function of its inputs so it can be tested on a synthetic frame.
    """
    p = pw[(pw.season_type == "REG") & pw.position.isin(SKILL) & pw.season.isin(seasons)].copy()
    p["ppr"] = p.fantasy_points_ppr.fillna(0.0)
    # Only players who actually appeared. nflverse carries no row for an inactive, but it does
    # carry rows with all-zero lines; keeping those would pad every team with free zeros and
    # drag the concentration measures toward zero.
    touched = p[["completions", "attempts", "carries", "targets", "receptions"]].fillna(0).sum(axis=1) > 0
    p = p[touched | (p.ppr != 0)]

    rows = []
    for (gid, tm), d in p.groupby(["game_id", "team"], sort=False):
        tot = d.ppr.sum()
        if tot <= 0:
            continue
        pos = d.groupby("position").ppr.sum()
        # shares off clipped points: a negative line (INTs, fumbles) must not make a share > 1
        cl = d.ppr.clip(lower=0)
        sh = cl / max(cl.sum(), 1e-9)
        qbs = d[d.position == "QB"].sort_values("ppr", ascending=False)
        rbs = d[d.position == "RB"].sort_values("ppr", ascending=False)
        wrs = d[d.position == "WR"].sort_values("ppr", ascending=False)
        rows.append(dict(
            game_id=gid, team=tm, tot=tot,
            qb=pos.get("QB", 0.0), rb=pos.get("RB", 0.0), wr=pos.get("WR", 0.0), te=pos.get("TE", 0.0),
            hhi=float((sh ** 2).sum()), top1=float(sh.max()),
            n10=int((d.ppr >= 10).sum()), n20=int((d.ppr >= 20).sum()),
            qb2=float(qbs.ppr.iloc[1]) if len(qbs) > 1 else 0.0,
            rb1=float(rbs.ppr.iloc[0]) if len(rbs) else 0.0,
            wr1=float(wrs.ppr.iloc[0]) if len(wrs) else 0.0,
            att=float(d.attempts.fillna(0).sum()), car=float(d.carries.fillna(0).sum())))
    tg = pd.DataFrame(rows)

    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line", "roof", "div_game"]
    tg = tg.merge(sch[cols], on="game_id", how="left")
    tg = tg.dropna(subset=["home_score", "away_score"])
    tg["is_home"] = tg.team == tg.home_team
    tg["pts_for"] = np.where(tg.is_home, tg.home_score, tg.away_score)
    tg["margin"] = tg.pts_for - np.where(tg.is_home, tg.away_score, tg.home_score)
    tg["game_total"] = tg.home_score + tg.away_score
    # spread_line is the HOME team's line; flip it so it is always this team's
    tg["spread"] = np.where(tg.is_home, tg.spread_line, -tg.spread_line)
    for c in ("qb", "rb", "wr", "te"):
        tg[c + "_sh"] = tg[c] / tg.tot
    tg["pass_share"] = tg.att / (tg.att + tg.car).clip(lower=1)
    return tg


def md_table(df, index_name=""):
    """DataFrame -> markdown rows. Hand-rolled because pandas needs the `tabulate` package for
    this, which is not a dependency of the repo and is not worth adding for three tables."""
    head = [index_name] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [f"{v}" for v in row.values]) + " |")
    return "\n".join(out)


def r2(tg, xs, y):
    d = tg.dropna(subset=xs + [y])
    if len(d) < 30:
        return float("nan")
    X, yy = d[xs].values, d[y].values
    return LinearRegression().fit(X, yy).score(X, yy)


def report(tg, db=False):
    L = []
    def w(s=""): L.append(s); print(s)

    w(f"# Fantasy-point shape of an NFL game — {tg.game_id.nunique()} games, "
      f"{len(tg)} team-games, seasons {tg.season.min()}-{tg.season.max()}")
    w()
    w("Scoring is PPR (fantasy_points_ppr). Shape features only; margin/spread/total are held out.")
    w()

    Z = StandardScaler().fit_transform(tg[FEATS].values)

    w("## Are there discrete game types? No.")
    w()
    w("| k | silhouette | cluster sizes |")
    w("|---|---|---|")
    for k in range(2, 9):
        km = KMeans(k, n_init=25, random_state=0).fit(Z)
        w(f"| {k} | {silhouette_score(Z, km.labels_):.3f} | "
          f"{', '.join(str(x) for x in sorted(np.bincount(km.labels_), reverse=True))} |")
    w()
    w("A silhouette this low means one continuous cloud, not separable groups. Use the axes below.")
    w()

    pca = PCA().fit(Z); P = pca.transform(Z)
    for i in range(4):
        tg[f"PC{i+1}"] = P[:, i]
    w("## The axes that actually exist")
    w()
    w(f"Variance explained: {', '.join(f'{v:.0%}' for v in pca.explained_variance_ratio_[:4])} "
      f"(first four: {pca.explained_variance_ratio_[:4].sum():.0%})")
    w()
    ld = pd.DataFrame(pca.components_[:4].T, index=FEATS, columns=[f"PC{i}" for i in range(1, 5)])
    w(md_table(ld.round(2), "feature"))
    w()
    ctx = ["margin", "game_total", "pts_for", "spread", "total_line", "att", "car"]
    cc = tg[[f"PC{i}" for i in range(1, 5)] + ctx].corr().loc[[f"PC{i}" for i in range(1, 5)], ctx]
    w("Correlation with game context (held out of the fit):")
    w()
    w(md_table(cc.round(3), "axis"))
    w()

    tg["side"] = np.where(tg.margin > 0, "W", np.where(tg.margin < 0, "L", "T"))
    tg["mb"] = pd.cut(tg.margin.abs(), [-1, 3, 10, 17, 200], labels=["0-3", "4-10", "11-17", "18+"])
    w("## Game-script asymmetry: winner vs loser by final margin")
    w()
    w("| margin | side | n | team tot | QB | RB | WR | TE | carries | att | QB2 | RB1 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mb in ["0-3", "4-10", "11-17", "18+"]:
        for sd in ["W", "L"]:
            d = tg[(tg.mb == mb) & (tg.side == sd)]
            if not len(d):
                continue
            w(f"| {mb} | {sd} | {len(d)} | {d.tot.mean():.1f} | {d.qb.mean():.1f} | {d.rb.mean():.1f} | "
              f"{d.wr.mean():.1f} | {d.te.mean():.1f} | {d.car.mean():.1f} | {d.att.mean():.1f} | "
              f"{d.qb2.mean():.2f} | {d.rb1.mean():.1f} |")
    w()
    w("The losing side throws MORE and its receivers score LESS -- extra attempts arrive with "
      "sacks, incompletions and picks attached. In share terms the loser's WRs do gain "
      "(blowouts: wr_sh "
      f"{tg[(tg.mb=='18+')&(tg.side=='L')].wr_sh.mean():.3f} vs winner "
      f"{tg[(tg.mb=='18+')&(tg.side=='W')].wr_sh.mean():.3f}), but within a much smaller pie.")
    w()

    tb = pd.cut(tg.game_total, [-1, 34, 44, 54, 500], labels=["<=34", "35-44", "45-54", "55+"])
    w("## Scoring environment: by combined game total")
    w()
    w("| total | n | team tot | QB | RB | WR | TE | n>=10 | n>=20 | top1_sh | hhi |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ["<=34", "35-44", "45-54", "55+"]:
        d = tg[tb == t]
        if not len(d):
            continue
        w(f"| {t} | {len(d)} | {d.tot.mean():.1f} | {d.qb.mean():.1f} | {d.rb.mean():.1f} | "
          f"{d.wr.mean():.1f} | {d.te.mean():.1f} | {d.n10.mean():.2f} | {d.n20.mean():.2f} | "
          f"{d.top1.mean():.3f} | {d.hhi.mean():.3f} |")
    w()
    w("Concentration is invariant: top1_sh and hhi barely move across a pie that grows ~60%. "
      "A shootout scales the distribution rather than reshaping it -- what rises is breadth at a "
      "fixed threshold (n>=10, n>=20). QB share climbs and RB share falls; WR share is flat.")
    w()

    w("## Level is the game's; composition is the team's")
    w()
    ident = pd.get_dummies(tg.team.astype(str) + "_" + tg.season.astype(str)).astype(float)
    G = tg[["margin", "game_total"]].values
    w("| target | team-season identity | game context | both |")
    w("|---|---|---|---|")
    for t in ["tot", "qb_sh", "rb_sh", "wr_sh", "te_sh", "top1", "pass_share"]:
        y = tg[t].values
        a = LinearRegression().fit(ident.values, y).score(ident.values, y)
        b = LinearRegression().fit(G, y).score(G, y)
        c = np.hstack([ident.values, G])
        w(f"| {t} | {a:.3f} | {b:.3f} | {LinearRegression().fit(c, y).score(c, y):.3f} |")
    w()
    w(f"Identity R2 is in-sample with {ident.shape[1]} dummies on {len(tg)} rows, so inflated by "
      f"roughly {ident.shape[1]/len(tg):.3f}; the ratio against game context still holds.")
    w()

    w("## How much is knowable before kickoff")
    w()
    w("| target | pregame (spread, total_line) | realized (margin, game total) |")
    w("|---|---|---|")
    for t in ["tot", "qb", "rb", "wr", "te", "rb_sh", "wr_sh", "qb_sh", "PC1", "PC2"]:
        w(f"| {t} | {r2(tg, ['spread','total_line'], t):.3f} | {r2(tg, ['margin','game_total'], t):.3f} |")
    w()
    for x, y in [("total_line", "game_total"), ("spread", "margin")]:
        c = tg[[x, y]].corr().iloc[0, 1]
        w(f"- market {x} vs {y}: r={c:.3f} (R2={c**2:.3f})")
    w()
    w("The market is the ceiling: a shape cannot be forecast better than the game itself, and "
      "composition is near-unforecastable from anything -- ~1% pregame, ~3% with hindsight.")
    w()

    w("## The one monotone gradient: the favourite's running back")
    w()
    sb = pd.cut(tg.spread, [-99, -7.5, -3.5, -0.5, 0.5, 3.5, 7.5, 99],
                labels=["dog 8+", "dog 4-7", "dog 1-3", "pk", "fav 1-3", "fav 4-7", "fav 8+"])
    w("| pregame spread | n | RB pts | RB1 | carries | rb_sh | team tot | win% |")
    w("|---|---|---|---|---|---|---|---|")
    for lab in ["dog 8+", "dog 4-7", "dog 1-3", "pk", "fav 1-3", "fav 4-7", "fav 8+"]:
        d = tg[sb == lab]
        if len(d) < 10:
            continue
        w(f"| {lab} | {len(d)} | {d.rb.mean():.1f} | {d.rb1.mean():.1f} | {d.car.mean():.1f} | "
          f"{d.rb_sh.mean():.3f} | {d.tot.mean():.1f} | {(d.margin>0).mean()*100:.1f} |")
    w()
    w("Monotone in every step, but note rb_sh barely moves: this is a level effect, not a mix "
      "effect. Check whether the model already prices it before treating it as an edge.")
    w()

    if db:
        w("## Cross-tab against the quarter-trajectory labels (`game_scripts`)")
        w()
        p = os.path.join(ROOT, "db", "nfl_odds.db")
        if not os.path.exists(p):
            w(f"`{os.path.relpath(p, ROOT)}` not present — skipped. Run on the PC that has the DB.")
        else:
            try:
                con = sqlite3.connect(p)
                gs = pd.read_sql("SELECT game_id, game_script FROM game_scripts", con); con.close()
                j = tg.merge(gs, on="game_id", how="inner")
                if j.empty:
                    w("No game_id overlap — cluster_game_scripts.py may key on event_id instead; "
                      "join on (season, week, home_team) there.")
                else:
                    w(f"{j.game_id.nunique()} games matched.")
                    w()
                    g = j.groupby("game_script").agg(n=("tot", "size"), tot=("tot", "mean"),
                        PC1=("PC1", "mean"), PC2=("PC2", "mean"), PC3=("PC3", "mean"),
                        margin=("margin", "mean"), game_total=("game_total", "mean")).round(2)
                    w(md_table(g, "game_script"))
                    w()
                    w("If PC1 separates the high-total labels and PC2 the blowout labels, two "
                      "independent routes (quarter trajectories, fantasy shape) found the same "
                      "structure. Flat columns mean they are measuring different things.")
            except Exception as e:
                w(f"DB read failed: {str(e)[:120]}")
        w()
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023-2025", help="e.g. 2023-2025 or 2019")
    ap.add_argument("--db", action="store_true", help="cross-tab against db/nfl_odds.db game_scripts")
    ap.add_argument("--out", default=os.path.join("outputs", "reports", "game_shape_study.md"))
    ap.add_argument("--dump", default=None, help="also write the team-game frame to this parquet")
    A = ap.parse_args()

    seasons = seasons_arg(A.seasons)
    pw, sch = load(seasons)
    tg = team_game_shapes(pw, sch, seasons)
    if tg.empty:
        raise SystemExit("no team-games built — check the seasons requested")
    md = report(tg, db=A.db)

    out = A.out if os.path.isabs(A.out) else os.path.join(ROOT, A.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(md + "\n")
    print(f"\nwrote {os.path.relpath(out, ROOT)}")
    if A.dump:
        tg.to_parquet(A.dump); print(f"wrote {A.dump}")


if __name__ == "__main__":
    main()
