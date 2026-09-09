"""Early-season waiver-wire RB gems: which week-1..3 signals best predict rest-of-season
RB2+ production for RBs nobody drafted?

Universe: RBs outside the preseason FFA top-36 at the position (or absent from FFA
entirely) — i.e. on the wire in a 12-team league — who played in week k (k=1,2,3).
Target: rest-of-season PPR per game (weeks k+1..17, >=4 games) and HIT = top-24 RB by
ROS ppg that season (RB2 or better), STAR = top-12.
Signals through week k: snap share, team RB-carry share, carries/g, targets/g, touches,
target share, TDs, rushing EPA, YPC, PPR/g, draft round, age, years exp, preseason FFA
rank, whether the team's preseason RB1 is absent, and week-over-week trend.
Output: outputs/reports/waiver_rb_study.md + a scorer for 2026 (scripts/waiver_rb_screen.py
reuses the fitted rule). 2013-2025 REG.
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t); L.append(t)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
wk = pd.read_sql("""SELECT player_id, player_display_name p, season, week, team, carries, rushing_yards,
    rushing_tds, rushing_epa, targets, receptions, receiving_yards, receiving_tds, target_share,
    fantasy_points_ppr fp FROM nflv_weekly WHERE season_type='REG' AND position='RB'
    AND season BETWEEN 2013 AND 2025""", con)
teamcar = pd.read_sql("""SELECT season, week, team, SUM(carries) team_car, SUM(attempts) team_att
    FROM nflv_weekly WHERE season_type='REG' AND season BETWEEN 2013 AND 2025 GROUP BY 1,2,3""", con)
snaps = pd.read_sql("""SELECT season, week, pfr_player_id, offense_pct FROM nflv_snaps
    WHERE game_type='REG' AND position='RB' AND season BETWEEN 2013 AND 2025""", con)
ros = pd.read_sql("""SELECT DISTINCT season, gsis_id player_id, pfr_id, years_exp, entry_year, draft_number, birth_date
    FROM nflv_rosters WHERE position='RB' AND season BETWEEN 2013 AND 2025""", con)
ffa = pd.read_sql("SELECT season, player_id, ffa_pos_rank, ffa_points, ffa_adp FROM nflv_ffa_proj WHERE position='RB'", con)
con.close()
wk["team"] = wk.team.map(lambda t: FIX.get(t, t)); teamcar["team"] = teamcar.team.map(lambda t: FIX.get(t, t))
ros = ros.dropna(subset=["player_id"]).drop_duplicates(["season", "player_id"])
wk = wk.merge(ros, on=["season", "player_id"], how="left")
wk = wk.merge(snaps.rename(columns={"pfr_player_id": "pfr_id"}), on=["season", "week", "pfr_id"], how="left")
wk = wk.merge(teamcar, on=["season", "week", "team"], how="left")
wk = wk.merge(ffa, on=["season", "player_id"], how="left")
wk["car_share"] = wk.carries / wk.team_car.replace(0, np.nan)
wk["touches"] = wk.carries + wk.receptions
wk["age"] = wk.season - pd.to_datetime(wk.birth_date, errors="coerce").dt.year
wk["draft_number"] = pd.to_numeric(wk.draft_number, errors="coerce"); wk["years_exp"] = pd.to_numeric(wk.years_exp, errors="coerce")
wk["draft_round"] = np.where(wk.draft_number.isna(), 8, np.ceil(wk.draft_number.fillna(0) / 32)).astype(float)
wk["pre_rank"] = wk.ffa_pos_rank.fillna(99)
# preseason RB1 per team (best FFA rank on that team's roster this season, by week-1 team)
team_rb1 = (wk[wk.week <= 3].sort_values("pre_rank").drop_duplicates(["season", "team"])
            [["season", "team", "player_id"]].rename(columns={"player_id": "rb1_id"}))
wk = wk.merge(team_rb1, on=["season", "team"], how="left")
# was the team's RB1 absent (no row) in this week?
played = wk[["season", "week", "player_id"]].assign(played=1).rename(columns={"player_id": "rb1_id"})
wk = wk.merge(played, on=["season", "week", "rb1_id"], how="left")
wk["rb1_absent"] = ((wk.played.isna()) & (wk.rb1_id != wk.player_id)).astype(int)

# rest-of-season target
def ros_ppg(k):
    r = wk[(wk.week > k) & (wk.week <= 17)].groupby(["season", "player_id"]).agg(g=("fp", "size"), ppg=("fp", "mean")).reset_index()
    r = r[r.g >= 4]
    r["rank"] = r.groupby("season").ppg.rank(ascending=False)
    return r.rename(columns={"ppg": "ros_ppg", "g": "ros_g", "rank": "ros_rank"})

FEATS = ["snap_pct", "car_share", "carries_g", "targets_g", "touches_g", "tshare", "tds", "epa", "ypc", "ppg",
         "draft_round", "age", "years_exp", "pre_rank", "rb1_absent", "last_snap", "snap_trend"]
def frame(k):
    c = wk[wk.week <= k].sort_values("week")
    g = c.groupby(["season", "player_id"])
    f = pd.DataFrame({
        "snap_pct": g.offense_pct.mean(), "last_snap": g.offense_pct.last(),
        "snap_trend": g.offense_pct.last() - g.offense_pct.first(),
        "car_share": g.car_share.mean(), "carries_g": g.carries.mean(), "targets_g": g.targets.mean(),
        "touches_g": g.touches.mean(), "tshare": g.target_share.mean(),
        "tds": (g.rushing_tds.sum() + g.receiving_tds.sum()), "epa": g.rushing_epa.mean(),
        "ypc": g.rushing_yards.sum() / g.carries.sum().replace(0, np.nan), "ppg": g.fp.mean(),
        "draft_round": g.draft_round.first(), "age": g.age.first(), "years_exp": g.years_exp.first(),
        "pre_rank": g.pre_rank.first(), "rb1_absent": g.rb1_absent.max(), "games": g.fp.size(),
        "team": g.team.last(), "p": g.p.last()}).reset_index()
    f = f.merge(ros_ppg(k), on=["season", "player_id"], how="left")
    f["hit"] = (f.ros_rank <= 24).astype(int); f["star"] = (f.ros_rank <= 12).astype(int); f["flex"] = (f.ros_rank <= 36).astype(int)
    f.loc[f.ros_rank.isna(), ["hit", "star", "flex"]] = 0
    return f

say(f"# Early-season waiver-wire RB study ({pd.Timestamp.now():%Y-%m-%d})\n")
say("Universe = RBs outside the preseason FFA top-36 (undrafted in a 12-team league) who played "
    "through week k. HIT = finishes top-24 RB by rest-of-season PPR/game (>=4 games); STAR = top-12. 2013-25 REG.\n")
best = {}
for k in (1, 2, 3):
    f = frame(k); avail = f[(f.pre_rank > 36) & (f.games >= 1) & ~f.season.isin([2014, 2015])].copy()  # FFA table has <15 RBs those years
    say(f"\n## After week {k}: {len(avail):,} available RBs, {avail.hit.sum()} hits ({avail.hit.mean():.1%}), "
        f"{avail.star.sum()} stars ({avail.star.mean():.1%})\n")
    say("| signal | AUC (hit) | top-decile hit rate | top-decile star rate | mean ROS ppg top decile |")
    say("|---|---|---|---|---|")
    rows = []
    for c in FEATS:
        d = avail.dropna(subset=[c])
        if d[c].nunique() < 3: continue
        sign = 1 if c not in ("draft_round", "age", "pre_rank") else -1
        auc = roc_auc_score(d.hit, sign * d[c]) if d.hit.nunique() > 1 else np.nan
        top = d[sign * d[c] >= np.quantile(sign * d[c], 0.9)]
        rows.append((c, auc, top.hit.mean(), top.star.mean(), top.ros_ppg.mean(), len(top)))
    for c, auc, h, st, ppg, n in sorted(rows, key=lambda x: -x[1]):
        say(f"| {c} | {auc:.3f} | {h:.0%} (n={n}) | {st:.0%} | {ppg:.1f} |")
    # simple rules
    say(f"\nSimple screens after week {k} (available RBs):\n")
    say("| rule | n | RB2+ (top-24) | RB1 (top-12) | flex (top-36) | ROS ppg |"); say("|---|---|---|---|---|---|")
    rules = {
        "snap_pct >= 50%": avail.snap_pct >= 0.5,
        "snap_pct >= 60%": avail.snap_pct >= 0.6,
        "car_share >= 50%": avail.car_share >= 0.5,
        "touches/g >= 12": avail.touches_g >= 12,
        "touches/g >= 15": avail.touches_g >= 15,
        "targets/g >= 3": avail.targets_g >= 3,
        "RB1 absent & car_share >= 40%": (avail.rb1_absent == 1) & (avail.car_share >= 0.4),
        "snap >= 50% & draft rd <= 3": (avail.snap_pct >= 0.5) & (avail.draft_round <= 3),
        "snap >= 50% & age <= 25": (avail.snap_pct >= 0.5) & (avail.age <= 25),
        "touches >= 12 & targets >= 2": (avail.touches_g >= 12) & (avail.targets_g >= 2),
        "snap >= 55% & touches >= 12 & age <= 26": (avail.snap_pct >= 0.55) & (avail.touches_g >= 12) & (avail.age <= 26),
    }
    for lab, m in rules.items():
        s = avail[m.fillna(False)]
        if len(s): say(f"| {lab} | {len(s)} | {s.hit.mean():.0%} | {s.star.mean():.0%} | {s.flex.mean():.0%} | {s.ros_ppg.mean():.1f} |")
    # walk-forward logistic: leave-one-season-out, precision of the top-5 per season
    cols = ["snap_pct", "car_share", "touches_g", "targets_g", "draft_round", "age", "rb1_absent", "ppg"]
    d = avail.dropna(subset=cols).copy(); prec = []; picks = []
    for s in sorted(d.season.unique()):
        tr, te = d[d.season != s], d[d.season == s]
        m = LogisticRegression(C=0.5, max_iter=1000).fit(tr[cols].values, tr.hit.values)
        te = te.assign(score=m.predict_proba(te[cols].values)[:, 1]).sort_values("score", ascending=False)
        top5 = te.head(5); prec.append(top5.hit.mean()); picks.append(top5.assign(season=s))
    picks = pd.concat(picks)
    say(f"\nLeave-one-season-out logistic on ({', '.join(cols)}): top-5 per season hit rate "
        f"**{np.mean(prec):.0%}** (base rate {avail.hit.mean():.0%}), star rate {picks.star.mean():.0%}, "
        f"mean ROS ppg {picks.ros_ppg.mean():.1f}.")
    m = LogisticRegression(C=0.5, max_iter=1000).fit(d[cols].values, d.hit.values)
    say("Coefficients (standardised): " + ", ".join(f"{c} {b*d[c].std():+.2f}" for c, b in zip(cols, m.coef_[0])))
    best[k] = (m, cols, d[cols].mean(), d[cols].std())
    if k == 1:
        say("\nTop-5 picks per season after week 1 (what the rule would have told you):\n")
        say("| season | player | team | snap% | touches/g | RB1 absent | ROS ppg | hit |"); say("|---|---|---|---|---|---|---|---|")
        for r in picks.sort_values(["season", "score"], ascending=[False, False]).itertuples():
            say(f"| {r.season} | {r.p} | {r.team} | {r.snap_pct:.0%} | {r.touches_g:.1f} | {int(r.rb1_absent)} | "
                f"{r.ros_ppg if pd.notna(r.ros_ppg) else 0:.1f} | {'HIT' if r.hit else ''} |")

# how fast does the signal decay? does a week-1 hit still look like a hit by week 3?
say("\n## Timing: act after week 1 or wait?\n")
f1, f3 = frame(1), frame(3)
a1 = f1[(f1.pre_rank > 36) & ~f1.season.isin([2014, 2015])]; a3 = f3[(f3.pre_rank > 36) & ~f3.season.isin([2014, 2015])]
say(f"Available RBs with snap% >= 50% after week 1: {int((a1.snap_pct>=0.5).sum())}, hit rate {a1[a1.snap_pct>=0.5].hit.mean():.0%}. "
    f"After week 3: {int((a3.snap_pct>=0.5).sum())}, hit rate {a3[a3.snap_pct>=0.5].hit.mean():.0%}. "
    f"The week-3 screen is more precise but the week-1 pool contains players who are gone by week 3.")
import pickle
pickle.dump({k: (v[0], v[1]) for k, v in best.items()}, open(os.path.join(ROOT, "outputs", "models", "waiver_rb_logit.pkl"), "wb"))
open(os.path.join(ROOT, "outputs", "reports", "waiver_rb_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/waiver_rb_study.md + outputs/models/waiver_rb_logit.pkl")
