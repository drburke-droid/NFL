"""Context effects on individual fantasy points — three questions for the SaberSim pipeline.

  A. Opponent strength: does the opponent's rolling fantasy points ALLOWED to a position
     (prior 6 games, no lookahead) explain what the projection misses?
  B. Vegas: do spread / total / implied team total explain the miss?
  C. Injury absence: when a team's RB1 / WR1 / TE1 / QB1 misses a game, where do the
     points go? Does the next man up absorb most? Does the team pass more without RB1?

Two baselines:  (1) FFA weekly consensus (2023-25 weeks with a file) — the pipeline's
actual baseline;  (2) each player's own prior-6-game average (2012-25) — bigger sample.
Residual = actual_ppr - baseline.  Output: outputs/reports/context_effects_study.md
"""
import os, re, sqlite3, warnings, glob
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS = ["QB", "RB", "WR", "TE"]
TEAM_FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t); L.append(t)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
wk = pd.read_sql("""SELECT player_id, player_display_name p, position, season, week, team, opponent_team opp,
                    attempts, passing_yards, passing_tds, carries, rushing_yards, targets, receptions,
                    receiving_yards, fantasy_points_ppr fp
                    FROM nflv_weekly WHERE season_type='REG' AND season BETWEEN 2011 AND 2025
                    AND position IN ('QB','RB','WR','TE')""", con)
gl = pd.read_sql("SELECT season, week, team, opp, game_total, team_spread spread, implied_team_total itt "
                 "FROM nflv_game_lines WHERE game_type='REG' AND season BETWEEN 2011 AND 2025", con)
con.close()
for c in ("team", "opp"):
    wk[c] = wk[c].map(lambda t: TEAM_FIX.get(t, t)); gl[c] = gl[c].map(lambda t: TEAM_FIX.get(t, t))
wk = wk.drop_duplicates(["player_id", "season", "week"])
wk["t"] = wk.season * 100 + wk.week

# ---------- team-week frame ----------
tw = wk.groupby(["season", "week", "t", "team", "opp"]).agg(
    att=("attempts", "sum"), car=("carries", "sum"), tgt=("targets", "sum"),
    fp_team=("fp", "sum")).reset_index()
tw["pass_rate"] = tw.att / (tw.att + tw.car)
tw = tw.merge(gl, on=["season", "week", "team", "opp"], how="left")

# ---------- A. opponent points allowed by position, rolling prior 6 team-games ----------
allowed = wk.groupby(["season", "week", "t", "opp", "position"]).fp.sum().reset_index().rename(
    columns={"opp": "def_team", "fp": "fp_allowed"})
allowed = allowed.sort_values("t")
def roll_prior(g, n=6):
    return g.shift(1).rolling(n, min_periods=3).mean()
allowed["dvp_r6"] = allowed.groupby(["def_team", "position", "season"]).fp_allowed.transform(roll_prior)
# league mean for the same position/season/week for a z-score
lg = allowed.groupby(["season", "week", "position"]).dvp_r6.agg(["mean", "std"]).reset_index()
allowed = allowed.merge(lg, on=["season", "week", "position"])
allowed["dvp_z"] = (allowed.dvp_r6 - allowed["mean"]) / allowed["std"]
dvp = allowed[["season", "week", "def_team", "position", "dvp_r6", "dvp_z"]].rename(columns={"def_team": "opp"})

# ---------- player baselines ----------
wk = wk.sort_values("t")
wk["own_r6"] = wk.groupby(["player_id", "season"]).fp.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
wk = wk.merge(dvp, on=["season", "week", "opp", "position"], how="left")
wk = wk.merge(gl, on=["season", "week", "team", "opp"], how="left")
wk["res_own"] = wk.fp - wk.own_r6

# FFA baseline 2023-25
rows = []
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    s_, w_ = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
    if s_ >= 2026 or w_ > 18: continue
    d = pd.read_csv(f, na_values=["NA"]); d = d[d.position.isin(POS)]
    if "rec" not in d.columns: continue
    z = lambda c: d[c].fillna(0)
    d["ffa"] = (z("pass_yds") * .04 + z("pass_tds") * 4 - z("pass_int") * 2 + z("rush_yds") * .1
                + z("rush_tds") * 6 + z("rec") + z("rec_yds") * .1 + z("rec_tds") * 6 - z("fumbles_lost") * 2)
    d["nname"] = d.player.map(norm); d["season"], d["week"] = s_, w_
    rows.append(d[["nname", "position", "season", "week", "ffa"]].drop_duplicates(["nname", "position"]))
ffa = pd.concat(rows)
wk["nname"] = wk.p.map(norm)
wf = wk.merge(ffa, on=["nname", "position", "season", "week"], how="inner")
wf = wf[wf.ffa >= 3]; wf["res_ffa"] = wf.fp - wf.ffa
say(f"# Context effects study ({pd.Timestamp.now():%Y-%m-%d})\n")
say(f"Box scores 2011-25 REG: {len(wk):,} player-weeks. FFA-baseline sample 2023-25: {len(wf):,} "
    f"player-weeks (FFA >= 3). Own-r6 sample: {wk.own_r6.notna().sum():,}.\n")

def decile_table(df, col, res, label, q=5):
    d = df.dropna(subset=[col, res]).copy()
    d["bin"] = pd.qcut(d[col], q, duplicates="drop")
    t = d.groupby("bin")[res].agg(["mean", "size"]).round(2)
    say(f"\n{label}\n"); say("| bin | mean residual | n |"); say("|---|---|---|")
    for b, r in t.iterrows(): say(f"| {b} | {r['mean']:+.2f} | {int(r['size'])} |")
def slope(df, x, y):
    d = df.dropna(subset=[x, y])
    if len(d) < 50: return np.nan, np.nan, len(d)
    b = np.polyfit(d[x], d[y], 1)[0]; r = np.corrcoef(d[x], d[y])[0, 1]
    return b, r, len(d)

say("## A. Opponent points allowed by position (prior-6, z-scored within week)\n")
say("Residual vs opponent DvP z-score. Slope = residual points per 1 SD of opponent generosity.\n")
say("| position | baseline | slope | corr | n |"); say("|---|---|---|---|---|")
for pos in POS:
    for lab, df, res in (("FFA", wf, "res_ffa"), ("own-r6", wk[wk.own_r6 >= 3], "res_own")):
        b, r, n = slope(df[df.position == pos], "dvp_z", res)
        say(f"| {pos} | {lab} | {b:+.3f} | {r:+.3f} | {n:,} |")
decile_table(wf, "dvp_z", "res_ffa", "FFA residual by opponent-DvP quintile (all positions):")
decile_table(wf[wf.position == "WR"], "dvp_z", "res_ffa", "WR only:")
decile_table(wf[wf.position == "RB"], "dvp_z", "res_ffa", "RB only:")

say("\n## B. Vegas game line\n")
say("| position | baseline | x | slope (pts per unit) | corr | n |"); say("|---|---|---|---|---|---|")
for pos in POS:
    for lab, df, res in (("FFA", wf, "res_ffa"), ("own-r6", wk[wk.own_r6 >= 3], "res_own")):
        for x in ("itt", "spread", "game_total"):
            b, r, n = slope(df[df.position == pos], x, res)
            say(f"| {pos} | {lab} | {x} | {b:+.3f} | {r:+.3f} | {n:,} |")
decile_table(wf, "itt", "res_ffa", "FFA residual by implied team total quintile (all positions):")
decile_table(wf, "spread", "res_ffa", "FFA residual by spread quintile (negative = favoured):")
# joint: does DvP add anything once ITT is in? walk-forward ridge on residual, 2024-25
from sklearn.linear_model import Ridge
say("\n### Walk-forward: does correcting the FFA baseline with these help? (fit on prior weeks, test 2024-25)\n")
wf["tk"] = wf.season * 100 + wf.week
feats = {"itt only": ["itt"], "dvp only": ["dvp_z"], "itt+spread": ["itt", "spread"], "itt+spread+dvp": ["itt", "spread", "dvp_z"]}
say("| position | features | MAE baseline | MAE corrected | gain |"); say("|---|---|---|---|---|")
for pos in POS:
    d = wf[(wf.position == pos)].dropna(subset=["itt", "spread", "dvp_z"]).copy()
    for lab, cols in feats.items():
        pred = pd.Series(np.nan, index=d.index)
        for t in sorted(d.tk.unique()):
            if t < 202401: continue
            tr, te = d[d.tk < t], d[d.tk == t]
            if len(tr) < 200 or not len(te): continue
            m = Ridge(alpha=10.0).fit(tr[cols].values, tr.res_ffa.values)
            pred.loc[te.index] = m.predict(te[cols].values)
        ok = pred.notna()
        mae0 = (d.res_ffa[ok]).abs().mean(); mae1 = (d.res_ffa[ok] - pred[ok]).abs().mean()
        say(f"| {pos} | {lab} | {mae0:.3f} | {mae1:.3f} | {(mae0-mae1)/mae0:+.2%} |")

# ---------- C. injury absence ----------
say("\n## C. When a starter is absent, where do the points go?\n")
say("Starter = team's leader at the position by prior-3-game usage (carries for RB, targets for WR/TE, "
    "attempts for QB), who PLAYED the previous team game. Absence = no box-score row that week while the "
    "team played. Deltas are vs the team's / player's own prior-3-game averages (2012-25 REG).\n")
wk["car_r3"] = wk.groupby(["player_id", "season"]).carries.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
wk["tgt_r3"] = wk.groupby(["player_id", "season"]).targets.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
wk["att_r3"] = wk.groupby(["player_id", "season"]).attempts.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
wk["fp_r3"] = wk.groupby(["player_id", "season"]).fp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
tw = tw.sort_values("t")
for c in ("att", "car", "pass_rate", "fp_team"):
    tw[f"{c}_r3"] = tw.groupby(["team", "season"])[c].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
# team game sequence
tw["gnum"] = tw.groupby(["team", "season"]).cumcount()
gseq = tw[["team", "season", "week", "gnum"]]
wk = wk.merge(gseq, on=["team", "season", "week"], how="left")
usage = {"RB": "car_r3", "WR": "tgt_r3", "TE": "tgt_r3", "QB": "att_r3"}
results = {}
for pos in POS:
    u = usage[pos]
    rows = []
    # candidate starter per team-game: leader by usage among players who played the team's previous game
    for (team, season), g in wk[wk.position == pos].groupby(["team", "season"]):
        games = tw[(tw.team == team) & (tw.season == season)].sort_values("gnum")
        for i in range(1, len(games)):
            prev, cur = games.iloc[i - 1], games.iloc[i]
            played_prev = g[g.week == prev.week]
            if not len(played_prev): continue
            lead = played_prev.sort_values(u, ascending=False).iloc[0]
            if pd.isna(lead[u]) or lead[u] < {"RB": 8, "WR": 4, "TE": 3, "QB": 15}[pos]: continue
            here = g[g.week == cur.week]
            absent = lead.player_id not in set(here.player_id)
            vac = lead.fp_r3 if pd.notna(lead.fp_r3) else lead.fp
            # same-position teammates this game
            mates = here[here.player_id != lead.player_id]
            top = mates.sort_values("fp", ascending=False).head(1)
            top_gain = float(top.fp.iloc[0] - (top.fp_r3.iloc[0] if pd.notna(top.fp_r3.iloc[0]) else 0)) if len(top) else 0.0
            rest_gain = float((mates.fp - mates.fp_r3.fillna(0)).sum() - top_gain) if len(mates) > 1 else 0.0
            others = wk[(wk.team == team) & (wk.season == season) & (wk.week == cur.week) & (wk.position != pos)]
            oth = {p_: float((others[others.position == p_].fp - others[others.position == p_].fp_r3.fillna(0)).sum()) for p_ in POS if p_ != pos}
            rows.append({"absent": absent, "vac": vac, "top_gain": top_gain, "rest_gain": rest_gain, **{f"gain_{k}": v for k, v in oth.items()},
                         "d_att": cur.att - cur.att_r3, "d_car": cur.car - cur.car_r3, "d_pr": cur.pass_rate - cur.pass_rate_r3,
                         "d_fp_team": cur.fp_team - cur.fp_team_r3, "season": season})
    r = pd.DataFrame(rows).dropna(subset=["vac"]); r = r[r.vac >= 3]   # a real starter, and no divide-by-zero shares
    results[pos] = r
    a, b = r[r.absent], r[~r.absent]
    say(f"\n### {pos}1 absent — {len(a):,} team-games (vs {len(b):,} with the starter playing); "
        f"avg vacated = {a.vac.mean():.1f} PPR\n")
    say("| quantity | starter absent | starter plays | difference |"); say("|---|---|---|---|")
    for lab, c in (("next man up, gain vs his own r3 (PPR)", "top_gain"), ("other same-pos players, gain", "rest_gain"),
                   *[(f"{p_}s total gain", f"gain_{p_}") for p_ in POS if p_ != pos],
                   ("team pass attempts, delta", "d_att"), ("team carries, delta", "d_car"),
                   ("team pass rate, delta", "d_pr"), ("team total PPR, delta", "d_fp_team")):
        say(f"| {lab} | {a[c].mean():+.2f} | {b[c].mean():+.2f} | {a[c].mean()-b[c].mean():+.2f} |")
    share_top = (a.top_gain - b.top_gain.mean()) / a.vac
    share_rest = (a.rest_gain - b.rest_gain.mean()) / a.vac
    oth_share = sum((a[f"gain_{p_}"] - b[f"gain_{p_}"].mean()) for p_ in POS if p_ != pos) / a.vac
    say(f"\nShare of the vacated {pos}1 points (net of the normal-week drift): next man up **{share_top.mean():.0%}**, "
        f"other {pos}s **{share_rest.mean():.0%}**, other positions **{oth_share.mean():.0%}**, "
        f"team total change {((a.d_fp_team.mean()-b.d_fp_team.mean())/a.vac.mean()):+.0%} of vacated.")
    say(f"Median next-man share {share_top.median():.0%}; next man absorbs >=50% in {(share_top>=0.5).mean():.0%} of cases.")

os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "reports", "context_effects_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/context_effects_study.md")
