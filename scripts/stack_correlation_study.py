"""How correlated are same-game outcomes once the projection is removed? Grounds the
"joint simulation" question: Model_Burke's quantiles are per-player marginals; a joint sim
would only add the dependence structure measured here.

Residual = actual PPR − FFA weekly baseline (2016-25, players projected ≥ 5). Roles by FFA
projection rank within team-week (QB1, RB1, WR1, WR2, TE1). Pairs within a game.
Reports Pearson correlation of residuals by pair, and P(both above own p75 residual) vs the
independence value 0.0625.
Usage: python scripts/stack_correlation_study.py
"""
import os, re, glob, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t, flush=True); L.append(t)
wk = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "weekly_skill_2015_2025.parquet"))
gl = pd.read_parquet(os.path.join(ROOT, "data", "sabersim", "game_lines_2015_2026.parquet"))
for c in ("team", "opponent_team"): wk[c] = wk[c].map(lambda t: FIX.get(t, t))
for c in ("team", "opp"): gl[c] = gl[c].map(lambda t: FIX.get(t, t))
wk = wk.drop_duplicates(["player_id", "season", "week"]); wk["nname"] = wk.player_display_name.map(norm)
ids = wk.sort_values(["season", "week"]).drop_duplicates(["nname", "position", "season"], keep="last")[["nname", "position", "season", "player_id"]]
def score_ppr(d):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    return (z("pass_yds") * .04 + z("pass_tds") * 4 - z("pass_int") * 2 + z("rush_yds") * .1 + z("rush_tds") * 6
            + z("rec") + z("rec_yds") * .1 + z("rec_tds") * 6 - z("fumbles_lost") * 2 + z("two_pts") * 2 + z("return_tds") * 6)
rows = []
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    s_, w_ = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
    if s_ >= 2026 or s_ < 2016 or w_ > 18: continue
    d = pd.read_csv(f, na_values=["NA"], low_memory=False); d = d[d.position.isin(["QB", "RB", "WR", "TE"])].copy()
    d["team"] = d.team.map(lambda t: FIX.get(t, t)); d["nname"] = d.player.map(norm); d["season"], d["week"] = s_, w_
    d["proj"] = score_ppr(d); rows.append(d[["nname", "position", "team", "season", "week", "proj"]].drop_duplicates(["nname", "position"]))
h = pd.concat(rows).merge(ids, on=["nname", "position", "season"], how="inner")
h = h.merge(wk[["player_id", "season", "week", "fantasy_points_ppr", "team"]].rename(columns={"fantasy_points_ppr": "actual", "team": "team_box"}),
            on=["player_id", "season", "week"], how="inner")
h = h[h.team == h.team_box]                       # FFA team must match the box score team (traded/misassigned rows out)
h = h.merge(gl[["season", "week", "team", "opp"]].drop_duplicates(["season", "week", "team"]), on=["season", "week", "team"], how="left").dropna(subset=["opp"])
h = h[h.proj >= 5].copy(); h["res"] = h.actual - h.proj
h["rk"] = h.sort_values("proj", ascending=False).groupby(["season", "week", "team", "position"]).cumcount() + 1
h["role"] = h.position + h.rk.astype(str)
h = h[h.role.isin(["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1"])]
# standardise residual within position so pairs are comparable; p75 flags within position
h["z"] = h.groupby("position").res.transform(lambda s: (s - s.mean()) / s.std())
h["hi"] = h.groupby("position").res.transform(lambda s: s > s.quantile(0.75)).astype(int)
h["game"] = h.apply(lambda r: "_".join([str(r.season), str(r.week)] + sorted([r.team, r.opp])), axis=1)
piv_z = h.pivot_table(index=["game", "team"], columns="role", values="z")
piv_hi = h.pivot_table(index=["game", "team"], columns="role", values="hi")
say(f"# Same-game residual correlation, FFA weekly baseline 2016-25 ({pd.Timestamp.now():%Y-%m-%d})\n")
say(f"{len(h):,} player-weeks projected ≥ 5, {piv_z.index.get_level_values(0).nunique():,} games.\n")
say("## Within a team\n"); say("| pair | Pearson r (z-residuals) | P(both > own p75) | independence | n |"); say("|---|---|---|---|---|")
pairs = [("QB1", "WR1"), ("QB1", "WR2"), ("QB1", "TE1"), ("QB1", "RB1"), ("WR1", "WR2"), ("WR1", "TE1"), ("RB1", "WR1"), ("RB1", "RB2"), ("RB1", "TE1")]
for a, b in pairs:
    d = piv_z[[a, b]].dropna(); e = piv_hi.loc[d.index, [a, b]]
    say(f"| {a}–{b} | {d[a].corr(d[b]):+.3f} | {((e[a] == 1) & (e[b] == 1)).mean():.3f} | 0.0625 | {len(d):,} |")
# across teams: join the opponent's roles
opp = piv_z.reset_index(); opp["team"] = opp.game.str.split("_").str[2:].apply(lambda t: t)   # placeholder replaced below
tz = piv_z.reset_index(); th = piv_hi.reset_index()
def opp_of(row):
    parts = row.game.split("_"); a, b = parts[2], parts[3]; return b if row.team == a else a
tz["opp"] = tz.apply(opp_of, axis=1); th["opp"] = th.apply(opp_of, axis=1)
oz = tz.merge(tz.drop(columns="opp").rename(columns={"team": "opp"}), on=["game", "opp"], suffixes=("", "_opp"))
oh = th.merge(th.drop(columns="opp").rename(columns={"team": "opp"}), on=["game", "opp"], suffixes=("", "_opp"))
say("\n## Across the two teams in a game\n"); say("| pair | Pearson r | P(both > own p75) | independence | n |"); say("|---|---|---|---|---|")
for a, b in [("QB1", "QB1_opp"), ("QB1", "WR1_opp"), ("WR1", "WR1_opp"), ("RB1", "QB1_opp"), ("RB1", "RB1_opp"), ("QB1", "RB1_opp")]:
    d = oz[[a, b]].dropna(); e = oh.loc[d.index, [a, b]]
    say(f"| {a}–{b} | {d[a].corr(d[b]):+.3f} | {((e[a] == 1) & (e[b] == 1)).mean():.3f} | 0.0625 | {len(d):,} |")
# how much of the within-team dependence is a shared "game environment" factor? team total residual
tt = h.groupby(["game", "team"]).res.sum().rename("team_res")
say("\n## Shared factor check\n")
d = piv_z.join(tt); d["others"] = d.team_res
for r in ["QB1", "WR1", "RB1", "TE1"]:
    x = h[h.role == r].merge(tt.reset_index(), on=["game", "team"]); x["rest"] = x.team_res - x.res
    say(f"- {r} residual vs rest-of-team residual (same game): r = {x.res.corr(x.rest):+.3f} (n={len(x):,})")
os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "reports", "stack_correlation_study.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote outputs/reports/stack_correlation_study.md")
