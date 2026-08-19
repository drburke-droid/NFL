"""
CONTEXT-DEPENDENCY SLEEPERS - do cheap players hit more in specific TEAM contexts?

Cheap pool (FFA AAV<=$8 or unlisted, 2016-2025, same frame family as
dart_synergy_study.py). Player-side traits x team-side context, hit = 6+
weekly-starter weeks in season T.

Team context for season T = the player's T team's T-1 profile (ex-ante-ish:
approximates what a drafter knew in August; QB changes between seasons add
noise and are acknowledged, not modeled):
  MOBILE_QB   T-1 QB rushing yards/game, top third of teams
  GOOD_PASS   T-1 passing EPA/game, top third   (QB quality)
  GOOD_RUSH   T-1 RB rushing EPA/game, top third (run-game/OL proxy)
  PASS_VOL    T-1 pass attempts share, top third

Player traits: YOUNG (age<=24 at T), FAST_RB (combine speed score wt*200/40^4,
top 40% of drafted RBs), FLASH (T-1 best-4-wk>=12).

Each hypothesis cell: n, hit rate, lift vs the position's cheap base, Fisher p.
Output: outputs/reports/context_sleepers.md
"""
import os, sqlite3
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
S = pd.read_sql("""SELECT player_id, player_display_name name, position, season,
                          recent_team team, games FROM nflv_season
                   WHERE position IN ('RB','WR','TE')""", con)
W = pd.read_sql("""SELECT player_id, position, season, week, team, fantasy_points_ppr pts,
                          passing_epa, rushing_epa, rushing_yards, attempts, carries
                   FROM nflv_weekly WHERE season_type='REG'""", con)
FFA = pd.read_sql("""SELECT season, player_id, ffa_aav FROM nflv_ffa_league
                     WHERE player_id IS NOT NULL""", con).drop_duplicates(["season", "player_id"])
AGE = pd.read_sql("SELECT player_id, season, age FROM season_dataset", con).drop_duplicates(["player_id", "season"])
CMB = pd.read_sql("SELECT pfr_id, player_name, pos, wt, forty FROM nflv_combine WHERE forty IS NOT NULL AND wt IS NOT NULL", con)
con.close()

# outcomes
SR = {"RB": 24, "WR": 24, "TE": 12}
sk = W[W.position.isin(["RB", "WR", "TE"])].copy()
sk["wk_rank"] = sk.groupby(["season", "week", "position"])["pts"].rank(ascending=False)
sk["starter_wk"] = sk.apply(lambda r: r.wk_rank <= SR[r.position], axis=1)
out = sk.groupby(["player_id", "season"]).starter_wk.sum().rename("sw").reset_index()

# team-season context (from ALL positions' weekly rows)
qbw = W[W.position == "QB"] if "QB" in set(W.position) else W[W.passing_epa.notna()]
tw = W.groupby(["team", "season", "week"]).agg(
    qb_rush=("rushing_yards", lambda s: np.nan),  # placeholder, filled below
).reset_index()[["team", "season", "week"]]
qb_rush = W[(W.position == "QB")].groupby(["team", "season", "week"]).rushing_yards.sum().rename("qbr")
pass_epa = W.groupby(["team", "season", "week"]).passing_epa.sum().rename("pepa")
rb_epa = W[W.position == "RB"].groupby(["team", "season", "week"]).rushing_epa.sum().rename("repa")
att = W.groupby(["team", "season", "week"]).agg(att=("attempts", "sum"), car=("carries", "sum"))
ctx = pd.concat([qb_rush, pass_epa, rb_epa, att], axis=1).reset_index()
ctx = ctx.groupby(["team", "season"]).agg(qbr=("qbr", "mean"), pepa=("pepa", "mean"),
                                          repa=("repa", "mean"), att=("att", "mean"), car=("car", "mean")).reset_index()
ctx["pass_share"] = ctx.att / (ctx.att + ctx.car)
for c, f in [("MOBILE_QB", "qbr"), ("GOOD_PASS", "pepa"), ("GOOD_RUSH", "repa"), ("PASS_VOL", "pass_share")]:
    ctx[c] = ctx.groupby("season")[f].transform(lambda s: (s.rank(pct=True) >= 2 / 3).astype(int))

# speed score (RB): map via pfr id when available, else name+pos merge on combine
CMB["speed"] = CMB.wt * 200 / (CMB.forty ** 4)
rbc = CMB[CMB.pos == "RB"].copy()
rbc["fast"] = (rbc.speed.rank(pct=True) >= 0.60).astype(int)

pools = []
for T in range(2016, 2026):
    prev = S[(S.season == T - 1) & (S.games >= 4)][["player_id", "name", "position"]]
    cur_team = S[S.season == T].set_index("player_id")["team"]
    aav = FFA[FFA.season == T].set_index("player_id")["ffa_aav"]
    p = prev.copy()
    p["aav"] = p.player_id.map(aav)
    p = p[(p.aav.fillna(0) <= 8)].copy()
    p["team"] = p.player_id.map(cur_team)
    p = p[p.team.notna()]
    w1 = W[(W.season == T - 1)].sort_values("pts", ascending=False).groupby("player_id").head(4) \
          .groupby("player_id")["pts"].mean()
    p["FLASH"] = (p.player_id.map(w1).fillna(0) >= 12).astype(int)
    p["age"] = p.player_id.map(AGE[AGE.season == T - 1].set_index("player_id")["age"]) + 1
    p["YOUNG"] = (p.age <= 24).astype(int)
    c = ctx[ctx.season == T - 1].set_index("team")
    for k in ("MOBILE_QB", "GOOD_PASS", "GOOD_RUSH", "PASS_VOL"):
        p[k] = p.team.map(c[k]).fillna(0).astype(int)
    p["sw"] = p.player_id.map(out[out.season == T].set_index("player_id")["sw"]).fillna(0)
    p["hit"] = p.sw >= 6
    p["season"] = T
    pools.append(p)
P = pd.concat(pools, ignore_index=True)

# fast flag by normalized name (combine table has no gsis id)
import re as _re
_nrm = lambda x: _re.sub(r"\s+", " ", _re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                         _re.sub(r"[^a-z ]", "", str(x).lower()))).strip()
rbc["nm"] = rbc.player_name.map(_nrm)
FASTN = rbc.drop_duplicates("nm").set_index("nm")["fast"]
P["FAST"] = P.name.map(lambda n: FASTN.get(_nrm(n), np.nan))

_out = []
def pr(*a): s = " ".join(str(x) for x in a); _out.append(s); print(s)

def cell(mask, pos, lbl):
    d = P[(P.position == pos) & mask]
    base = P[P.position == pos].hit.mean()
    if len(d) < 15:
        pr(f"  {lbl:44s} n={len(d):4d}  (too thin)"); return
    _, fp = stats.fisher_exact([[int(d.hit.sum()), len(d) - int(d.hit.sum())],
                                [int(P[P.position == pos].hit.sum()), len(P[P.position == pos])]])
    pr(f"  {lbl:44s} n={len(d):4d}  hit {d.hit.mean():5.1%}  lift {d.hit.mean()/base:4.1f}x  p={fp:.3f}")

pr(f"cheap pool n={len(P)}; base hit: RB {P[P.position=='RB'].hit.mean():.1%}, "
   f"WR {P[P.position=='WR'].hit.mean():.1%}, TE {P[P.position=='TE'].hit.mean():.1%}")
pr(f"FAST coverage (RBs with combine speed): {P[P.position=='RB'].FAST.notna().mean():.0%}\n")

pr("RB context cells:")
cell(P.YOUNG == 1, "RB", "YOUNG RB (control - is it just youth?)")
cell(P.MOBILE_QB == 1, "RB", "RB on MOBILE-QB team")
cell((P.MOBILE_QB == 1) & (P.YOUNG == 1), "RB", "YOUNG RB on MOBILE-QB team")
cell((P.MOBILE_QB == 0) & (P.YOUNG == 1), "RB", "YOUNG RB on pocket-QB team (control)")
cell(P.MOBILE_QB == 0, "RB", "RB on pocket-QB team")
cell(P.GOOD_RUSH == 1, "RB", "RB on GOOD-RUSH team (OL proxy)")
cell((P.GOOD_RUSH == 1) & (P.YOUNG == 1), "RB", "YOUNG RB on GOOD-RUSH team")
cell(P.FAST == 1, "RB", "FAST RB (speed score top-40%)")
cell((P.FAST == 1) & (P.YOUNG == 1), "RB", "YOUNG FAST RB")
cell((P.FAST == 1) & (P.MOBILE_QB == 1), "RB", "FAST RB on MOBILE-QB team")
cell((P.FAST == 1) & (P.YOUNG == 1) & (P.MOBILE_QB == 1), "RB", "YOUNG FAST RB on MOBILE-QB team")
cell((P.FAST == 1) & (P.GOOD_RUSH == 1), "RB", "FAST RB on GOOD-RUSH team")
pr("\nWR context cells:")
cell(P.YOUNG == 1, "WR", "YOUNG WR (control - is it just youth?)")
cell(P.GOOD_PASS == 1, "WR", "WR with GOOD QB (top-third pass EPA)")
cell((P.GOOD_PASS == 1) & (P.YOUNG == 1), "WR", "YOUNG WR with GOOD QB")
cell(P.GOOD_PASS == 0, "WR", "WR with bad/avg QB")
cell(P.PASS_VOL == 1, "WR", "WR on high-volume pass offense")
cell((P.PASS_VOL == 1) & (P.YOUNG == 1), "WR", "YOUNG WR on high-volume pass offense")
cell((P.GOOD_PASS == 1) & (P.FLASH == 1), "WR", "FLASH WR with GOOD QB")
pr("\nTE context cells:")
cell(P.GOOD_PASS == 1, "TE", "TE with GOOD QB")
cell((P.GOOD_PASS == 1) & (P.YOUNG == 1), "TE", "YOUNG TE with GOOD QB")

open(os.path.join(ROOT, "outputs", "reports", "context_sleepers.md"), "w", encoding="utf-8").write(
    "# Context-dependency sleepers - team context x cheap-player traits (2016-25)\n\n"
    "Generated by `scripts/context_sleeper_study.py`. Context = the player's CURRENT-season\n"
    "team's PRIOR-season profile (QB changes between seasons add noise). Lift vs the\n"
    "position's cheap-pool base; Fisher exact p vs that base.\n\n```\n" + "\n".join(_out) + "\n```\n")
print("\nwrote outputs/reports/context_sleepers.md")
