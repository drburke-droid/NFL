"""
A. BUZZ LEAD-LAG — which offseason month's Wikipedia pageviews predict cheap-pool
   hits? (August is the validated dart feature; can June/July — or the Feb->July
   ramp — give the edge EARLIER, before the market sees the camp spike?)

B. COLLEGE PEDIGREE — Power-5 vs small-school, by draft capital: do small-school
   RB/WR hit at different rates (rookie + year-2), controlling for round?

Pools/outcomes match bench_darts_backtest.py: cheap = FFA AAV <= $8 (played T-1,
RB/WR/TE); RELIABLE = 6+ weekly-starter weeks; STAR = top-12 positional PPG.
Report: outputs/reports/buzz_leadlag.md
"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
pd.set_option("display.width", 240)
START_RANK = {"RB": 24, "WR": 24, "TE": 12}

S = pd.read_sql("""SELECT player_id, player_display_name name, position, season, games,
                          fantasy_points_ppr FROM nflv_season
                   WHERE position IN ('RB','WR','TE')""", con)
W = pd.read_sql("""SELECT player_id, position, season, week, fantasy_points_ppr pts
                   FROM nflv_weekly WHERE season_type='REG' AND position IN ('RB','WR','TE')""", con)
FFA = pd.read_sql("""SELECT season, player_id, ffa_aav FROM nflv_ffa_league
                     WHERE player_id IS NOT NULL""", con).drop_duplicates(["season", "player_id"])
BZ = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
DRF = pd.read_sql("""SELECT gsis_id player_id, season draft_season, round, college, position
                     FROM nflv_draft WHERE season>=2016 AND position IN ('RB','WR')""", con)
con.close()

BZ["ym"] = BZ.ym.astype(int)
BZ["yr"] = BZ.ym // 100
BZ["mo"] = BZ.ym % 100
W["wk_rank"] = W.groupby(["season", "week", "position"])["pts"].rank(ascending=False)
W["starter_wk"] = W.apply(lambda r: r.wk_rank <= START_RANK[r.position], axis=1)
out = W.groupby(["player_id", "season"]).agg(starter_wks=("starter_wk", "sum")).reset_index()
ppg = S.assign(ppg=S.fantasy_points_ppr / S.games.clip(lower=1))
ppg["pos_rank"] = ppg[ppg.games >= 10].groupby(["season", "position"])["ppg"].rank(ascending=False)
out = out.merge(ppg[["player_id", "season", "pos_rank"]], on=["player_id", "season"], how="left")
out["reliable"] = out.starter_wks >= 6
out["star"] = out.pos_rank <= 12

L = ["# Buzz lead-lag + college pedigree\n"]

# ---------- A. month-by-month buzz ----------
L.append("## A. Which offseason month's wiki buzz predicts cheap-pool hits?\n")
L.append("Signal per month m of draft year T: log((views_m+10)/(median Oct(T-1)-Jan(T) +10)).")
L.append("Top-quartile signal within each season's cheap pool vs the pool baseline.\n")
frames = []
for T in range(2016, 2026):
    pool = S[(S.season == T - 1) & (S.games >= 4)][["player_id", "name", "position"]].copy()
    aav = FFA[FFA.season == T].set_index("player_id")["ffa_aav"]
    pool = pool[pool.player_id.map(aav).fillna(0) <= 8]
    base = BZ[((BZ.yr == T - 1) & (BZ.mo >= 10)) | ((BZ.yr == T) & (BZ.mo == 1))] \
        .groupby("player_id")["views"].median()
    for m in [2, 3, 4, 5, 6, 7, 8]:
        mv = BZ[(BZ.yr == T) & (BZ.mo == m)].set_index("player_id")["views"]
        pool[f"m{m}"] = np.log((pool.player_id.map(mv).fillna(0) + 10) /
                               (pool.player_id.map(base).fillna(0) + 10))
    pool["ramp27"] = pool.m7 - pool.m2          # Feb -> July change
    oT = out[out.season == T].set_index("player_id")
    pool["reliable"] = pool.player_id.map(oT["reliable"]).fillna(False)
    pool["star"] = pool.player_id.map(oT["star"]).fillna(False)
    pool["season"] = T
    frames.append(pool)
P = pd.concat(frames)
br, bs = P.reliable.mean(), P.star.mean()
L.append(f"pool n={len(P):,} · baseline reliable {br:.1%} star {bs:.1%}\n")
L.append("| signal | top-Q reliable | lift | top-Q star | lift |")
L.append("|---|---|---|---|---|")
for c, lab in [("m2", "Feb buzz"), ("m3", "Mar"), ("m4", "Apr"), ("m5", "May"),
               ("m6", "Jun"), ("m7", "Jul"), ("m8", "Aug"), ("ramp27", "Feb→Jul ramp")]:
    q = P.groupby("season")[c].transform(lambda s: s.rank(pct=True)) >= 0.75
    g = P[q]
    L.append(f"| {lab} | {g.reliable.mean():.1%} | {g.reliable.mean()/br:.1f}x | "
             f"{g.star.mean():.1%} | {g.star.mean()/bs:.1f}x |")

# ---------- B. college pedigree ----------
L.append("\n## B. College pedigree (P5 vs small-school), RB/WR draftees 2016-2025\n")
P5 = {"Alabama","Georgia","LSU","Florida","Tennessee","Auburn","Texas A&M","Ole Miss","Mississippi State",
      "Arkansas","Kentucky","South Carolina","Missouri","Vanderbilt","Ohio State","Michigan","Penn State",
      "Michigan State","Wisconsin","Iowa","Minnesota","Illinois","Indiana","Purdue","Northwestern","Nebraska",
      "Maryland","Rutgers","Texas","Oklahoma","Oklahoma State","Baylor","TCU","Texas Tech","Kansas State",
      "Kansas","Iowa State","West Virginia","Clemson","Florida State","Miami","North Carolina","NC State",
      "Duke","Wake Forest","Virginia","Virginia Tech","Pittsburgh","Louisville","Syracuse","Boston College",
      "Georgia Tech","USC","UCLA","Oregon","Washington","Stanford","California","Oregon State",
      "Washington State","Arizona","Arizona State","Utah","Colorado","Notre Dame"}
# nflv_draft abbreviates ("Ohio St.", "Miami (FL)") — normalize before matching
FIX = {"Mississippi": "Ole Miss", "Miami (FL)": "Miami", "Southern California": "USC", "Pitt": "Pittsburgh"}
DRF["college_n"] = DRF.college.str.replace(" St.", " State", regex=False).replace(FIX)
DRF["p5"] = DRF.college_n.isin(P5)
res = []
for _, r in DRF.iterrows():
    o = out[(out.player_id == r.player_id) & (out.season.isin([r.draft_season, r.draft_season + 1]))]
    res.append(dict(rnd=r["round"], p5=r.p5, hit=bool(o.reliable.any()), star=bool(o.star.any())))
R = pd.DataFrame(res)
L.append("Hit = 6+ weekly-starter weeks in rookie OR year-2 season; star = top-12 PPG either year.\n")
L.append("| round | school | n | hit | star |")
L.append("|---|---|---|---|---|")
for lo, hi, lab in [(1, 2, "rd 1-2"), (3, 4, "rd 3-4"), (5, 7, "rd 5-7")]:
    for p5 in [True, False]:
        g = R[(R.rnd >= lo) & (R.rnd <= hi) & (R.p5 == p5)]
        if len(g) >= 12:
            L.append(f"| {lab} | {'P5' if p5 else 'small/G5'} | {len(g)} | {g.hit.mean():.0%} | {g.star.mean():.0%} |")

txt = "\n".join(L)
open(os.path.join(ROOT, "outputs", "reports", "buzz_leadlag.md"), "w", encoding="utf-8").write(txt)
print(txt)
