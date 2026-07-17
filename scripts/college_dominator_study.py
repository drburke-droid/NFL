"""
Does college DOMINATOR (share of team production, final college season) predict
NFL hits for RB/WR draftees — beyond draft capital?

Classes 2018-2024 (college data 2014+, NFL outcomes through 2025). WR signal =
dom_rec (share of team receiving yds/TDs); RB = dom_scrim (share of scrimmage
yards). Outcomes as everywhere: HIT = 6+ weekly-starter weeks rookie or year-2;
STAR = top-12 positional PPG either year. Report: outputs/reports/college_dominator.md
"""
import os, re, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
pd.set_option("display.width", 220)

CP = pd.read_sql("SELECT * FROM nflv_college_prod", con)
DRF = pd.read_sql("""SELECT gsis_id player_id, season draft_season, round, college,
                            pfr_player_name name, position
                     FROM nflv_draft WHERE season BETWEEN 2018 AND 2026
                       AND position IN ('RB','WR')""", con)
W = pd.read_sql("""SELECT player_id, position, season, week, fantasy_points_ppr pts
                   FROM nflv_weekly WHERE season_type='REG' AND position IN ('RB','WR')""", con)
S = pd.read_sql("""SELECT player_id, position, season, games, fantasy_points_ppr
                   FROM nflv_season WHERE position IN ('RB','WR')""", con)
con.close()

SUF = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
def norm(s):
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ").replace(",", "")
    return re.sub(r"\s+", " ", SUF.sub("", s)).strip()

ALIAS = {"Mississippi": "Ole Miss", "Miami (FL)": "Miami", "Southern California": "USC",
         "Pitt": "Pittsburgh", "Central Florida": "UCF", "Brigham Young": "BYU",
         "Louisiana State": "LSU", "Texas Christian": "TCU", "Southern Methodist": "SMU"}
def norm_college(c):
    c = str(c).replace(" St.", " State")
    return ALIAS.get(c, c)

CP["nm"] = CP.player.map(norm)
CP["team_n"] = CP.team.map(norm_college)
DRF["nm"] = DRF.name.map(norm)
DRF["college_n"] = DRF.college.map(norm_college)

# outcomes
START_RANK = {"RB": 24, "WR": 24}
W["wk_rank"] = W.groupby(["season", "week", "position"])["pts"].rank(ascending=False)
W["starter_wk"] = W.apply(lambda r: r.wk_rank <= START_RANK[r.position], axis=1)
out = W.groupby(["player_id", "season"]).agg(sw=("starter_wk", "sum")).reset_index()
ppg = S.assign(ppg=S.fantasy_points_ppr / S.games.clip(lower=1))
ppg["pos_rank"] = ppg[ppg.games >= 10].groupby(["season", "position"])["ppg"].rank(ascending=False)
out = out.merge(ppg[["player_id", "season", "pos_rank"]], on=["player_id", "season"], how="left")

rows, misses = [], 0
for _, r in DRF.iterrows():
    fin = CP[(CP.season == r.draft_season - 1) & (CP.nm == r.nm)]
    if len(fin) > 1:
        tm = fin[fin.team_n == r.college_n]
        fin = tm if len(tm) else fin.iloc[:0]
    if not len(fin):
        misses += 1; continue
    f = fin.iloc[0]
    o = out[(out.player_id == r.player_id) & (out.season.isin([r.draft_season, r.draft_season + 1]))]
    rows.append(dict(cls=r.draft_season, name=r["name"], pos=r.position, rnd=r["round"],
                     dom=f.dom_rec if r.position == "WR" else f.dom_scrim,
                     hit=bool((o.sw >= 6).any()), star=bool((o.pos_rank <= 12).any())))
d = pd.DataFrame(rows)
matched = len(d) / (len(d) + misses)

L = [f"# College dominator vs NFL hits (classes 2018-2024, rookies/yr-2)\n",
     f"matched {len(d)}/{len(d)+misses} draftees ({matched:.0%}) to final-college-season production.",
     "WR metric = dom_rec (team receiving share), RB = dom_scrim (team scrimmage share).\n"]
val = d[d.cls <= 2024]
L.append("| pos | round | dominator | n | hit | star |")
L.append("|---|---|---|---|---|---|")
for pos in ["WR", "RB"]:
    for lo, hi, lab in [(1, 2, "rd 1-2"), (3, 7, "rd 3-7")]:
        g = val[(val.pos == pos) & (val.rnd >= lo) & (val.rnd <= hi)].copy()
        if len(g) < 20: continue
        g["ter"] = pd.qcut(g.dom, 3, labels=["low", "mid", "high"])
        for t in ["high", "mid", "low"]:
            x = g[g.ter == t]
            L.append(f"| {pos} | {lab} | {t} (avg {x.dom.mean():.0%}) | {len(x)} | {x.hit.mean():.0%} | {x.star.mean():.0%} |")

# continuous check
from scipy.stats import spearmanr
for pos in ["WR", "RB"]:
    g = val[(val.pos == pos) & (val.rnd >= 3)]
    r1 = spearmanr(g.dom, g.hit)[0]
    L.append(f"\n{pos} rd3-7: spearman(dominator, hit) = {r1:.3f} (n={len(g)})")

txt = "\n".join(L)
open(os.path.join(ROOT, "outputs", "reports", "college_dominator.md"), "w", encoding="utf-8").write(txt)
print(txt)
