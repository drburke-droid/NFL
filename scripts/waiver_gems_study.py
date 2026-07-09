"""
Waiver-wire "diamonds" by position, measured against THIS league's actual drafts
(outputs/espn_drafts.csv, 2023-25): which UNDRAFTED players went on to produce
starter-level weeks, when did they first flash (could an aggressive FAAB bidder have
them by week 6?), and how much starter production was recoverable after the flash.

Starter level = weekly positional top-N in league scoring (PPR + 6pt passTD - 1 INT):
QB/TE top-12, RB/WR top-24 (12 teams, 1QB/2RB/2WR/1TE/1FLEX).
Emergence week = first REG week hitting starter level. Post-flash value = starter-level
weeks AFTER the flash week (you pick them up the following Tuesday).
Also: the "streaming ceiling" - avg pts of the best undrafted player each week, vs the
replacement level baked into draft VORP. Appends to outputs/reports/waiver_wire.md.
"""
import csv, os, re, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

drafts = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
drafted = {}
for r in drafts:
    drafted.setdefault(int(r["season"]), set()).add(norm(r["player"]))

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
wk = pd.read_sql("""SELECT player_display_name nm, position, season, week,
                           fantasy_points_ppr, passing_tds, passing_interceptions
                    FROM nflv_weekly
                    WHERE season BETWEEN 2023 AND 2025 AND season_type='REG'
                      AND position IN ('QB','RB','WR','TE')""", con)
wk["pts"] = (wk.fantasy_points_ppr.fillna(0) + 2 * wk.passing_tds.fillna(0)
             + wk.passing_interceptions.fillna(0))
wk["nm"] = wk.nm.map(norm)
TOPN = {"QB": 12, "TE": 12, "RB": 24, "WR": 24}
wk["prk"] = wk.groupby(["season", "week", "position"]).pts.rank(ascending=False, method="min")
wk["starter_wk"] = wk.apply(lambda r: r.prk <= TOPN[r.position], axis=1)
wk["udfa"] = wk.apply(lambda r: r.nm not in drafted.get(r.season, set()), axis=1)

print("Undrafted-in-THIS-league players reaching weekly starter level (QB/TE top-12, RB/WR top-24)")
print("gem bar: >=4 starter-level weeks AFTER the first flash\n")
rows = []
for (season, pos), sub in wk[wk.udfa].groupby(["season", "position"]):
    gems = []
    for nm, p in sub.groupby("nm"):
        p = p.sort_values("week")
        fl = p[p.starter_wk]
        if not len(fl): continue
        w0 = int(fl.week.iloc[0])
        post = p[p.week > w0]
        n_post = int(post.starter_wk.sum())
        if n_post >= 4:
            gems.append((nm, w0, n_post, post.pts.mean()))
    gems.sort(key=lambda g: -g[2])
    by6 = [g for g in gems if g[1] <= 5]
    rows.append((season, pos, len(gems), len(by6), gems))
    ex = ", ".join(f"{g[0].title()} (wk{g[1]}: {g[2]} starter wks, {g[3]:.1f} ppg after)" for g in gems[:3])
    print(f"  {season} {pos}: {len(gems)} gems, {len(by6)} flashed by wk5  {('- ' + ex) if ex else ''}")

print("\nTOTALS by position (3 seasons):")
df = pd.DataFrame(rows, columns=["season", "pos", "gems", "by6", "detail"])
for pos in ("QB", "RB", "WR", "TE"):
    d = df[df.pos == pos]
    allg = [g for det in d.detail for g in det]
    sw = sum(g[2] for g in allg)
    print(f"  {pos}: {d.gems.sum()} gems ({d.gems.sum()/3:.1f}/yr), {d.by6.sum()} by wk5, "
          f"{sw} recoverable starter-weeks ({sw/3:.0f}/yr)")

print("\nStreaming ceiling: avg pts of the BEST undrafted player each week (vs draft-pool replacement):")
best = wk[wk.udfa].groupby(["season", "week", "position"]).pts.max().reset_index()
top3 = wk[wk.udfa].sort_values("pts", ascending=False).groupby(
    ["season", "week", "position"]).pts.nth([0, 1, 2]).reset_index() if True else None
for pos in ("QB", "RB", "WR", "TE"):
    b = best[best.position == pos]
    t24 = wk[(wk.position == pos) & (wk.prk == TOPN[pos])].pts.mean()
    print(f"  {pos}: best-udfa avg {b.pts.mean():.1f} ppg | starter cutline (rank {TOPN[pos]}) {t24:.1f} ppg")

# how many mid-priced draft picks ($8-30) actually beat what the wire later offered?
dr = pd.DataFrame(drafts)
dr["season"] = dr.season.astype(int); dr["bid"] = dr.bid.astype(float); dr["nm"] = dr.player.map(norm)
mid = dr[(dr.bid.between(8, 30)) & (dr.keeper == "False") & dr.pos.isin(["RB", "WR"])]
per = wk.groupby(["season", "nm", "position"]).agg(sw=("starter_wk", "sum")).reset_index()
mid = mid.merge(per, left_on=["season", "nm", "pos"], right_on=["season", "nm", "position"], how="left")
mid["sw"] = mid.sw.fillna(0)
print("\nmid-priced ($8-30, non-keeper) draft picks vs the bar a waiver gem sets (>=6 starter weeks):")
for pos in ("RB", "WR"):
    m = mid[mid.pos == pos]
    print(f"  {pos}: {len(m)} picks, {(m.sw >= 6).mean():.0%} delivered >=6 starter weeks "
          f"(median {m.sw.median():.0f}); $ spent on the misses: "
          f"${m[m.sw < 6].bid.sum():.0f} of ${m.bid.sum():.0f}")

os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
with open(os.path.join(ROOT, "outputs", "reports", "waiver_wire.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Waiver gems by position vs this league's drafts (`waiver_gems_study.py`)\n\n```\n"
             + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/waiver_wire.md")
