"""
BLOCKER ENTRENCHMENT — does the guy IN FRONT of an heir cap the heir's value?

User hypothesis (the Tank Bigsby / Saquon case): an heir behind a recently-paid,
prime-age, workload-dominant starter only eats via injury, and even then late in
the season — capping fantasy value. Test on all HEIR candidates 2016-2025
(backup with 40+ carries who out-ran his own lead back on ypc AND EPA/carry).

Blocker features at season T (ex-ante):
  age          blocker's age in T
  highpick     drafted round 1-2 within 3 years of T (team investment)
  newly_acq    joined the team within the last 2 seasons (signed/traded = paid)
  share        blocker's share of team RB carries in T-1

Outcomes for the heir in T: RELIABLE (6+ weekly top-24 RB weeks), STAR (top-12
season PPG), and TIMING — which week his starter-weeks arrive (early weeks are
worth more: weeks 1-9 vs 10-18).

Report: outputs/reports/blocker_entrenchment.md
"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
pd.set_option("display.width", 240)

S = pd.read_sql("""SELECT player_id, player_display_name name, position, season, recent_team team,
                          games, carries, rushing_yards, rushing_epa, fantasy_points_ppr
                   FROM nflv_season WHERE position='RB'""", con)
W = pd.read_sql("""SELECT player_id, season, week, fantasy_points_ppr pts
                   FROM nflv_weekly WHERE season_type='REG' AND position='RB'""", con)
DR = pd.read_sql("SELECT gsis_id player_id, season draft_season, round FROM nflv_draft", con) \
       .dropna().drop_duplicates("player_id")
AGE = pd.read_sql("SELECT player_id, season, age FROM season_dataset", con).drop_duplicates(["player_id", "season"])
con.close()

W["wk_rank"] = W.groupby(["season", "week"])["pts"].rank(ascending=False)
W["starter_wk"] = W.wk_rank <= 24

rows = []
for T in range(2016, 2026):
    rb = S[(S.season == T - 1) & (S.carries >= 40)].copy()
    rb["ypc"] = rb.rushing_yards / rb.carries
    rb["epc"] = rb.rushing_epa / rb.carries
    team_car = rb.groupby("team")["carries"].sum()
    for tm, g in rb.groupby("team"):
        if len(g) < 2: continue
        g = g.sort_values("carries", ascending=False)
        lead = g.iloc[0]
        for _, b in g.iloc[1:].iterrows():
            if not (b.ypc > lead.ypc and b.epc > lead.epc): continue
            # blocker features at T
            bl_age = AGE[(AGE.player_id == lead.player_id) & (AGE.season == T - 1)]["age"]
            bl_age = (bl_age.iloc[0] + 1) if len(bl_age) else 27
            drow = DR[DR.player_id == lead.player_id]
            highpick = int(len(drow) > 0 and drow.iloc[0]["round"] <= 2 and T - drow.iloc[0].draft_season <= 3)
            prev2 = S[(S.player_id == lead.player_id) & (S.season == T - 2)]
            newly = int(len(prev2) > 0 and prev2.iloc[0].team != lead.team)
            share = lead.carries / max(team_car[tm], 1)
            # heir outcome in T
            hw = W[(W.player_id == b.player_id) & (W.season == T)]
            swks = hw[hw.starter_wk]["week"].tolist()
            ppgT = S[(S.player_id == b.player_id) & (S.season == T) & (S.games >= 10)]
            star = False
            if len(ppgT):
                pool = S[(S.season == T) & (S.games >= 10)].assign(ppg=lambda d: d.fantasy_points_ppr / d.games)
                star = (pool.ppg > ppgT.iloc[0].fantasy_points_ppr / ppgT.iloc[0].games).sum() < 12
            rows.append(dict(season=T, heir=b["name"], blocker=lead["name"],
                             bl_age=bl_age, highpick=highpick, newly=newly, share=round(share, 2),
                             starter_wks=len(swks), early_wks=sum(1 for w in swks if w <= 9),
                             first_wk=min(swks) if swks else None,
                             reliable=len(swks) >= 6, star=star))

d = pd.DataFrame(rows)
d["entrenched"] = ((d.highpick == 1) | ((d.newly == 1) & (d.bl_age < 29))) & (d.share >= 0.55)
d["vulnerable"] = (d.bl_age >= 29) & (~d.entrenched)

L = ["# Blocker entrenchment — does the starter in front cap the heir?\n",
     f"HEIR candidates 2016-2025: n={len(d)}. Baseline: reliable {d.reliable.mean():.0%}, star {d.star.mean():.1%},",
     f"median starter-weeks {d.starter_wks.median():.0f}, median first starter-week {d.first_wk.median():.0f}.\n"]

def block(label, mask):
    g = d[mask]
    if len(g) < 8: return f"**{label}** (n={len(g)}): too small"
    fw = g[g.first_wk.notna()].first_wk
    return (f"**{label}** (n={len(g)}): reliable {g.reliable.mean():.0%}, star {g.star.mean():.1%}, "
            f"avg starter-wks {g.starter_wks.mean():.1f} (early wks 1-9: {g.early_wks.mean():.1f}), "
            f"median first starter-week {fw.median():.0f}" if len(fw) else "no starter weeks")

for lab, m in [("ENTRENCHED blocker (recent rd1-2 pick, or newly-acquired prime-age; 55%+ share)", d.entrenched),
               ("VULNERABLE blocker (age 29+, not entrenched)", d.vulnerable),
               ("middle (neither)", ~d.entrenched & ~d.vulnerable),
               ("blocker age <27", d.bl_age < 27), ("blocker age 27-28", (d.bl_age >= 27) & (d.bl_age < 29)),
               ("blocker age 29+", d.bl_age >= 29),
               ("blocker share >= 65%", d.share >= 0.65), ("blocker share < 55%", d.share < 0.55),
               ("newly-acquired blocker", d.newly == 1), ("recent rd1-2 blocker", d.highpick == 1)]:
    L.append("- " + block(lab, m))

L.append("\n## Heirs who hit STAR — who was the blocker?\n```")
for _, r in d[d.star].sort_values("season").iterrows():
    L.append(f"  {r.season} {r.heir:<22} behind {r.blocker:<22} age {r.bl_age:.0f} share {r.share:.0%}"
             f"{' HIGHPICK' if r.highpick else ''}{' NEW' if r.newly else ''} -> first starter-wk {r.first_wk:.0f}")
L.append("```")

out = os.path.join(ROOT, "outputs", "reports", "blocker_entrenchment.md")
open(out, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\nWrote {out}")
