"""
POSITIONAL FILL score — the opportunity-cost metric for auction allocation.

Question: at each position, what's the probability of filling the slot with an
"adequate" player WITHOUT paying up at the draft — via a cheap late pick or an
early-season waiver add — and what PPG can that fill realistically deliver?
The gap between a paid-up starter and the expected fill is the true opportunity
cost of allocating auction $ to that position.

Method (seasons 2016-2025, league scoring: PPR + 6-pt pass TD):
  - Draft-cost pools from FFA preseason ADP (nflv_ffa_league):
      LATE-DRAFT pool: ADP 100-170  (the $1-3 end of a 12-team draft)
      WAIVER pool:     ADP > 170 or not ranked (undrafted)
  - "Discoverable": flashed by week 5 (a starter-level weekly finish in wks 1-4:
      QB/TE top-12 of week, RB/WR top-24) — you could realistically have added them.
  - Fill quality = rest-of-season PPG (weeks 5-17, min 6 games).
  - Tier bars per position from that season's ROS ranks among ALL players:
      elite = QB top-5 / RB top-6 / WR top-6 / TE top-3
      starter = QB top-12 / RB top-24 / WR top-24 / TE top-12

Outputs per position: P(fill >= starter), P(fill >= elite), expected fill PPG
(median best + median 3rd-best discoverable = what's left after competition),
and the opportunity-cost table. Report: outputs/reports/positional_fill.md
"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
OUT = os.path.join(ROOT, "outputs", "reports")
pd.set_option("display.width", 220)

SEASONS = range(2016, 2026)
ELITE = {"QB": 5, "RB": 6, "WR": 6, "TE": 3}
START = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}

con = sqlite3.connect(DB)
W = pd.read_sql("""SELECT player_id, player_display_name player, position, season, week,
                          passing_tds, passing_interceptions, fantasy_points_ppr
                   FROM nflv_weekly
                   WHERE season_type='REG' AND position IN ('QB','RB','WR','TE')
                     AND season>=2016""", con)
F = pd.read_sql("""SELECT season, player_id, ffa_adp FROM nflv_ffa_league
                   WHERE player_id IS NOT NULL""", con)

W["pts"] = W.fantasy_points_ppr + np.where(W.position.eq("QB"),
                                           2*W.passing_tds.fillna(0) + W.passing_interceptions.fillna(0), 0)
W = W.merge(F.drop_duplicates(["season","player_id"]), on=["season","player_id"], how="left")

# weekly positional rank (for the "flash" test)
W["wk_rank"] = W.groupby(["season","week","position"])["pts"].rank(ascending=False)

L = []
def p(t=""):
    print(t); L.append(t)

p("# Positional FILL score — what can you get without paying?\n")
p("Pools by FFA preseason ADP: LATE-DRAFT = ADP 100-170 ($1-3 picks), WAIVER = ADP>170/unranked.")
p("Discoverable = starter-level weekly finish in weeks 1-4. Fill quality = weeks 5-17 PPG (min 6 g).\n")

rows, detail = [], []
for pos in ["QB","RB","WR","TE"]:
    for yr in SEASONS:
        w = W[(W.position.eq(pos)) & (W.season.eq(yr))]
        ros = w[w.week.between(5,17)].groupby(["player_id","player"]) \
               .agg(g=("pts","count"), ppg=("pts","mean")).reset_index()
        ros = ros[ros.g>=6]
        ros["ros_rank"] = ros.ppg.rank(ascending=False)
        elite_bar = ros[ros.ros_rank<=ELITE[pos]].ppg.min()
        start_bar = ros[ros.ros_rank<=START[pos]].ppg.min()

        early = w[w.week<=4]
        flashed = set(early[early.wk_rank<=START[pos]].player_id)
        adp = w.drop_duplicates("player_id").set_index("player_id")["ffa_adp"]

        for pool, lo, hi in [("late", 100, 170), ("waiver", 170, 10**6)]:
            if pool=="late":
                ids = set(adp[(adp>=lo)&(adp<=hi)].index)
            else:
                ids = set(adp[(adp>hi-10**6+170)].index) | set(adp[adp.isna()].index)
            cand = ros[ros.player_id.isin(ids & flashed)].sort_values("ppg", ascending=False)
            best = cand.ppg.iloc[0] if len(cand) else np.nan
            third = cand.ppg.iloc[2] if len(cand)>=3 else np.nan
            rows.append(dict(pos=pos, season=yr, pool=pool,
                             n_starter=int((cand.ppg>=start_bar).sum()),
                             n_elite=int((cand.ppg>=elite_bar).sum()),
                             best_ppg=best, third_ppg=third,
                             start_bar=start_bar, elite_bar=elite_bar))
            if len(cand):
                b = cand.iloc[0]
                detail.append(dict(pos=pos, season=yr, pool=pool, player=b.player, ros_ppg=round(b.ppg,1)))

R = pd.DataFrame(rows)

p("## Fill probability & expected fill PPG (per position, 2016-2025)\n")
p("```")
p(f"{'pos':4s} {'pool':7s} {'P(>=starter)':>12s} {'P(>=elite)':>10s} {'E[#starter/yr]':>14s} "
  f"{'med best PPG':>12s} {'med 3rd PPG':>11s} {'starter bar':>11s} {'elite bar':>9s}")
for pos in ["QB","RB","WR","TE"]:
    for pool in ["late","waiver"]:
        g = R[(R.pos.eq(pos)) & (R.pool.eq(pool))]
        p(f"{pos:4s} {pool:7s} {(g.n_starter>0).mean():>12.0%} {(g.n_elite>0).mean():>10.0%} "
          f"{g.n_starter.mean():>14.1f} {g.best_ppg.median():>12.1f} {g.third_ppg.median():>11.1f} "
          f"{g.start_bar.median():>11.1f} {g.elite_bar.median():>9.1f}")
p("```\n")

p("## Best discoverable fill by season (waiver pool)\n")
D = pd.DataFrame(detail)
p("```")
for pos in ["QB","RB","WR","TE"]:
    d = D[(D.pos.eq(pos)) & (D.pool.eq("waiver"))]
    p(f"{pos}: " + " | ".join(f"{r.season} {r.player} {r.ros_ppg}" for r in d.itertuples()))
p("```\n")

# ---------------- opportunity cost ----------------
p("## Opportunity cost of paying up (tier-1 PPG vs realistic fill)\n")
p("Cost of NOT paying = (tier-1 season pts) - (expected fill season pts over the same 13 ROS weeks")
p("+ startable early weeks). Using median-3rd-best discoverable fill (competition-adjusted).\n")
board = pd.read_sql("""SELECT b.player_display_name player, b.position, b.proj_pts, b.pred_ppg,
                              f.ffa_points, f.ffa_aav
                       FROM draft_board_2026 b
                       LEFT JOIN nflv_ffa_league f ON f.player_id=b.player_id AND f.season=2026
                       WHERE b.position IN ('QB','RB','WR','TE')""", con)
p("```")
p(f"{'pos':4s} {'tier-1 PPG (elite bar)':>21s} {'fill PPG (waiver 3rd)':>21s} {'PPG gap':>8s} {'season pts gap':>14s}")
gaps = {}
for pos in ["QB","RB","WR","TE"]:
    g = R[(R.pos.eq(pos)) & (R.pool.eq("waiver"))]
    t1 = g.elite_bar.median(); fill = g.third_ppg.median()
    gap = t1 - fill; gaps[pos] = gap*13
    p(f"{pos:4s} {t1:>21.1f} {fill:>21.1f} {gap:>8.1f} {gap*13:>14.0f}")
p("```")
p("\nInterpretation: season points you CANNOT recover from the wire if you skip tier-1 at that position.")
p("Higher = position deserves your auction dollars; lower = fill it cheap and reallocate.\n")

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "positional_fill.md"), "w", encoding="utf-8").write("\n".join(L))
print("Wrote outputs/reports/positional_fill.md")
con.close()
