"""
Kyler Murray + Justin Jefferson united in MIN (2026) — what does history say?

Four angles:
  A. Where each sits on the skill curve right now (alpha_skill percentiles).
  B. Kyler's seasons by quality of his best WR (the DeAndre Hopkins natural
     experiment: prime-elite WR 2020-22 vs not).
  C. Jefferson's weekly production split by starting QB (and QB skill tier) —
     how QB-sensitive is he really?
  D. Every "newly united elite QB + elite WR" pair 2013-2025: first-year
     outcomes for both sides vs their prior baselines.

League scoring: QB = PPR + 2*passTD + 1*INT (6-pt league), WR = PPR.
Report: outputs/reports/kyler_jj_2026.md
"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
OUT = os.path.join(ROOT, "outputs", "reports")
pd.set_option("display.width", 220)

con = sqlite3.connect(DB)
S = pd.read_sql("""SELECT player_id, player_display_name player, position, season, recent_team team,
                          games, attempts, passing_tds, passing_interceptions, passing_epa,
                          targets, target_share, receiving_epa, wopr, fantasy_points_ppr
                   FROM nflv_season WHERE position IN ('QB','WR')""", con)
try:
    AGE = pd.read_sql("SELECT player_id, season, age FROM season_dataset", con).drop_duplicates(["player_id","season"])
except Exception:
    AGE = pd.DataFrame(columns=["player_id","season","age"])
S = S.merge(AGE, on=["player_id","season"], how="left")
W = pd.read_sql("""SELECT player_id, player_display_name player, position, season, week, team,
                          attempts, passing_tds, passing_interceptions,
                          targets, receptions, receiving_yards, receiving_tds, fantasy_points_ppr
                   FROM nflv_weekly WHERE season_type='REG'""", con)
A = pd.read_sql("SELECT player_id, position, season, alpha_skill, plays FROM player_skill_alpha", con)

S["lppg"] = np.where(S.position.eq("QB"),
                     (S.fantasy_points_ppr + 2*S.passing_tds.fillna(0) + S.passing_interceptions.fillna(0)) / S.games.clip(lower=1),
                     S.fantasy_points_ppr / S.games.clip(lower=1))

# alpha percentile within position-season
A["apct"] = A.groupby(["position","season"])["alpha_skill"].rank(pct=True)
S = S.merge(A[["player_id","season","apct","alpha_skill"]], on=["player_id","season"], how="left")

KY = S[(S.player=="Kyler Murray") & (S.position=="QB")].player_id.iloc[0]
JJ = S[(S.player=="Justin Jefferson") & (S.position=="WR")].player_id.iloc[0]

L = []  # report lines
def p(txt=""):
    print(txt); L.append(txt)

# ---------------- A. current skill ----------------
p("# Kyler Murray + Justin Jefferson — MIN 2026 pairing study\n")
p("## A. Where they sit on the skill curve (alpha_skill percentile within position)\n")
cur = S[(S.player_id.isin([KY,JJ])) & (S.season>=2021)].sort_values(["player","season"])
p("```")
p(cur[["player","season","team","games","lppg","target_share","apct"]]
  .assign(lppg=lambda d: d.lppg.round(1), target_share=lambda d: d.target_share.round(3),
          apct=lambda d: (100*d.apct).round(0)).to_string(index=False))
p("```\n")

# ---------------- B. Kyler by best-WR quality ----------------
p("## B. Kyler by quality of his best WR (the Hopkins experiment)\n")
wr = S[S.position=="WR"].copy()
rows = []
for yr in sorted(S[S.player_id.eq(KY)].season.unique()):
    ks = S[(S.player_id.eq(KY)) & (S.season.eq(yr))].iloc[0]
    mates = wr[(wr.season.eq(yr)) & (wr.team.eq(ks.team)) & (wr.games>=6)]
    if len(mates)==0: continue
    best = mates.sort_values("apct", ascending=False).iloc[0]
    rows.append(dict(season=yr, team=ks.team, kyler_games=ks.games, kyler_lppg=round(ks.lppg,1),
                     epa_play=round(ks.passing_epa/max(ks.attempts,1),3),
                     best_wr=best.player, wr_apct=round(100*(best.apct if pd.notna(best.apct) else 0)),
                     wr_ppg=round(best.lppg,1)))
kb = pd.DataFrame(rows)
p("```"); p(kb.to_string(index=False)); p("```")
elite = kb[kb.wr_apct>=90]; non = kb[kb.wr_apct<90]
if len(elite) and len(non):
    p(f"\nWith a top-decile WR: **{elite.kyler_lppg.mean():.1f} league PPG**, EPA/play {elite.epa_play.mean():.3f} "
      f"(n={len(elite)} seasons) — without: **{non.kyler_lppg.mean():.1f}**, EPA/play {non.epa_play.mean():.3f} (n={len(non)}).\n")

# ---------------- C. Jefferson by starting QB ----------------
p("## C. Jefferson weekly, split by his starting QB\n")
jw = W[(W.player_id.eq(JJ))].copy()
qbw = W[(W.position=="QB") & (W.attempts>=10)].sort_values("attempts", ascending=False) \
        .drop_duplicates(["season","week","team"])[["season","week","team","player_id","player"]] \
        .rename(columns={"player_id":"qb_id","player":"qb"})
jw = jw.merge(qbw, on=["season","week","team"], how="left")
jw = jw[jw.qb.notna() & (jw.targets>0)]
qa = A[A.position=="QB"][["player_id","season","apct"]].rename(columns={"player_id":"qb_id","apct":"qb_apct"})
jw = jw.merge(qa, on=["qb_id","season"], how="left")
byqb = jw.groupby("qb").agg(games=("week","count"), ppg=("fantasy_points_ppr","mean"),
                            tgt_g=("targets","mean"), yds_g=("receiving_yards","mean"),
                            qb_apct=("qb_apct","mean")).sort_values("games", ascending=False)
byqb = byqb[byqb.games>=3]
p("```")
p(byqb.assign(ppg=byqb.ppg.round(1), tgt_g=byqb.tgt_g.round(1), yds_g=byqb.yds_g.round(1),
              qb_apct=(100*byqb.qb_apct).round(0)).to_string())
p("```")
jw["tier"] = pd.cut(jw.qb_apct, [0,.5,.8,1.0], labels=["QB below median","QB 50-80th pct","QB top-20%"])
tier = jw.groupby("tier", observed=True).agg(games=("week","count"), ppg=("fantasy_points_ppr","mean"),
                                             tgt_g=("targets","mean"))
p("\nBy QB skill tier:")
p("```"); p(tier.round(1).to_string()); p("```\n")

# ---------------- D. newly united elite pairs ----------------
p("## D. Newly united elite QB + elite WR, 2013-2025 (first season together)\n")
qb = S[(S.position=="QB") & (S.games>=8)].copy()
wr8 = S[(S.position=="WR") & (S.games>=8)].copy()
qb["prk"] = qb.groupby("season")["lppg"].rank(ascending=False)
wr8["prk"] = wr8.groupby("season")["lppg"].rank(ascending=False)

def best_recent_rank(df, pid, T):
    x = df[(df.player_id.eq(pid)) & (df.season.isin([T-1, T-2]))]
    return x.prk.min() if len(x) else np.nan

pairs = []
for T in range(2013, 2026):
    qs = qb[qb.season.eq(T)]
    ws = wr8[wr8.season.eq(T)]
    for _, q in qs.iterrows():
        q_prev = S[(S.player_id.eq(q.player_id)) & (S.season.eq(T-1)) & (S.games>=1)]
        if len(q_prev)==0: continue
        for _, w in ws[ws.team.eq(q.team)].iterrows():
            w_prev = S[(S.player_id.eq(w.player_id)) & (S.season.eq(T-1)) & (S.games>=1)]
            if len(w_prev)==0: continue
            if q_prev.iloc[0].team == w_prev.iloc[0].team: continue  # already together
            qr = best_recent_rank(qb, q.player_id, T); wrk = best_recent_rank(wr8, w.player_id, T)
            if pd.isna(qr) or pd.isna(wrk) or qr>12 or wrk>12: continue  # both top-12 recently
            pairs.append(dict(season=T, team=q.team, qb=q.player, wr=w.player,
                              wr_age=w.age if pd.notna(w.age) else np.nan,
                              qb_prev_ppg=round(q_prev.iloc[0].lppg,1), qb_new_ppg=round(q.lppg,1),
                              qb_d=round(q.lppg-q_prev.iloc[0].lppg,1),
                              wr_prev_ppg=round(w_prev.iloc[0].lppg,1), wr_new_ppg=round(w.lppg,1),
                              wr_d=round(w.lppg-w_prev.iloc[0].lppg,1),
                              wr_ts=round(w.target_share if pd.notna(w.target_share) else 0,3)))
pr = pd.DataFrame(pairs).sort_values("season")
p("```"); p(pr.to_string(index=False)); p("```")
if len(pr):
    p(f"\nn={len(pr)} pairs. QB first-year change: median **{pr.qb_d.median():+.1f} PPG** "
      f"({(pr.qb_d>0).mean():.0%} improved). WR: median **{pr.wr_d.median():+.1f} PPG** "
      f"({(pr.wr_d>0).mean():.0%} improved).")
    p(f"WR median target share in year one together: {pr.wr_ts.median():.1%}\n")
    yng = pr[pr.wr_age < 28]; old = pr[pr.wr_age >= 28]
    if len(yng):
        p(f"**Age is the fault line.** WRs under 28 in year one together (n={len(yng)}): "
          f"median **{yng.wr_d.median():+.1f} PPG**, {(yng.wr_d>0).mean():.0%} improved "
          f"({', '.join(yng.wr + ' ' + yng.wr_d.map('{:+.1f}'.format))}).")
    if len(old):
        p(f"WRs 28+ (n={len(old)}): median **{old.wr_d.median():+.1f} PPG**, {(old.wr_d>0).mean():.0%} improved — "
          f"the pairing-flop list is almost entirely aging WRs changing teams.\n")

# ---------------- E. synthesis ----------------
p("## E. 2026 synthesis\n")
ffa = pd.read_sql("""SELECT player, ffa_points, ffa_pos_rank, ffa_adp, ffa_aav FROM nflv_ffa_league
                     WHERE season=2026 AND player IN ('Kyler Murray','Justin Jefferson')""", con)
brd = pd.read_sql("""SELECT player_display_name player, team, proj_pts, pred_ppg, proj_games, auction
                     FROM draft_board_2026
                     WHERE player_display_name IN ('Kyler Murray','Justin Jefferson')""", con)
p("```"); p("FFA league:"); p(ffa.to_string(index=False))
p("our board:"); p(brd.round(1).to_string(index=False)); p("```")
p("""
**Jefferson:** his weekly history maps QB quality directly to his scoring — top-20% QB
21.7 PPR PPG, 50-80th pct QB 18.6, below-median 15.4, and 11.0 in the 2025 McCarthy
games (6th-pct QB) despite a 30% target share. Kyler (~64th pct, and 78th-pct Cousins
is the nearest long-sample comp at 19.8 PPG over 54 games) puts Jefferson's 2026
per-game estimate around **18-19.5 PPR PPG** (~300-320 pts over 17), well above both
FFA (274) and our board (222, which is anchored to his QB-cratered 2025). At age 27
he profiles with the young-elite movers who improved, not the aging flops.

**Kyler:** his own history is the cleanest comp — with a top-decile WR (Hopkins
2020-22) he averaged **24.9 league PPG** vs 20.4 without, and Jefferson is a better
WR than any he's ever had. Per-game upside is QB5-8 in a 6-pt league. The real risk
is games (5 in 2025, 8 in 2023, 11 in 2022): at 13-15 games x 23-25 PPG he lands
**300-360**, right on FFA's 339. Our board's 165 is stale — it still has him on ARI
with 10.1 games and no Jefferson.

**Bottom line:** the market prices (JJ AAV $46 / ADP 19.5; Kyler AAV $7.7 / ADP 115)
both look cheap relative to this analysis. Jefferson is a strong buy at market —
his down 2025 was a QB artifact, not decline. Kyler at a 6th-round ADP is one of
the best QB values on the board in this scoring: pay the injury discount, pair
him with a durable $1-4 QB, and you get top-5-QB weeks at a backup price.""")

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "kyler_jj_2026.md"), "w", encoding="utf-8").write("\n".join(L))
print("\nWrote outputs/reports/kyler_jj_2026.md")
con.close()
