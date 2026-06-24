"""
2026 draft-day targets report. Reads board_2026 (veterans + rookies, with
breakout/hit probabilities + certainty inputs), computes custom-scored projection,
league VORP, base auction $, and a certainty score, then categorizes targets:
  ANCHORS (safe, high floor) | BREAKOUTS (YoY-primed) | SLEEPERS (cheap upside)
  | ROOKIES (immediate impact) | FADES (priced up / risky)
Writes outputs/models/DRAFT_2026_TARGETS.md.

Caveat: no 2026 ECR/FFA yet, so projections are prior-year-anchored model guesses.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"db","nfl_odds.db")
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"outputs","models")
TEAMS=12; BUDGET=200; ROSTER=16
BASE={"QB":TEAMS,"RB":TEAMS*2,"WR":TEAMS*2,"TE":TEAMS}; FLEX=["RB","WR","TE"]; FLEX_N=TEAMS


def main():
    con=sqlite3.connect(DB)
    b=pd.read_sql("SELECT * FROM board_2026",con)
    s25=pd.read_sql("SELECT player_id, games, passing_tds, passing_interceptions FROM nflv_season WHERE season=2025",con).drop_duplicates("player_id")
    con.close()
    b=b.merge(s25,on="player_id",how="left")

    # custom scoring: RB/WR/TE == PPR; QB adjust to 6-pt pass TD / -1 INT
    b["proj_pts"]=b["pred_ppg"]*b["proj_games"]
    qb=b.position=="QB"
    rtd=(b.passing_tds/b.games.clip(lower=1)); rint=(b.passing_interceptions/b.games.clip(lower=1))
    rtd=rtd.fillna(rtd[qb].median()); rint=rint.fillna(rint[qb].median())
    b.loc[qb,"proj_pts"]=b.loc[qb,"proj_pts"]+2*(rtd[qb]*b.loc[qb,"proj_games"])+1*(rint[qb]*b.loc[qb,"proj_games"])
    b=b[b.proj_pts.notna()].copy()
    b["pos_rank"]=b.groupby("position")["proj_pts"].rank(ascending=False,method="min").astype(int)

    # VORP with flex
    fp=b[(b.position.isin(FLEX))&(b.apply(lambda r:r.pos_rank>BASE[r.position],axis=1))]
    flex=fp.sort_values("proj_pts",ascending=False).head(FLEX_N)
    fc=flex.position.value_counts().to_dict()
    starters={p:BASE[p]+(fc.get(p,0) if p in FLEX else 0) for p in BASE}
    repl={}
    for p in BASE:
        pp=b[b.position==p].sort_values("proj_pts",ascending=False)["proj_pts"].values
        repl[p]=float(pp[starters[p]]) if starters[p]<len(pp) else float(pp[-1])
    b["vorp"]=b["proj_pts"]-b["position"].map(repl)

    # base auction $ (skill pool; K/DST reserve ~$24)
    draftable=b.sort_values("vorp",ascending=False).head(TEAMS*ROSTER-TEAMS*2)
    sumv=draftable["vorp"].clip(lower=0).sum()
    disc=(TEAMS*BUDGET-24)-len(draftable)
    per=disc/sumv if sumv>0 else 0
    b["auction"]=(1+b["vorp"].clip(lower=0)*per).round().astype(int)
    b.loc[~b.index.isin(draftable.index),"auction"]=b["vorp"].apply(lambda v:1 if v>-99 else 1)
    b=b.sort_values("vorp",ascending=False).reset_index(drop=True)
    b["ovr"]=np.arange(1,len(b)+1)

    # certainty: low prior CV + durable games (veterans). rookies inherently uncertain.
    b["certainty"]=np.nan
    vmask=b.is_rookie==0
    cv=b.loc[vmask,"prior_cv"]; gp=b.loc[vmask,"prior_games"]
    b.loc[vmask,"certainty"]=(0.6*(1-(cv-cv.min())/(cv.max()-cv.min()))+0.4*(gp/17).clip(0,1))

    con=sqlite3.connect(DB); b.to_sql("draft_board_2026",con,if_exists="replace",index=False); con.close()

    def fmt(d,cols):
        return d[cols].to_string(index=False)

    L=[]
    L.append("# 2026 Draft-Day Targets — Deep Dive\n")
    L.append("> **Big caveat:** 2026 ECR/ADP/FFA isn't published yet, so these are **model projections** "
             "(prior-year-anchored, no market input) — a reasoned guess, not gospel. Keepers unknown, so the full "
             "pool incl. all rookies is treated as available. Cost = base auction $ (12-team, $200, 16 roster, "
             "K/DST ≈ $1). Scoring = your custom league (RB/WR/TE = PPR; QB 6-pt pass TD / −1 INT).\n")
    L.append("Built from `board_2026`: 2026 model projection + **breakout probability** (P of ≥+4 PPG jump vs 2025) "
             "for veterans, **rookie-hit probability** for the 2026 class, and a **certainty** score "
             "(prior consistency + durability).\n")

    # ANCHORS: top VORP, high certainty, durable
    anc=b[(b.is_rookie==0)&(b.certainty>=b["certainty"].quantile(.6))&(b.prior_games>=13)].sort_values("vorp",ascending=False).head(14)
    L.append("\n## 🟢 Anchors — safe, high-floor early picks")
    L.append("High projected value **and** high production certainty (low week-to-week volatility, durable). Build around these.\n")
    L.append("| Ovr | Pos | Player | Team | Proj pts | $ | Certainty |")
    L.append("|---|---|---|---|---|---|---|")
    for _,r in anc.iterrows():
        L.append(f"| {r.ovr} | {r.position} | {r.player_display_name} | {r.team} | {r.proj_pts:.0f} | ${r.auction} | {r.certainty:.2f} |")

    # BREAKOUTS: ascending young, high breakout_prob
    brk=b[(b.is_rookie==0)&(b.age<=26)&(b.pred_ppg>=8)].sort_values("breakout_prob",ascending=False).head(14)
    L.append("\n## 🚀 Breakout / explosion-primed (YoY)")
    L.append("Young, ascending, with the breakout profile (room to grow + pedigree + positive regression). Highest upside relative to last year.\n")
    L.append("| Pos | Player | Team | Age | 2025 PPG | Proj PPG | Breakout P | $ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _,r in brk.iterrows():
        L.append(f"| {r.position} | {r.player_display_name} | {r.team} | {r.age:.0f} | {r.prior_ppg:.1f} | {r.pred_ppg:.1f} | {r.breakout_prob:.0%} | ${r.auction} |")

    # SLEEPERS: cheap (auction<=8) but real upside (breakout_prob high or vorp>0)
    slp=b[(b.is_rookie==0)&(b.auction<=8)&(b.pred_ppg>=6)&(b.breakout_prob>=0.45)&(b.age<=27)].sort_values("breakout_prob",ascending=False).head(14)
    L.append("\n## 💤 Sleepers — cheap shots with upside")
    L.append("Low projected cost (≤ $8) but a meaningful breakout chance — the late-round dart throws that win leagues.\n")
    L.append("| Pos | Player | Team | Age | Proj PPG | Breakout P | $ |")
    L.append("|---|---|---|---|---|---|---|")
    for _,r in slp.iterrows():
        L.append(f"| {r.position} | {r.player_display_name} | {r.team} | {r.age:.0f} | {r.pred_ppg:.1f} | {r.breakout_prob:.0%} | ${r.auction} |")

    # ROOKIES: high hit_prob
    rk=b[b.is_rookie==1].sort_values("hit_prob",ascending=False).head(14)
    L.append("\n## 🌟 Rookies — immediate-impact (out of the gate)")
    L.append("2026 class ranked by year-1 hit probability (draft capital + landing spot; no combine yet). These are the rookies most likely to produce *now*.\n")
    L.append("| Pos | Player | Team | Proj PPG | Hit P | $ |")
    L.append("|---|---|---|---|---|---|")
    for _,r in rk.iterrows():
        L.append(f"| {r.position} | {r.player_display_name} | {r.team} | {r.pred_ppg:.1f} | {r.hit_prob:.0%} | ${r.auction} |")

    # FADES: expensive (auction>=20) but low certainty / decline risk
    fade=b[(b.is_rookie==0)&(b.auction>=15)].sort_values("certainty").head(10)
    L.append("\n## ⚠️ Caution — priced up but lower certainty")
    L.append("Expensive (≥ $15) players with the **lowest certainty** in that price range (week-to-week volatility, age, or durability risk). Not auto-fades, but don't pay the premium blindly.\n")
    L.append("| Pos | Player | Team | Age | Proj pts | $ | Certainty |")
    L.append("|---|---|---|---|---|---|---|")
    for _,r in fade.iterrows():
        L.append(f"| {r.position} | {r.player_display_name} | {r.team} | {r.age:.0f} | {r.proj_pts:.0f} | ${r.auction} | {r.certainty:.2f} |")

    L.append("\n---\n*Scripts: `build_2026_targets.py` → `board_2026`; `report_2026_targets.py` → `draft_board_2026`. "
             "Re-run once 2026 FFA/ADP publish for market-anchored numbers.*")
    open(os.path.join(OUT,"DRAFT_2026_TARGETS.md"),"w",encoding="utf-8").write("\n".join(L))
    print("Wrote DRAFT_2026_TARGETS.md")
    print(f"Replacement $: {repl}")
    print("\nTop 10 overall by VORP:")
    print(b.head(10)[["ovr","position","player_display_name","team","proj_pts","auction","is_rookie"]].to_string(index=False))


if __name__ == "__main__":
    main()
