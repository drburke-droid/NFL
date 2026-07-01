"""Historical fantasy trend sweep 2011-2024 (weekly fantasy_points_ppr) for actionable draft/keeper insight.
Each trend labelled: EDGE / DRAFT-STRATEGY / ALREADY-PRICED. 2025 usage from pbp where noted."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nfl_data_py as nfl
YRS=list(range(2011,2025))
w=nfl.import_weekly_data(YRS); w=w[w.season_type=="REG"]
POS=["QB","RB","WR","TE"]
s=w[w.position.isin(POS)].groupby(["player_id","season","position"]).agg(
    g=("fantasy_points_ppr","size"),fp=("fantasy_points_ppr","sum"),
    rush=("rushing_yards","sum"),rushtd=("rushing_tds","sum")).reset_index()
s["ppg"]=s.fp/s.g
pl=nfl.import_players()[["gsis_id","birth_date","draft_year"]]
dob=dict(zip(pl.gsis_id,pd.to_datetime(pl.birth_date,errors="coerce"))); dy=dict(zip(pl.gsis_id,pl.draft_year))
s["age"]=[ (pd.Timestamp(f"{yr}-09-01")-dob[p]).days/365.25 if p in dob and pd.notna(dob[p]) else np.nan for p,yr in zip(s.player_id,s.season)]
s["rook"]=[1 if dy.get(p)==yr else 0 for p,yr in zip(s.player_id,s.season)]
nxt={(r.player_id,r.season):r.ppg for r in s.itertuples()}
s["next"]=[nxt.get((p,yr+1),np.nan) for p,yr in zip(s.player_id,s.season)]

print("="*70+"\n1) YEAR-OVER-YEAR REPEATABILITY BY POSITION (keeper safety)")
print("   corr(PPG_Y, PPG_Y+1) for >=8-game seasons; higher = safer keeper")
for p in POS:
    d=s[(s.position==p)&(s.g>=8)].dropna(subset=["next"])
    de=d[d.season<=2017]; dl=d[d.season>=2018]
    print(f"   {p}: all r={np.corrcoef(d.ppg,d.next)[0,1]:.2f} (n={len(d)}) | 2011-17 {np.corrcoef(de.ppg,de.next)[0,1]:.2f} | 2018-24 {np.corrcoef(dl.ppg,dl.next)[0,1]:.2f}")
# top-12 repeat rate
print("   Top-12 finish repeat rate (top-12 in Y -> top-12 in Y+1):")
for p in POS:
    reps=[]
    for yr in range(2011,2024):
        a=s[(s.position==p)&(s.season==yr)&(s.g>=8)].nlargest(12,"fp")
        b=set(s[(s.position==p)&(s.season==yr+1)&(s.g>=8)].nlargest(12,"fp").player_id)
        if len(a): reps.append(np.mean([pid in b for pid in a.player_id]))
    print(f"     {p}: {np.mean(reps)*100:.0f}%")

print("="*70+"\n2) POSITIONAL SCARCITY — PPG of the tier your league starts (early vs recent)")
for p,ranks in [("QB",[6,12]),("RB",[12,24,36]),("WR",[12,24,36]),("TE",[6,12])]:
    row=f"   {p}: "
    for rk in ranks:
        e=s[(s.position==p)&(s.season.between(2011,2017))&(s.g>=8)].groupby("season").apply(lambda x:x.nlargest(rk,"fp").ppg.min()).mean()
        l=s[(s.position==p)&(s.season.between(2018,2024))&(s.g>=8)].groupby("season").apply(lambda x:x.nlargest(rk,"fp").ppg.min()).mean()
        row+=f"#{rk} {e:.1f}->{l:.1f}  "
    print(row)

print("="*70+"\n3) AGE CURVES — mean next-yr PPG DELTA by age (decline timing)")
for p in POS:
    d=s[(s.position==p)&(s.g>=8)].dropna(subset=["next","age"]).copy(); d["chg"]=d.next-d.ppg
    buckets=[(21,24),(24,26),(26,28),(28,30),(30,33)]
    print(f"   {p}: "+"  ".join(f"{a}-{b}:{d[(d.age>=a)&(d.age<b)].chg.mean():+.1f}" for a,b in buckets))

print("="*70+"\n4) QB RUSHING EDGE — rushing share of top-12 QB fantasy pts")
for era in [(2011,2015),(2016,2020),(2021,2024)]:
    dd=s[(s.position=="QB")&(s.season.between(*era))&(s.g>=8)]
    top=dd.groupby("season",group_keys=False).apply(lambda x:x.nlargest(12,"fp"))
    rushpts=(top.rush*0.1+top.rushtd*6); sh=(rushpts.sum()/top.fp.sum())*100
    print(f"   {era[0]}-{era[1]}: {sh:.0f}% of QB1 points from rushing")

print("="*70+"\n5) ROOKIE WR EARLY IMPACT — rookie WR PPG by era (>=8 g)")
for era in [(2011,2015),(2016,2020),(2021,2024)]:
    d=s[(s.position=="WR")&(s.rook==1)&(s.season.between(*era))&(s.g>=8)]
    top=s[(s.position=="WR")&(s.rook==1)&(s.season.between(*era))&(s.g>=8)]
    print(f"   {era[0]}-{era[1]}: mean {d.ppg.mean():.1f} PPG (n={len(d)}), # rookie WR >=12 PPG/yr: {(d.ppg>=12).sum()/(era[1]-era[0]+1):.1f}")

print("="*70+"\n6) WR CONCENTRATION ('death of the WR2') — share of WR fantasy pts")
for era in [(2011,2015),(2016,2020),(2021,2024)]:
    dd=s[(s.position=="WR")&(s.season.between(*era))&(s.g>=8)]
    def sh(lo,hi):
        v=[]
        for yr in range(era[0],era[1]+1):
            x=s[(s.position=="WR")&(s.season==yr)&(s.g>=8)].nlargest(hi,"fp"); tot=s[(s.position=="WR")&(s.season==yr)].fp.sum()
            v.append(x.iloc[lo:hi].fp.sum()/tot)
        return np.mean(v)*100
    print(f"   {era[0]}-{era[1]}: top1-12 {sh(0,12):.0f}%  WR13-24 {sh(12,24):.0f}%  WR25-48 {sh(24,48):.0f}%  of all WR pts")

print("="*70+"\n7) TE TARGET SHARE — top-tier TE dominance of TE production")
for era in [(2011,2015),(2016,2020),(2021,2024)]:
    v6,v12=[],[]
    for yr in range(era[0],era[1]+1):
        x=s[(s.position=="TE")&(s.season==yr)&(s.g>=8)]; tot=s[(s.position=="TE")&(s.season==yr)].fp.sum()
        if len(x): v6.append(x.nlargest(6,"fp").fp.sum()/tot); v12.append(x.nlargest(12,"fp").fp.sum()/tot)
    print(f"   {era[0]}-{era[1]}: top-6 TE = {np.mean(v6)*100:.0f}% of TE pts, top-12 = {np.mean(v12)*100:.0f}%")

print("="*70+"\n8) WEEKLY CONSISTENCY (floor/ceiling) — median week-to-week CV by position")
ww=w[w.position.isin(POS)].copy()
cv=ww.groupby(["player_id","season","position"]).fantasy_points_ppr.agg(["mean","std","size"]).reset_index()
cv=cv[(cv["size"]>=10)&(cv["mean"]>=8)]  # relevant starters
for p in POS:
    d=cv[cv.position==p]; print(f"   {p}: median CV {(d['std']/d['mean']).median():.2f}  (lower = steadier week-to-week floor)")
