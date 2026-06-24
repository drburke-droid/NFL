"""
Multi-year draft-capital test: does SUSTAINED investment in a position group over
a 5-year window predict that unit's performance in the FOLLOWING 3 seasons
(i.e. "drafted DL high over years 1-5 -> better DST in years 6-8")?

Two capital measures per 5-yr window ending in year Y (covers Y-4..Y):
  - cap5      : absolute pick-value capital (confounded: bad teams pick higher)
  - share5    : capital share of the position group (DEVOTION; controls for volume)
Outcome = mean over the next 3 seasons (Y+1..Y+3). Spearman + partial (control
for the team's avg quality during the capital window) + top-vs-bottom devotion.
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
GROUPS = ["QB","RB","WR","TE","OL","DL","EDGE/LB","LB","DB"]


def main():
    con = sqlite3.connect(DB)
    p = pd.read_sql("SELECT * FROM team_draft_panel", con).sort_values(["team","season"])
    con.close()
    capcols = [f"cap_{g}" for g in GROUPS if f"cap_{g}" in p.columns]

    # 5-year cumulative capital (require full 5 years)
    for c in capcols:
        p[c.replace("cap_","c5_")] = p.groupby("team")[c].transform(lambda s: s.rolling(5, min_periods=5).sum())
    c5cols = [c.replace("cap_","c5_") for c in capcols]
    p["c5_total"] = p[c5cols].sum(axis=1)
    p["c5_passrush"] = p.get("c5_DL",0) + p.get("c5_EDGE/LB",0)
    # devotion shares
    for c in c5cols + ["c5_passrush"]:
        p[c.replace("c5_","sh_")] = p[c] / p["c5_total"]
    # avg team quality during the capital window (control)
    p["winq"] = p.groupby("team")["point_diff"].transform(lambda s: s.rolling(5, min_periods=5).mean())

    # forward 3-year average outcomes (Y+1..Y+3)
    def fwd3(col):
        g = p.groupby("team")[col]
        return (g.shift(-1) + g.shift(-2) + g.shift(-3)) / 3
    for m in ["custom_pts","sacks","pass_epa","sacks_allowed","wins","point_diff","pts_against"]:
        p[f"f3_{m}"] = fwd3(m)

    def part(x, y, z="winq"):
        d = p[[x, y, z]].dropna()
        if len(d) < 25: return None
        rxy=spearmanr(d[x],d[y])[0]; rxz=spearmanr(d[x],d[z])[0]; ryz=spearmanr(d[y],d[z])[0]
        den=np.sqrt((1-rxz**2)*(1-ryz**2))
        return rxy, (rxy-rxz*ryz)/den if den else np.nan, len(d)

    print("=== Sustained 5-yr capital -> next-3-year unit outcome (raw rho | partial controlling team quality) ===")
    tests = [
        ("DL share","sh_DL","f3_custom_pts","DST pts"),
        ("DL share","sh_DL","f3_sacks","def sacks"),
        ("Pass-rush share (DL+EDGE)","sh_passrush","f3_sacks","def sacks"),
        ("Pass-rush share (DL+EDGE)","sh_passrush","f3_custom_pts","DST pts"),
        ("OL share","sh_OL","f3_sacks_allowed","sacks allowed"),
        ("OL share","sh_OL","f3_pass_epa","pass EPA"),
        ("QB share","sh_QB","f3_pass_epa","pass EPA"),
        ("DB share","sh_DB","f3_pts_against","pts allowed"),
    ]
    for lbl,x,y,yl in tests:
        r=part(x,y)
        if r: print(f"  {lbl:26s} -> {yl:14s} raw {r[0]:+.3f} | partial {r[1]:+.3f} (n={r[2]})")

    print("\n  (absolute capital, for comparison — confounded by 'bad teams pick higher')")
    for lbl,x,y,yl in [("DL capital(5yr)","c5_DL","f3_custom_pts","DST"),
                       ("Total capital(5yr)","c5_total","f3_wins","wins")]:
        r=part(x,y)
        if r: print(f"  {lbl:26s} -> {yl:14s} raw {r[0]:+.3f} | partial {r[1]:+.3f} (n={r[2]})")

    print("\n=== Top-quartile vs bottom-quartile DEVOTION (5-yr share) -> next-3yr outcome ===")
    for lbl, x, y, yl in [("DL share","sh_DL","f3_custom_pts","DST pts"),
                          ("Pass-rush share","sh_passrush","f3_sacks","def sacks"),
                          ("OL share","sh_OL","f3_pass_epa","pass EPA"),
                          ("QB share","sh_QB","f3_pass_epa","pass EPA")]:
        d=p[[x,y]].dropna()
        if len(d)<40: continue
        hi=d[d[x]>=d[x].quantile(.75)][y]; lo=d[d[x]<=d[x].quantile(.25)][y]
        print(f"  {lbl:16s}: top-25% devotion -> {yl} {hi.mean():7.1f}  vs bottom-25% {lo.mean():7.1f}  (diff {hi.mean()-lo.mean():+.1f})")

    print("\n=== Non-overlapping windows (cleaner, less autocorrelation) ===")
    for cap_end, lbl in [(2015,"2011-2015 -> 2016-2018"), (2020,"2016-2020 -> 2021-2023")]:
        w=p[p.season==cap_end]
        for x,y,yl in [("sh_DL","f3_custom_pts","DST"),("sh_passrush","f3_sacks","def sacks"),("sh_QB","f3_pass_epa","passEPA")]:
            d=w[[x,y]].dropna()
            if len(d)>=20:
                rho=spearmanr(d[x],d[y])[0]; print(f"  {lbl} | {x:14s}->{yl:9s} rho {rho:+.3f} (n={len(d)})")


if __name__ == "__main__":
    main()
