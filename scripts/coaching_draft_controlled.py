"""
Controlled follow-up to coaching_draft_analysis.py.

The raw correlations are dominated by two confounds:
  - Draft capital: BAD teams pick higher and more often, so capital correlates
    NEGATIVELY with outcomes by construction. Must control for team quality.
  - HC change: teams fire coaches after bad years, so a "new-HC bump" is partly
    mean reversion. Must compare to equally-bad teams that kept their coach.

This script re-tests the hypotheses controlling for prior team quality
(point differential), and stratifies the HC bump by prior record.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")


def partial_spearman(df, x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    d = df[[x, y, z]].dropna()
    if len(d) < 30: return None
    rxy = spearmanr(d[x], d[y])[0]; rxz = spearmanr(d[x], d[z])[0]; ryz = spearmanr(d[y], d[z])[0]
    denom = np.sqrt((1-rxz**2)*(1-ryz**2))
    return (None if denom == 0 else (rxy - rxz*ryz)/denom, len(d))


def main():
    con = sqlite3.connect(DB)
    p = pd.read_sql("SELECT * FROM team_draft_panel", con).sort_values(["team","season"])
    con.close()
    p["prev_pd"] = p.groupby("team")["point_diff"].shift(1)   # prior-year quality (control)
    p["prev_wins"] = p.groupby("team")["wins"].shift(1)

    print("=== Draft capital -> outcome, CONTROLLED for prior team quality (point diff) ===")
    print("   (raw rho | partial rho controlling prior quality)")
    tests = [("r3_DL","sacks","DL capital -> def sacks"),
             ("r3_DL","custom_pts","DL capital -> DST pts"),
             ("r3_OL","sacks_allowed","OL capital -> sacks allowed"),
             ("r3_OL","pass_epa","OL capital -> pass EPA"),
             ("r3_QB","pass_epa","QB capital -> pass EPA"),
             ("r3_total","wins","total capital -> wins")]
    for x,y,lbl in tests:
        d=p[[x,y]].dropna(); raw=spearmanr(d[x],d[y])[0]
        pc=partial_spearman(p,x,y,"prev_pd")
        print(f"  {lbl:32s} raw {raw:+.3f} | partial {pc[0]:+.3f} (n={pc[1]})")

    print("\n=== Draft capital -> NEXT-YEAR IMPROVEMENT (delta vs prior year) ===")
    for m in ["sacks","custom_pts","sacks_allowed","pass_epa"]:
        p[f"nd_{m}"] = p.groupby("team")[m].shift(-1) - p[m]   # change into next year
    for x,y,lbl in [("r3_DL","nd_sacks","DL capital -> Δdef sacks next yr"),
                    ("r3_DL","nd_custom_pts","DL capital -> ΔDST next yr"),
                    ("r3_OL","nd_sacks_allowed","OL capital -> Δsacks allowed next yr"),
                    ("r3_QB","nd_pass_epa","QB capital -> Δpass EPA next yr")]:
        d=p[[x,y]].dropna()
        if len(d)>=30:
            rho,pv=spearmanr(d[x],d[y]); print(f"  {lbl:36s} rho {rho:+.3f} p={pv:.3f} n={len(d)}")

    print("\n=== HC change bump, CONTROLLED for prior record (mean-reversion control) ===")
    for m in ["pass_epa","point_diff","wins","custom_pts"]:
        p[f"d_{m}"] = p[m] - p.groupby("team")[m].shift(1)
    p["winbkt"] = pd.cut(p["prev_wins"], [-1,4,7,10,18], labels=["≤4","5-7","8-10","11+"])
    cohort = p[p.prev_coach.notna()].copy()
    print(f"  {'prior wins':10s} {'grp':8s} {'n':>4s} {'Δwins':>7s} {'Δpt_diff':>9s} {'Δpass_epa':>10s}")
    for bkt in ["≤4","5-7","8-10","11+"]:
        for chg,lbl in [(1,"new HC"),(0,"kept HC")]:
            sub=cohort[(cohort.winbkt==bkt)&(cohort.hc_change==chg)]
            if len(sub)<5: continue
            print(f"  {bkt:10s} {lbl:8s} {len(sub):4d} {sub['d_wins'].mean():7.2f} {sub['d_point_diff'].mean():9.1f} {sub['d_pass_epa'].mean():10.1f}")
    # net coaching effect within each bucket
    print("\n  Net new-HC effect vs kept-HC at same prior record (Δwins):")
    for bkt in ["≤4","5-7","8-10","11+"]:
        a=cohort[(cohort.winbkt==bkt)&(cohort.hc_change==1)]["d_wins"]
        b=cohort[(cohort.winbkt==bkt)&(cohort.hc_change==0)]["d_wins"]
        if len(a)>=5 and len(b)>=5:
            print(f"    prior {bkt:6s}: new HC {a.mean():+.2f} vs kept {b.mean():+.2f}  ->  net {a.mean()-b.mean():+.2f} wins")


if __name__ == "__main__":
    main()
