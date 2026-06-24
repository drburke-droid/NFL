"""
Coordinator (OC/DC) change analysis, 2011-2025.

Coaching staff sourced via research (data/coaches.csv). We first VALIDATE the
researched head coaches against the schedule-derived HC (ground truth) to gauge
data quality, then test:
  C1  New OC -> year-1 offense bump (controlled for mean reversion)
  C2  New DC -> year-1 defense bump (controlled)
  C3  Hiring an OC who ran a GOOD offense elsewhere -> bigger year-1 bump
      (the key hypothesis: does the OC's prior-team offense quality carry over?)
"""
import os, sqlite3, re, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "coaches.csv")


def norm(n):
    return re.sub(r"\s+"," ", re.sub(r"[^a-z ]","", str(n).lower().replace(".",""))).strip()


def main():
    coaches = pd.read_csv(CSV)
    con = sqlite3.connect(DB)
    coaches.to_sql("nflv_coaches", con, if_exists="replace", index=False)
    panel = pd.read_sql("SELECT * FROM team_draft_panel", con)
    con.close()

    # ---- validate researched HC vs schedule HC ----
    v = coaches.merge(panel[["team","season","coach"]], on=["team","season"], how="inner")
    v["match"] = v.apply(lambda r: norm(r["hc"])==norm(r["coach"]), axis=1)
    print(f"=== HC validation vs schedule ground truth: {v['match'].mean():.1%} match ({v['match'].sum()}/{len(v)}) ===")
    mism = v[~v["match"]][["team","season","hc","coach"]]
    print(f"  {len(mism)} mismatches (mostly interim-coach convention differences):")
    print(mism.head(12).to_string(index=False))

    df = panel.merge(coaches, on=["team","season"], how="left").sort_values(["team","season"])
    for role in ["oc","dc","hc"]:
        df[f"prev_{role}"] = df.groupby("team")[role].shift(1)
        df[f"{role}_change"] = ((df[role]!=df[f"prev_{role}"]) & df[f"prev_{role}"].notna()).astype(int)
    for m in ["pass_epa","custom_pts","point_diff","sacks","pts_against","rush_epa"]:
        df[f"d_{m}"] = df[m] - df.groupby("team")[m].shift(1)
    # prior-year offense/defense buckets (mean-reversion control)
    df["off_q"] = df.groupby("season")["pass_epa"].rank(pct=True)
    df["prev_off_q"] = df.groupby("team")["off_q"].shift(1)
    df["prev_off_bkt"] = pd.cut(df["prev_off_q"], [0,.33,.66,1], labels=["bottom","mid","top"])

    print("\n=== C1: New OC -> year-1 offense change (Δ team pass EPA), by prior offense tier ===")
    print(f"  {'prior off':10s} {'grp':8s} {'n':>4s} {'Δpass_epa':>10s} {'Δpoints':>9s}")
    coh = df[df.prev_oc.notna()].copy()
    for bkt in ["bottom","mid","top"]:
        for chg,lbl in [(1,"new OC"),(0,"kept OC")]:
            s=coh[(coh.prev_off_bkt==bkt)&(coh.oc_change==chg)]
            if len(s)<5: continue
            print(f"  {bkt:10s} {lbl:8s} {len(s):4d} {s['d_pass_epa'].mean():10.1f} {s['d_point_diff'].mean():9.1f}")
    print("  Net new-OC effect (Δpass_epa, new − kept) within prior-offense tier:")
    for bkt in ["bottom","mid","top"]:
        a=coh[(coh.prev_off_bkt==bkt)&(coh.oc_change==1)]["d_pass_epa"]; b=coh[(coh.prev_off_bkt==bkt)&(coh.oc_change==0)]["d_pass_epa"]
        if len(a)>=5 and len(b)>=5: print(f"    prior {bkt:6s}: {a.mean()-b.mean():+.1f} pass-EPA")

    print("\n=== C2: New DC -> year-1 defense change, by prior defense tier ===")
    df["def_q"] = df.groupby("season")["custom_pts"].rank(pct=True)
    df["prev_def_q"]=df.groupby("team")["def_q"].shift(1)
    df["prev_def_bkt"]=pd.cut(df["prev_def_q"],[0,.33,.66,1],labels=["bottom","mid","top"])
    cohd=df[df.prev_dc.notna()].copy()
    for bkt in ["bottom","mid","top"]:
        a=cohd[(cohd.prev_def_bkt==bkt)&(cohd.dc_change==1)]["d_custom_pts"]; b=cohd[(cohd.prev_def_bkt==bkt)&(cohd.dc_change==0)]["d_custom_pts"]
        if len(a)>=5 and len(b)>=5: print(f"  prior {bkt:6s}: new DC ΔDST {a.mean():+.1f} vs kept {b.mean():+.1f} -> net {a.mean()-b.mean():+.1f}")

    print("\n=== C3: OC hired FROM a good offense -> does the bump carry over? ===")
    # build oc -> (season -> team) to find prior-year team where this OC was OC
    oc_loc = {}
    for _,r in coaches.iterrows():
        oc_loc.setdefault(r["oc"], {})[r["season"]] = r["team"]
    rows=[]
    for _,r in df.iterrows():
        oc, tm, yr = r["oc"], r["team"], r["season"]
        if pd.isna(oc): continue
        prev_team = oc_loc.get(oc, {}).get(yr-1)
        if prev_team and prev_team != tm:          # OC moved from another NFL team
            prev = df[(df.team==prev_team)&(df.season==yr-1)]
            if len(prev):
                rows.append({"team":tm,"season":yr,"oc":oc,"from":prev_team,
                             "from_off_q": float(prev["off_q"].iloc[0]),
                             "bump": float(r["d_pass_epa"]) if pd.notna(r["d_pass_epa"]) else np.nan})
    moves = pd.DataFrame(rows).dropna(subset=["bump","from_off_q"])
    print(f"  OC moves between NFL teams (was OC elsewhere prior year): n={len(moves)}")
    if len(moves):
        good = moves[moves.from_off_q>=0.6]; bad = moves[moves.from_off_q<=0.4]
        print(f"  OC came from TOP-tier offense (n={len(good)}): new team Δpass_epa = {good['bump'].mean():+.1f}")
        print(f"  OC came from BOTTOM-tier offense (n={len(bad)}): new team Δpass_epa = {bad['bump'].mean():+.1f}")
        rho,p = spearmanr(moves.from_off_q, moves.bump)
        print(f"  corr(prior-team offense quality, new-team offense bump): rho {rho:+.3f} p={p:.3f}")
        print("\n  Notable OC moves (from good offenses):")
        print(good.sort_values("from_off_q",ascending=False).head(8)[["season","oc","from","team","from_off_q","bump"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
