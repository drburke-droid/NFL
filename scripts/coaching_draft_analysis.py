"""
Deep dive: do team draft-capital allocation and head-coach changes correlate with
on-field outcomes (2011-2025)?

Hypotheses tested:
  H1  DL draft capital  -> defensive sacks / DST points
  H2  OL draft capital  -> pass protection (fewer sacks allowed) + pass EPA (QB play)
  H3  QB draft capital  -> pass EPA
  H4  Total draft capital -> wins / point differential
  H5  Head-coach change  -> year-1 change in offense/defense/wins

Draft from nflv_draft; outcomes from nflv_weekly + nflv_team_def; HC + scores from
nflverse schedules. (OC/DC are not available in nflverse — handled separately.)
"""
import os, sqlite3, math, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import nflreadpy as nflr
from scipy.stats import spearmanr

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
os.makedirs(OUT, exist_ok=True)
YEARS = range(2011, 2026)

GROUP = {  # position -> group
  "QB":"QB","RB":"RB","FB":"RB","WR":"WR","TE":"TE",
  "T":"OL","OT":"OL","G":"OL","OG":"OL","C":"OL","OL":"OL",
  "DE":"DL","DT":"DL","NT":"DL","DL":"DL",
  "OLB":"EDGE/LB","ILB":"LB","MLB":"LB","LB":"LB",
  "CB":"DB","S":"DB","FS":"DB","SAF":"DB","DB":"DB",
  "K":"ST","P":"ST","LS":"ST","KR":"ST",
}
ABBR = {"OAK":"LV","SD":"LAC","SDG":"LAC","STL":"LA","LAR":"LA","JAC":"JAX"}


def pick_value(pick):
    """Smooth approximation of the Jimmy Johnson draft-value curve."""
    if pd.isna(pick) or pick > 262: return 0.0
    return 1000.0 * math.exp(-(pick-1)/40.0)


def build_panel():
    con = sqlite3.connect(DB)
    draft = pd.read_sql("SELECT season, pick, team, position FROM nflv_draft WHERE season>=2008", con)
    wk = pd.read_sql("""SELECT team, season, passing_epa, rushing_epa, sacks_suffered, passing_yards
                        FROM nflv_weekly WHERE season_type='REG' AND season>=2011""", con)
    tdef = pd.read_sql("SELECT team, season, sacks, custom_pts FROM nflv_team_def WHERE season>=2011", con)
    con.close()

    draft["team"] = draft["team"].replace(ABBR)
    draft["grp"] = draft["position"].map(GROUP).fillna("OTHER")
    draft["val"] = draft["pick"].map(pick_value)
    cap = draft.groupby(["team","season","grp"])["val"].sum().unstack(fill_value=0).reset_index()
    cap.columns = ["team","season"] + [f"cap_{c}" for c in cap.columns[2:]]
    capcols = [c for c in cap.columns if c.startswith("cap_")]
    # rolling 3-yr cumulative draft capital (this + prior 2 drafts)
    cap = cap.sort_values(["team","season"])
    roll = cap.groupby("team")[capcols].apply(lambda g: g.rolling(3, min_periods=1).sum()).reset_index(drop=True)
    roll.columns = [c.replace("cap_","r3_") for c in roll.columns]
    cap = pd.concat([cap, roll], axis=1)

    # team offense outcomes
    off = wk.groupby(["team","season"]).agg(
        pass_epa=("passing_epa","sum"), rush_epa=("rushing_epa","sum"),
        sacks_allowed=("sacks_suffered","sum"), pass_yds=("passing_yards","sum")).reset_index()

    # HC + wins from schedules
    s = nflr.load_schedules().to_pandas()
    s = s[(s.season>=2011)&(s.game_type=="REG")].copy()
    rows=[]
    for _,r in s.iterrows():
        if pd.notna(r["result"]):
            rows.append((r["season"], ABBR.get(r["home_team"],r["home_team"]), r["home_coach"], r["home_score"], r["away_score"], 1 if r["result"]>0 else (0.5 if r["result"]==0 else 0)))
            rows.append((r["season"], ABBR.get(r["away_team"],r["away_team"]), r["away_coach"], r["away_score"], r["home_score"], 1 if r["result"]<0 else (0.5 if r["result"]==0 else 0)))
    g = pd.DataFrame(rows, columns=["season","team","coach","pf","pa","win"])
    rec = g.groupby(["team","season"]).agg(wins=("win","sum"), games=("win","size"),
            pts_for=("pf","sum"), pts_against=("pa","sum"),
            coach=("coach", lambda x: x.mode().iat[0] if len(x.mode()) else None)).reset_index()
    rec["point_diff"] = rec["pts_for"] - rec["pts_against"]
    rec = rec.sort_values(["team","season"])
    rec["prev_coach"] = rec.groupby("team")["coach"].shift(1)
    rec["hc_change"] = ((rec["coach"]!=rec["prev_coach"]) & rec["prev_coach"].notna()).astype(int)

    panel = (rec.merge(off, on=["team","season"], how="left")
                .merge(tdef, on=["team","season"], how="left")
                .merge(cap, on=["team","season"], how="left"))
    return panel, capcols


def main():
    panel, capcols = build_panel()
    panel = panel[panel.season.between(2011,2025)].copy()
    print(f"Panel: {len(panel)} team-seasons, {panel.season.min()}-{panel.season.max()}")
    r3 = [c.replace("cap_","r3_") for c in capcols]

    def corr(xcol, ycol, lag=0, label=""):
        d = panel.copy()
        if lag:
            d["_y"] = d.groupby("team")[ycol].shift(-lag)  # outcome `lag` years later
            ycol = "_y"
        d = d[[xcol, ycol]].dropna()
        if len(d) < 30: return None
        rho, p = spearmanr(d[xcol], d[ycol])
        return (rho, p, len(d))

    print("\n=== H1: DL draft capital (rolling 3yr) -> defense ===")
    for lag in [0,1,2]:
        for y in ["sacks","custom_pts"]:
            r = corr("r3_DL", y, lag)
            if r: print(f"  r3_DL -> {y:11s} (+{lag}yr): rho {r[0]:+.3f} p={r[1]:.3f} n={r[2]}")

    print("\n=== H2: OL draft capital (3yr) -> pass protection & QB play ===")
    for lag in [0,1,2]:
        for y in ["sacks_allowed","pass_epa"]:
            r = corr("r3_OL", y, lag)
            if r: print(f"  r3_OL -> {y:13s} (+{lag}yr): rho {r[0]:+.3f} p={r[1]:.3f} n={r[2]}")

    print("\n=== H3: QB draft capital (3yr) -> pass EPA ===")
    for lag in [0,1,2]:
        r = corr("r3_QB", "pass_epa", lag)
        if r: print(f"  r3_QB -> pass_epa (+{lag}yr): rho {r[0]:+.3f} p={r[1]:.3f} n={r[2]}")

    print("\n=== H4: total draft capital (3yr) -> wins / point diff ===")
    panel["r3_total"] = panel[r3].sum(axis=1)
    for lag in [0,1,2]:
        for y in ["wins","point_diff"]:
            r = corr("r3_total", y, lag)
            if r: print(f"  r3_total -> {y:10s} (+{lag}yr): rho {r[0]:+.3f} p={r[1]:.3f} n={r[2]}")
    # also EDGE/DB capital -> defense
    print("\n  (bonus) EDGE/LB & DB capital -> def sacks / DST:")
    for x in ["r3_EDGE/LB","r3_DB"]:
        if x in panel:
            for y in ["sacks","custom_pts"]:
                r=corr(x,y,1)
                if r: print(f"    {x} -> {y} (+1yr): rho {r[0]:+.3f} p={r[1]:.3f} n={r[2]}")

    print("\n=== H5: head-coach change -> year-1 change vs prior year ===")
    p = panel.sort_values(["team","season"]).copy()
    for m in ["pass_epa","rush_epa","custom_pts","point_diff","wins"]:
        p[f"d_{m}"] = p[m] - p.groupby("team")[m].shift(1)
    chg = p[p.hc_change==1]; same = p[(p.hc_change==0)&(p.prev_coach.notna())]
    print(f"  new-HC seasons n={len(chg)}, continuity seasons n={len(same)}")
    print(f"  {'metric':12s} {'newHC dY':>9s} {'cont dY':>9s} {'diff':>7s}")
    for m in ["pass_epa","rush_epa","custom_pts","point_diff","wins"]:
        a=chg[f"d_{m}"].mean(); b=same[f"d_{m}"].mean()
        print(f"  {m:12s} {a:9.2f} {b:9.2f} {a-b:+7.2f}")

    panel.to_sql("team_draft_panel", sqlite3.connect(DB), if_exists="replace", index=False)
    print("\nSaved team_draft_panel.")


if __name__ == "__main__":
    main()
