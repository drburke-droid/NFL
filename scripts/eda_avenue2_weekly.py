"""
Avenue 2 EDA: deepen what explains WEEKLY DFS performance.

Complements the existing walk-forward model with four analyses:
  A. Volume vs efficiency: which is stickier week to week (where the signal is).
  B. Expected fantasy points (xFP) as a forward signal vs raw points, and whether
     actual-minus-expected ("luck") persists.
  C. Prop-market calibration: how good are book lines, and is there residual signal.
  D. Existing-model residual diagnostics: where dfs_predictions systematically misses.

Outputs CSVs + charts to outputs/eda/, prints summary.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "eda")
os.makedirs(OUT, exist_ok=True)


def autocorr_lag1(df, value, key=("player_id","season")):
    """Mean lag-1 within-group correlation of a weekly value."""
    df = df.sort_values(list(key)+["week"])
    df["lag"] = df.groupby(list(key))[value].shift(1)
    s = df[[value,"lag"]].dropna()
    return spearmanr(s["lag"], s[value])[0], len(s)


def main():
    con = sqlite3.connect(DB)
    wk = pd.read_sql("""SELECT player_id,position,season,week,season_type,
                        targets,carries,receptions,receiving_yards,rushing_yards,
                        fantasy_points_ppr FROM nflv_weekly
                        WHERE season_type='REG' AND season>=2015""", con)
    ffo = pd.read_sql("""SELECT CAST(season AS INT) season, week, player_id, position,
                         total_fantasy_points, total_fantasy_points_exp
                         FROM nflv_ff_opp""", con)

    # ---------- A. Volume vs efficiency stickiness ----------
    print("=== A. Week-to-week stickiness (lag-1 Spearman) ===")
    A = {}
    skill = wk[wk["position"].isin(["RB","WR","TE"])].copy()
    skill["yds_per_opp"] = (skill["receiving_yards"]+skill["rushing_yards"]) / \
                           (skill["targets"]+skill["carries"]).replace(0,np.nan)
    skill["opportunities"] = skill["targets"]+skill["carries"]
    for name,val in [("opportunities (vol)","opportunities"),
                     ("targets","targets"),
                     ("yards/opportunity (eff)","yds_per_opp"),
                     ("fantasy_points_ppr","fantasy_points_ppr")]:
        r,n = autocorr_lag1(skill.dropna(subset=[val]), val)
        A[name]=round(r,3); print(f"  {name:28s} lag1 rho={r:.3f}  (n={n:,})")
    json.dump(A, open(os.path.join(OUT,"a2_stickiness.json"),"w"), indent=2)

    # ---------- B. xFP forward signal (split-half) ----------
    print("\n=== B. First-half -> second-half forward correlation (per-game) ===")
    ffo = ffo[ffo["position"].isin(["QB","RB","WR","TE"])].copy()
    ffo["half"] = np.where(ffo["week"]<=9, "H1", "H2")
    agg = ffo.groupby(["player_id","season","half"]).agg(
        g=("week","size"),
        fp=("total_fantasy_points","sum"),
        xfp=("total_fantasy_points_exp","sum")).reset_index()
    agg["fp_pg"]=agg["fp"]/agg["g"]; agg["xfp_pg"]=agg["xfp"]/agg["g"]
    h1=agg[agg["half"]=="H1"]; h2=agg[agg["half"]=="H2"]
    m=h1.merge(h2, on=["player_id","season"], suffixes=("_h1","_h2"))
    m=m[(m["g_h1"]>=4)&(m["g_h2"]>=4)]
    B={
      "H1 actual -> H2 actual": round(spearmanr(m["fp_pg_h1"], m["fp_pg_h2"])[0],3),
      "H1 EXPECTED -> H2 actual": round(spearmanr(m["xfp_pg_h1"], m["fp_pg_h2"])[0],3),
      "H1 luck(actual-exp) -> H2 actual": round(spearmanr((m["fp_pg_h1"]-m["xfp_pg_h1"]), m["fp_pg_h2"])[0],3),
      "n": int(len(m)),
    }
    for k,v in B.items(): print(f"  {k:40s} {v}")
    json.dump(B, open(os.path.join(OUT,"a2_xfp_forward.json"),"w"), indent=2)

    # ---------- C. Prop-market calibration ----------
    print("\n=== C. Prop-line calibration vs actual (2023-25) ===")
    props = pd.read_sql("""SELECT event_id, market, player_name, AVG(point) point
                           FROM player_props WHERE outcome_type='Over'
                           GROUP BY event_id,market,player_name""", con)
    stats = pd.read_sql("""SELECT event_id, player_display_name player_name, position,
                           receiving_yards, rushing_yards, passing_yards, receptions
                           FROM player_stats""", con)
    MARK={"player_reception_yds":"receiving_yards","player_rush_yds":"rushing_yards",
          "player_pass_yds":"passing_yards","player_receptions":"receptions"}
    Crows=[]
    for mk,act in MARK.items():
        p=props[props["market"]==mk].merge(stats[["event_id","player_name",act]],
                                           on=["event_id","player_name"], how="inner").dropna()
        if len(p)<100: continue
        over=(p[act]>p["point"]).mean()
        mae=(p[act]-p["point"]).abs().mean()
        rho=spearmanr(p["point"], p[act])[0]
        Crows.append({"market":mk,"n":len(p),"over_rate":round(over,3),
                      "line_mae":round(mae,2),"line_corr_actual":round(rho,3)})
    Cdf=pd.DataFrame(Crows); print(Cdf.to_string(index=False))
    Cdf.to_csv(os.path.join(OUT,"a2_prop_calibration.csv"), index=False)

    # ---------- D. Existing-model residual diagnostics ----------
    print("\n=== D. dfs_predictions residual diagnostics ===")
    pred = pd.read_sql("SELECT * FROM dfs_predictions", con)
    pred["abs_err"]=pred["error"].abs(); pred["signed"]=pred["actual_ppr"]-pred["predicted_ppr"]
    by_pos = pred.groupby("position").agg(n=("abs_err","size"),mae=("abs_err","mean"),
              bias=("signed","mean"),corr=("predicted_ppr", lambda s: spearmanr(s, pred.loc[s.index,"actual_ppr"])[0]))
    print("by position:\n", by_pos.round(3))
    pred["ou_bucket"]=pd.cut(pred["over_under"],[0,42,46,50,99],labels=["<=42","42-46","46-50","50+"])
    pred["spread_bucket"]=pd.cut(pred["abs_spread"],[-1,3,7,10,99],labels=["pickem","3-7","7-10","10+"])
    by_ou=pred.groupby("ou_bucket").agg(n=("abs_err","size"),mae=("abs_err","mean"),bias=("signed","mean"))
    by_sp=pred.groupby("spread_bucket").agg(n=("abs_err","size"),mae=("abs_err","mean"),bias=("signed","mean"))
    print("\nby over/under:\n", by_ou.round(3))
    print("\nby spread:\n", by_sp.round(3))
    by_pos.round(3).to_csv(os.path.join(OUT,"a2_resid_bypos.csv"))
    by_ou.round(3).to_csv(os.path.join(OUT,"a2_resid_byou.csv"))
    by_sp.round(3).to_csv(os.path.join(OUT,"a2_resid_byspread.csv"))

    # chart: prop line vs actual for reception yds
    p=props[props["market"]=="player_reception_yds"].merge(
        stats[["event_id","player_name","receiving_yards"]], on=["event_id","player_name"]).dropna()
    if len(p):
        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(p["point"], p["receiving_yards"], s=4, alpha=.15)
        lim=max(p["point"].quantile(.99), p["receiving_yards"].quantile(.99))
        ax.plot([0,lim],[0,lim],"r--"); ax.set_xlabel("rec-yds line"); ax.set_ylabel("actual rec yds")
        ax.set_title("Prop line vs actual (reception yds)"); fig.tight_layout()
        fig.savefig(os.path.join(OUT,"a2_prop_recyds.png"), dpi=110); plt.close(fig)
    con.close()
    print("\nSaved avenue-2 CSVs + charts to", OUT)


if __name__ == "__main__":
    main()
