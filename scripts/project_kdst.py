"""
Honest K/DST projections for the draft tool.

Both positions are near-random year-over-year (prior->next Spearman ~0.17, and
"repeat last year" is worse than predicting the mean). So we:
  1. Test whether a Vegas/pass-rush-aware DST model beats the naive baseline.
  2. Project 2026 with HEAVY shrinkage toward the positional mean (reflecting the
     weak signal), then collapse to tiers. This flattens VORP so K/DST stop
     implying false precision; the auction already caps them at ~$1.

Writes nflv_kdst_proj (season 2026): position, name, team, proj_pts, tier, conf.
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
# Optimal shrinkage = the year-over-year correlation: best predictor is
# mean + r*(last_year - mean). Empirically r(DST)~0.15, r(K)~0.18.
DST_SHRINK = 0.15
K_SHRINK = 0.18


def main():
    con = sqlite3.connect(DB)
    dst = pd.read_sql("SELECT * FROM nflv_team_def", con)
    kick = pd.read_sql("SELECT * FROM nflv_kicking", con)
    gl = pd.read_sql("SELECT season, team, team_spread FROM nflv_game_lines WHERE game_type='REG'", con)

    # team strength proxy: avg point spread (negative = favored = strong)
    strength = gl.groupby(["season","team"])["team_spread"].mean().reset_index().rename(columns={"team_spread":"strength"})

    # ---- DST panel with prior-year features + prior strength ----
    d = dst.sort_values(["team","season"]).copy()
    for c in ["custom_pts","sacks","ints"]:
        d[f"prior_{c}"] = d.groupby("team")[c].shift(1)
    d["prior2_pts"] = d.groupby("team")["custom_pts"].shift(2)
    d = d.merge(strength.assign(season=strength.season+1).rename(columns={"strength":"prior_strength"}),
                on=["season","team"], how="left")
    feats = ["prior_custom_pts","prior_sacks","prior_ints","prior2_pts","prior_strength"]
    panel = d.dropna(subset=["prior_custom_pts","prior_sacks","prior_ints"]).copy()

    # ---- walk-forward validation: model vs baselines ----
    print("=== DST next-year projection — walk-forward (test 2016-2025) ===")
    preds=[]
    for T in range(2016, 2026):
        tr = panel[panel.season < T]; te = panel[panel.season == T]
        if len(te)==0 or len(tr)<40: continue
        m = Ridge(alpha=5.0).fit(tr[feats].fillna(tr[feats].mean()), tr["custom_pts"])
        p = te.copy(); p["pred"] = m.predict(te[feats].fillna(tr[feats].mean())); preds.append(p)
    pr = pd.concat(preds)
    print(f"  model     Spearman {spearmanr(pr.pred, pr.custom_pts)[0]:.3f} | MAE {mean_absolute_error(pr.custom_pts,pr.pred):.1f}")
    print(f"  repeat-LY Spearman {spearmanr(pr.prior_custom_pts, pr.custom_pts)[0]:.3f} | MAE {mean_absolute_error(pr.custom_pts,pr.prior_custom_pts):.1f}")
    print(f"  flat-mean MAE {mean_absolute_error(pr.custom_pts,[pr.custom_pts.mean()]*len(pr)):.1f}")

    # ---- 2026 DST projection ----
    # The model can't beat the mean (above), so don't trust features. Keep only the
    # least-bad weak signal (prior year) and shrink it hard toward the league mean.
    full = panel[panel.season<=2025]
    mean_dst = dst[dst.season>=2021]["custom_pts"].mean()
    cur = d[d.season==2025].copy()
    cur["proj_pts"] = mean_dst + DST_SHRINK*(cur["custom_pts"] - mean_dst)   # prior-year, 70% shrunk
    dst26 = cur[["team","proj_pts"]].copy(); dst26["position"]="DST"; dst26["name"]=dst26["team"]+" DST"

    # ---- 2026 K projection: shrink prior ppg heavily to mean, x games ----
    k = kick.sort_values(["player_id","season"]).copy()
    k["ppg"]=k["custom_pts"]/k["games"].clip(lower=1)
    k25 = k[(k.season==2025)&(k.games>=6)].copy()
    kmean = k[k.games>=8]["ppg"].mean()
    k25["proj_ppg"] = kmean + K_SHRINK*(k25["ppg"]-kmean)
    k25["proj_pts"] = k25["proj_ppg"]*16
    k26 = k25[["name","team","proj_pts"]].copy(); k26["position"]="K"

    out = pd.concat([dst26[["position","name","team","proj_pts"]],
                     k26[["position","name","team","proj_pts"]]], ignore_index=True)
    # tiers (gap-based within position)
    out["tier"]=0
    for pos,gap in [("DST",3),("K",6)]:
        sub=out[out.position==pos].sort_values("proj_pts",ascending=False)
        tier=1; prev=None
        for i in sub.index:
            v=out.at[i,"proj_pts"]
            if prev is not None and (prev-v)>gap: tier+=1
            out.at[i,"tier"]=tier; prev=v
    out["conf"]="low"; out["season"]=2026
    out["proj_pts"]=out["proj_pts"].round(1)
    out.to_sql("nflv_kdst_proj", con, if_exists="replace", index=False)
    con.close()
    print(f"\nnflv_kdst_proj: {len(out)} rows  (spread compressed via {int((1-DST_SHRINK)*100)}%/{int((1-K_SHRINK)*100)}% shrinkage (DST/K))")
    print("DST range:", round(out[out.position=='DST'].proj_pts.min(),1), "-", round(out[out.position=='DST'].proj_pts.max(),1),
          "| K range:", round(out[out.position=='K'].proj_pts.min(),1), "-", round(out[out.position=='K'].proj_pts.max(),1))
    print(out[out.position=='DST'].nlargest(5,'proj_pts')[['name','proj_pts','tier']].to_string(index=False))


if __name__ == "__main__":
    main()
