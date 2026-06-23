"""
Benchmark the season model against the market (preseason ADP/ECR).

For the overlap seasons (2021-2025) and players who have BOTH a model projection
(needs a prior season) and a preseason ADP, compare how well each ranks players
vs their ACTUAL season finish. Also tests whether ADP adds info beyond the model
and whether a model+market ensemble beats either alone.
"""
import os, sqlite3
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
POS = ["QB","RB","WR","TE"]


def main():
    con = sqlite3.connect(DB)
    adp = pd.read_sql("SELECT season,player_id,pos,adp_overall,pos_rank,ecr FROM nflv_adp WHERE player_id IS NOT NULL", con)
    pred = pd.read_sql("""SELECT player_id,player_display_name,position,season,pred_ppg,
                          finish_pred,next_ppg,finish_act,next_games FROM season_predictions""", con)
    con.close()

    m = pred.merge(adp[["season","player_id","pos_rank","ecr","adp_overall"]],
                   on=["season","player_id"], how="inner")
    m = m.dropna(subset=["next_ppg","pred_ppg","ecr","finish_act"])
    print(f"Overlap (veterans w/ ADP + model), 2021-25: {len(m):,} player-seasons")
    print("by season:", m.groupby("season").size().to_dict())

    # ensemble: average of model and market ranks within season/position
    m["model_rank"] = m.groupby(["season","position"])["pred_ppg"].rank(ascending=False)
    m["mkt_rank"]   = m.groupby(["season","position"])["ecr"].rank(ascending=True)
    m["ens_rank"]   = (m["model_rank"] + m["mkt_rank"]) / 2

    def avg_rho(col, sign=1):
        """Mean within season/position Spearman vs actual finish (lower finish=better)."""
        vals=[]
        for (s,p),g in m.groupby(["season","position"]):
            if len(g) < 12: continue
            r = spearmanr(sign*g[col], g["finish_act"])[0]
            vals.append(r)
        return np.mean(vals)

    print("\n=== Rank vs ACTUAL finish (mean within-pos Spearman; higher=better ranking) ===")
    print(f"  Market (ADP/ECR):     {avg_rho('mkt_rank'):.3f}")
    print(f"  Season model:         {avg_rho('model_rank'):.3f}")
    print(f"  Model + Market ens.:  {avg_rho('ens_rank'):.3f}")

    # correlation directly with next_ppg
    def rho_ppg(col, sign=1):
        vals=[]
        for (s,p),g in m.groupby(["season","position"]):
            if len(g)<12: continue
            vals.append(spearmanr(sign*g[col], g["next_ppg"])[0])
        return np.mean(vals)
    print("\n=== Predicting actual PPG (mean within-pos Spearman) ===")
    print(f"  Market (-ECR):        {rho_ppg('ecr', sign=-1):.3f}")
    print(f"  Season model (pred):  {rho_ppg('pred_ppg', sign=1):.3f}")

    print("\n  By position (model | market) rank-vs-finish:")
    for p in POS:
        sub = m[m["position"]==p]
        def rp(col):
            v=[spearmanr(sub[sub.season==s][col], sub[sub.season==s]["finish_act"])[0]
               for s in sub.season.unique() if (sub.season==s).sum()>=12]
            return np.mean(v) if v else float("nan")
        print(f"    {p}: model {rp('model_rank'):.3f}  market {rp('mkt_rank'):.3f}")

    # biggest model-vs-market disagreements and who was right
    m["disagree"] = m["mkt_rank"] - m["model_rank"]   # +ve: model higher on player than market
    m["finish_pctl"] = m.groupby(["season","position"])["finish_act"].rank(pct=True)
    big = m.reindex(m["disagree"].abs().sort_values(ascending=False).index)
    cols=["season","player_display_name","position","mkt_rank","model_rank","next_ppg","finish_act"]
    print("\n=== Largest model-vs-market disagreements ===")
    print("  Model HIGHER than market (model liked them more):")
    print(big[big.disagree>0].head(8)[cols].round(1).to_string(index=False))
    print("\n  Market HIGHER than model (market liked them more):")
    print(big[big.disagree<0].head(8)[cols].round(1).to_string(index=False))

    os.makedirs(OUT, exist_ok=True)
    m.to_csv(os.path.join(OUT,"adp_benchmark.csv"), index=False)
    print("\nSaved adp_benchmark.csv")


if __name__ == "__main__":
    main()
