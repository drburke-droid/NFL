"""
What signals identify rookies who produce OUT OF THE GATE (year-1 fantasy starters)?

Reuses the rookie dataset (draft capital, combine athleticism, age, landing spot).
"Hit" = rookie-year PPG at a fantasy-startable level for the position. Reports
base rates by draft capital, the discriminating signals, a walk-forward classifier,
and the immediate-impact profile.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import importlib.util
_mr = importlib.util.spec_from_file_location("mr", os.path.join(os.path.dirname(__file__),"model_rookie.py"))
R = importlib.util.module_from_spec(_mr); _mr.loader.exec_module(R)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
HIT_PPG = {"QB":14.0,"RB":10.0,"WR":9.0,"TE":7.0}
FEATS = ["draft_pick","draft_round","age","forty","vertical","broad_jump","cone","shuttle",
         "bench","height","weight","team_off_ppg_prior","team_pos_ppg_prior",
         "team_pass_att_prior","team_rush_att_prior","pos_id"]


def main():
    con = sqlite3.connect(DB)
    df = R.build(con); con.close()
    df["hit"] = df.apply(lambda r: int(r["ppg"] >= HIT_PPG.get(r["position"], 99)), axis=1)
    print(f"Rookie seasons: {len(df):,} (2011-2025) | immediate-hit base rate: {df.hit.mean():.1%}")
    print("by position:", df.groupby('position')['hit'].mean().round(3).to_dict())

    # base rate by draft capital
    print("\n=== Immediate-hit rate by draft slot ===")
    df["slot"]=pd.cut(df["draft_pick"],[0,15,40,75,150,400],labels=["R1 top15","R1-2","R3","R4-5","R6-UDFA"])
    print(df.groupby(["position","slot"])["hit"].agg(["mean","size"]).round(2).to_string())

    # discriminating signals
    print("\n=== Top discriminating signals (hit vs miss) ===")
    rows=[]
    for s in FEATS:
        d=df[[s,"hit"]].dropna()
        if len(d)<100 or d[s].nunique()<5: continue
        rows.append({"signal":s,"hit_mean":d[d.hit==1][s].mean(),"miss_mean":d[d.hit==0][s].mean(),
                     "rho":spearmanr(d[s],d["hit"])[0]})
    sg=pd.DataFrame(rows).sort_values("rho",key=lambda x:x.abs(),ascending=False)
    print(sg.head(12).to_string(index=False,float_format=lambda v:f"{v:.2f}"))

    # walk-forward classifier
    params=dict(objective="binary",n_estimators=300,learning_rate=0.03,num_leaves=20,
                min_child_samples=25,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
    preds=[]
    for T in range(2015,2026):
        tr=df[df.season<T]; te=df[df.season==T]
        if len(te)==0 or tr.hit.sum()<10: continue
        spw=(len(tr)-tr.hit.sum())/max(tr.hit.sum(),1)
        m=lgb.LGBMClassifier(scale_pos_weight=spw,**params).fit(tr[FEATS].astype(float).fillna(-1),tr.hit)
        t=te.copy(); t["prob"]=m.predict_proba(te[FEATS].astype(float).fillna(-1))[:,1]; preds.append(t)
    p=pd.concat(preds)
    auc=roc_auc_score(p.hit,p.prob); k=int(0.2*len(p)); top=p.nlargest(k,"prob")
    print(f"\n=== Walk-forward rookie-hit classifier (test 2015-2025, n={len(p)}) ===")
    print(f"  ROC AUC {auc:.3f} | base rate {p.hit.mean():.1%}")
    print(f"  Precision in top 20%: {top.hit.mean():.1%} ({top.hit.mean()/p.hit.mean():.2f}x base) | recall {top.hit.sum()/p.hit.sum():.0%}")
    mfull=lgb.LGBMClassifier(scale_pos_weight=(len(df)-df.hit.sum())/df.hit.sum(),**params).fit(df[FEATS].astype(float).fillna(-1),df.hit)
    imp=pd.DataFrame({"signal":FEATS,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    print("  Top model signals:", ", ".join(imp.head(7)["signal"]))

    # profile
    print("\n=== Immediate-hit profile (hit rate by signal level) ===")
    for s,lab in [("draft_pick","Draft pick (lower=better)"),("team_pos_ppg_prior","Team prior pos production (vacancy)"),
                  ("age","Draft age"),("weight","Weight")]:
        d=df[[s,"hit"]].dropna()
        d["q"]=pd.qcut(d[s].rank(method="first"),4,labels=["Q1(low)","Q2","Q3","Q4(high)"])
        print(f"  {lab:34s} " + "  ".join(f"{q}:{r:.0%}" for q,r in d.groupby('q')['hit'].mean().items()))


if __name__ == "__main__":
    main()
