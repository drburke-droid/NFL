"""
What signals identify players who SIGNIFICANTLY outperform their prior season?

Breakout = next-season PPG at least +4 above prior-season PPG (a ~68-point jump),
among players with a real prior role (>=4 games). Uses season_dataset (prior-year
production/role/context + FFA consensus). Reports the discriminating signals, a
walk-forward classifier (can we beat the base rate?), and the breakout profile.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import importlib.util
_ms = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__),"model_season.py"))
M = importlib.util.module_from_spec(_ms); _ms.loader.exec_module(M)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB","RB","WR","TE"]
JUMP = 4.0

SIGNALS = ["prior_ppg","prior2_ppg","prior_cv","prior_ppr_std","prior_off_pct","prior_target_share",
           "prior_air_yards_share","prior_wopr","prior_total_tds","prior_rec_td_rate","prior_receiving_epa",
           "prior_targets_pg","prior_receptions_pg","age","years_exp","draft_pick","draft_round",
           "ffa_points","ffa_ceiling","ffa_uncertainty","ffa_dropoff","ffa_vor"]


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    df = M.attach_ffa(df, con); con.close()
    df = df[(df.next_ppg.notna()) & (df.prior_games>=4) & (df.position.isin(POS))].copy()
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    # ceiling headroom from consensus (how much higher is FFA ceiling vs its point projection)
    df["ffa_ceiling_gap"] = df["ffa_ceiling"] - df["ffa_points"]
    SIG = SIGNALS + ["ffa_ceiling_gap","pos_id"]
    df["breakout"] = ((df.next_ppg - df.prior_ppg) >= JUMP).astype(int)
    print(f"Rows: {len(df):,} | breakout base rate (+{JUMP} PPG): {df.breakout.mean():.1%}")
    print("by position:", df.groupby('position')['breakout'].mean().round(3).to_dict())

    # univariate signal strength: mean value among breakouts vs non, + point-biserial
    print("\n=== Top discriminating signals (breakout vs not) ===")
    rows=[]
    for s in SIG:
        d=df[[s,"breakout"]].dropna()
        if len(d)<100: continue
        rho=spearmanr(d[s],d["breakout"])[0]
        rows.append({"signal":s,"breakout_mean":d[d.breakout==1][s].mean(),
                     "nonbreak_mean":d[d.breakout==0][s].mean(),"rho":rho})
    sg=pd.DataFrame(rows).sort_values("rho",key=lambda x:x.abs(),ascending=False)
    print(sg.head(14).to_string(index=False,float_format=lambda v:f"{v:.3f}"))

    # walk-forward classifier
    params=dict(objective="binary",n_estimators=350,learning_rate=0.03,num_leaves=24,
                min_child_samples=40,subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
    preds=[]
    for T in range(2016,2026):
        tr=df[df.season<T]; te=df[df.season==T]
        if len(te)==0 or tr.breakout.sum()<20: continue
        spw=(len(tr)-tr.breakout.sum())/max(tr.breakout.sum(),1)
        m=lgb.LGBMClassifier(scale_pos_weight=spw,**params).fit(tr[SIG].astype(float).fillna(-1),tr.breakout)
        t=te.copy(); t["prob"]=m.predict_proba(te[SIG].astype(float).fillna(-1))[:,1]; preds.append(t)
    p=pd.concat(preds)
    auc=roc_auc_score(p.breakout,p.prob); ap=average_precision_score(p.breakout,p.prob)
    k=int(0.2*len(p)); top=p.nlargest(k,"prob")
    print(f"\n=== Walk-forward breakout classifier (test 2016-2025, n={len(p)}) ===")
    print(f"  ROC AUC {auc:.3f} | AP {ap:.3f} | base rate {p.breakout.mean():.1%}")
    print(f"  Precision in model's top 20%: {top.breakout.mean():.1%}  ({top.breakout.mean()/p.breakout.mean():.2f}x base)")
    print(f"  Recall captured by top 20%: {top.breakout.sum()/p.breakout.sum():.0%}")

    mfull=lgb.LGBMClassifier(scale_pos_weight=(len(df)-df.breakout.sum())/df.breakout.sum(),**params)
    mfull.fit(df[SIG].astype(float).fillna(-1),df.breakout)
    imp=pd.DataFrame({"signal":SIG,"importance":mfull.feature_importances_}).sort_values("importance",ascending=False)
    print("\n  Top model signals:", ", ".join(imp.head(8)["signal"]))

    # the breakout profile (who breaks out) — compare quartiles of key signals
    print("\n=== Breakout profile (breakout rate by signal level) ===")
    for s,lab in [("age","Age"),("prior_ppg","Prior PPG"),("prior_off_pct","Prior snap %"),
                  ("ffa_ceiling_gap","FFA ceiling headroom"),("prior_cv","Prior volatility (CV)")]:
        d=df[[s,"breakout"]].dropna()
        d["q"]=pd.qcut(d[s],4,labels=["Q1(low)","Q2","Q3","Q4(high)"],duplicates="drop")
        rates=d.groupby("q")["breakout"].mean()
        print(f"  {lab:22s} " + "  ".join(f"{q}:{r:.0%}" for q,r in rates.items()))


if __name__ == "__main__":
    main()
