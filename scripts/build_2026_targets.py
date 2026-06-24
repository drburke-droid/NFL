"""
Unified 2026 player board for draft-day targeting (keepers unknown -> all players,
incl. rookies, available). For each player projects:
  - pred_ppg          : season model (prior-year-anchored; no 2026 FFA yet)
  - breakout_prob     : P(>=+4 PPG jump vs 2025) for veterans
  - hit_prob          : P(startable rookie season) for rookies
  - prior_cv/prior_games : production certainty + durability
Writes board_2026 (superset of season_proj_2026 schema) for the draft tool + report.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
def _imp(name):
    s=importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__),name+".py"))
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
BSD=_imp("build_season_dataset"); MS=_imp("model_season"); MR=_imp("model_rookie")

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS=["QB","RB","WR","TE"]
GBM=dict(n_estimators=400,learning_rate=0.03,num_leaves=24,min_child_samples=40,
         subsample=0.8,colsample_bytree=0.8,random_state=0,verbosity=-1)
BSIG=[c for c in MS.FEATURES]  # prior-only signals for breakout (no FFA in 2026)


def veterans(con):
    sf=BSD.build_season_frame(con)
    ds=pd.read_sql("SELECT * FROM season_dataset", con)
    prior=sf[sf.season==2025][BSD.FEAT_COLS].copy()
    prior.columns=["player_id","player_display_name","position","prior_season","prior_team"]+["prior_"+c for c in BSD.FEAT_COLS[5:]]
    prior["season"]=2026
    prior2=sf[sf.season==2024][["player_id","ppg"]].rename(columns={"ppg":"prior2_ppg"})
    ctx=sf[sf.season==2025][["player_id","recent_team","age","years_exp","height","weight",
        "draft_round","draft_pick","forty","vertical","broad_jump","cone","shuttle"]].rename(columns={"recent_team":"team"})
    ctx["age"]+=1; ctx["years_exp"]+=1
    df=prior.merge(prior2,on="player_id",how="left").merge(ctx,on="player_id",how="left")
    df["team_change"]=0; df["pos_id"]=df["position"].map({p:i for i,p in enumerate(POS)})
    df=df[df["prior_games"].fillna(0)>=3].copy()

    # FFA-anchor the projection IF 2026 consensus exists (else prior-year-only guess)
    ds=MS.attach_ffa(ds,con); df=MS.attach_ffa(df,con)
    has_ffa = int(df["ffa_points"].notna().sum())
    feats = (MS.FEATURES+MS.FFA_FEATURES) if has_ffa>=20 else MS.FEATURES
    print(f"  2026 FFA coverage: {has_ffa} players -> {'FFA-ANCHORED' if has_ffa>=20 else 'prior-year-only (no 2026 market yet)'}")
    tr=ds[ds.next_ppg.notna()].copy(); tr["pos_id"]=tr["position"].map({p:i for i,p in enumerate(POS)})
    msm=lgb.LGBMRegressor(objective="regression_l1",**GBM).fit(tr[feats+["pos_id"]].astype(float).fillna(-1),tr.next_ppg)
    df["pred_ppg"]=msm.predict(df[feats+["pos_id"]].astype(float).fillna(-1))
    # breakout classifier (prior-only)
    tr["breakout"]=((tr.next_ppg-tr.prior_ppg)>=4).astype(int)
    spw=(len(tr)-tr.breakout.sum())/max(tr.breakout.sum(),1)
    bm=lgb.LGBMClassifier(objective="binary",scale_pos_weight=spw,**GBM).fit(tr[BSIG+["pos_id"]].astype(float).fillna(-1),tr.breakout)
    df["breakout_prob"]=bm.predict_proba(df[BSIG+["pos_id"]].astype(float).fillna(-1))[:,1]
    df["proj_games"]=MS.project_games(df)
    df["is_rookie"]=0; df["hit_prob"]=np.nan
    # unified distribution from the overhaul model (central/floor/ceiling/bust/boom)
    po=pd.read_sql("SELECT player_id, central, floor, ceiling, bust, boom FROM proj_overhaul", con)
    df=df.merge(po, on="player_id", how="left")
    df["pred_ppg"]=df["central"].fillna(df["pred_ppg"])           # one projection, everywhere
    return df[["player_id","player_display_name","position","team","age","prior_ppg","prior_cv","prior_games",
               "pred_ppg","floor","ceiling","bust","boom","breakout_prob","hit_prob","proj_games","is_rookie"]]


PFR2NFLV={"GNB":"GB","KAN":"KC","LAR":"LA","LVR":"LV","NOR":"NO","NWE":"NE","SFO":"SF","TAM":"TB"}


def rookies(con):
    hist=MR.build(con)                       # historical rookies (features + ppg)
    feats=MR.BASE
    rk=lgb.LGBMRegressor(objective="regression_l1",**GBM).fit(hist[feats].astype(float).fillna(-1),hist["ppg"])
    hist["hit"]=hist.apply(lambda r:int(r["ppg"]>={"QB":14,"RB":10,"WR":9,"TE":7}.get(r["position"],99)),axis=1)
    spw=(len(hist)-hist.hit.sum())/max(hist.hit.sum(),1)
    hc=lgb.LGBMClassifier(objective="binary",scale_pos_weight=spw,**GBM).fit(hist[feats].astype(float).fillna(-1),hist["hit"])

    d=pd.read_sql("""SELECT gsis_id player_id, pfr_player_id, pfr_player_name name, position, team,
                     pick draft_pick, round draft_round, age FROM nflv_draft
                     WHERE season=2026 AND position IN ('QB','RB','WR','TE')""",con)
    d["team"]=d["team"].replace(PFR2NFLV)
    # 2026 combine athleticism (join by PFR id)
    cb=pd.read_sql("""SELECT pfr_id, forty, vertical, broad_jump, cone, shuttle, bench, ht, wt
                      FROM nflv_combine WHERE season=2026""",con)
    def to_in(h):
        try:
            if pd.isna(h): return np.nan
            if "-" in str(h): f,i=str(h).split("-"); return int(f)*12+int(i)
            return float(h)
        except Exception: return np.nan
    cb["height"]=cb["ht"].map(to_in); cb["weight"]=pd.to_numeric(cb["wt"],errors="coerce")
    d=d.merge(cb.drop(columns=["ht","wt"]), left_on="pfr_player_id", right_on="pfr_id", how="left")
    # landing spot from 2025 team production
    seas=pd.read_sql("SELECT recent_team team, position, attempts, carries, fantasy_points_ppr FROM nflv_season WHERE season=2025",con)
    ta=seas.groupby("team").agg(team_off_ppg_prior=("fantasy_points_ppr",lambda s:s.sum()/17),
        team_pass_att_prior=("attempts",lambda s:s.sum()/17), team_rush_att_prior=("carries",lambda s:s.sum()/17)).reset_index()
    tp=seas.groupby(["team","position"]).agg(team_pos_ppg_prior=("fantasy_points_ppr",lambda s:s.sum()/17)).reset_index()
    d=d.merge(ta,on="team",how="left").merge(tp,on=["team","position"],how="left")
    d["pos_id"]=d["position"].map({p:i for i,p in enumerate(POS)})
    d["pred_ppg"]=rk.predict(d[feats].astype(float).fillna(-1)).clip(min=0)
    d["hit_prob"]=hc.predict_proba(d[feats].astype(float).fillna(-1))[:,1]
    # rookie distribution: per-position quantile floor/ceiling + calibrated bust/boom
    RB_BUST={"QB":14,"RB":10,"WR":9,"TE":7}; RB_ELITE={"QB":18,"RB":14,"WR":13,"TE":10}
    Xd=d[feats].astype(float).fillna(-1)
    d["floor"]=np.nan; d["ceiling"]=np.nan; d["bust"]=np.nan; d["boom"]=np.nan
    for pos in POS:
        hm=hist[hist.position==pos]; idx=d.index[d.position==pos]
        if len(hm)<30 or len(idx)==0: continue
        Xh=hm[feats].astype(float).fillna(-1)
        for a,nm in [(0.15,"floor"),(0.85,"ceiling")]:
            d.loc[idx,nm]=lgb.LGBMRegressor(objective="quantile",alpha=a,**GBM).fit(Xh,hm["ppg"]).predict(Xd.loc[idx]).clip(min=0)
        for line,nm in [(RB_BUST[pos],"bust"),(RB_ELITE[pos],"boom")]:
            y=(hm["ppg"]<line).astype(int) if nm=="bust" else (hm["ppg"]>=line).astype(int)
            if y.sum()<8 or y.sum()>len(y)-8: d.loc[idx,nm]=round(float(y.mean()),3); continue
            cc=CalibratedClassifierCV(lgb.LGBMClassifier(objective="binary",**GBM),method="sigmoid",cv=3).fit(Xh,y)
            d.loc[idx,nm]=np.clip(cc.predict_proba(Xd.loc[idx])[:,1],0.02,0.95).round(3)
    d["player_display_name"]=d["name"]; d["prior_ppg"]=np.nan; d["prior_cv"]=np.nan
    d["prior_games"]=np.nan; d["breakout_prob"]=np.nan; d["is_rookie"]=1
    d["proj_games"]=15.0
    return d[["player_id","player_display_name","position","team","age","prior_ppg","prior_cv","prior_games",
              "pred_ppg","floor","ceiling","bust","boom","breakout_prob","hit_prob","proj_games","is_rookie"]]


def main():
    con=sqlite3.connect(DB)
    v=veterans(con); r=rookies(con)
    board=pd.concat([v,r],ignore_index=True)
    board=board[board.pred_ppg.notna()].copy()
    # enforce monotone distribution: floor <= central(pred_ppg) <= ceiling
    board["floor"]=board[["floor","pred_ppg"]].min(axis=1).round(1)
    board["ceiling"]=board[["ceiling","pred_ppg"]].max(axis=1).round(1)
    board.to_sql("board_2026", con, if_exists="replace", index=False)
    con.close()
    print(f"board_2026: {len(board)} players ({(board.is_rookie==1).sum()} rookies)")
    print("\nTop 10 rookies by projected PPG:")
    print(r.sort_values("pred_ppg",ascending=False).head(10)[["player_display_name","position","team","draft_pick" if "draft_pick" in r else "pred_ppg","pred_ppg","hit_prob"]].round(2).to_string(index=False) if "draft_pick" in r else r.sort_values("pred_ppg",ascending=False).head(10)[["player_display_name","position","team","pred_ppg","hit_prob"]].round(2).to_string(index=False))
    print("\nTop 12 veteran breakout candidates (ascending: age<=26, projected role pred_ppg>=8):")
    vb=v[(v.pred_ppg>=8)&(v.age<=26)].sort_values("breakout_prob",ascending=False).head(12)
    print(vb[["player_display_name","position","team","age","prior_ppg","pred_ppg","breakout_prob"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
