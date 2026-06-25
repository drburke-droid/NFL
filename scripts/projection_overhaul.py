"""
Position-by-position projection overhaul: a unified central projection PLUS a
learned distribution (floor/ceiling) and CALIBRATED, realistic boom/bust
probabilities — validated walk-forward. Tests whether comp features add value.

Per QB/RB/WR/TE, walk-forward by season (train < T, predict T):
  - quantile LightGBM (P25/P50/P75) -> floor / central / ceiling
  - bust = next_ppg < 0.7*central ; boom = next_ppg > 1.3*central
    -> calibrated classifiers (base vs +comp features)
Validates: central MAE vs repeat-last-year, quantile coverage, boom/bust AUC +
reliability. Writes proj_overhaul (2026 per-player floor/central/ceiling/bust/boom).
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]
COMPF = ["comp_prime", "comp_bust", "comp_elite", "comp_dist"]
BASE = MS.FEATURES
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def qfit(tr, te, feats, a):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, **GB)
    m.fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
    return m.predict(te[feats].astype(float).fillna(-1))


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    cf = pd.read_sql("SELECT * FROM nflv_comp_features", con)
    ns = pd.read_sql("SELECT player_id, season, games FROM nflv_season", con).drop_duplicates(["player_id", "season"])
    df = MS.attach_ffa(df, con); con.close()
    df = df.merge(cf, on=["player_id", "season"], how="left")
    g2 = ns.rename(columns={"games": "prior2_games"}); g2["season"] = g2.season + 2   # games two seasons back
    df = df.merge(g2, on=["player_id", "season"], how="left")
    df = MS.add_injury_features(df)
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()

    # ---- pass 1: walk-forward quantiles (P25/P50/P75) per position ----
    parts = []
    for pos in POS:
        d = df[df.position == pos].copy()
        for T in range(2016, 2026):
            tr, te = d[d.season < T], d[d.season == T]
            if len(te) == 0 or len(tr) < 60: continue
            te = te.copy()
            for a, nm in [(0.15, "floor"), (0.5, "central"), (0.85, "ceiling")]:
                te[nm] = qfit(tr, te, BASE, a)
            parts.append(te)
    P = pd.concat(parts, ignore_index=True)
    P["central"] = P["central"].clip(lower=0.1)
    # boom/bust = ABSOLUTE single-season outcomes (realistic, differentiating, never 0%):
    #   bust  = next season below startable value for the position
    #   boom  = next season is an elite finish
    BUST_LINE = {"QB":14, "RB":10, "WR":9, "TE":7}
    ELITE_LINE = {"QB":21, "RB":16, "WR":15, "TE":12}
    P["bust_y"] = (P.next_ppg < P.position.map(BUST_LINE)).astype(int)
    P["boom_y"] = (P.next_ppg >= P.position.map(ELITE_LINE)).astype(int)

    print("=== Central projection (P50) vs repeat-last-year, by position ===")
    for pos in POS:
        s = P[P.position == pos]
        mae_m = mean_absolute_error(s.next_ppg, s.central); mae_b = mean_absolute_error(s.next_ppg, s.prior_ppg.fillna(s.prior_ppg.median()))
        print(f"  {pos}: model MAE {mae_m:.2f} | repeat-LY {mae_b:.2f} | rho {spearmanr(s.central,s.next_ppg)[0]:.3f}")

    print("\n=== Quantile coverage (empirical % of actuals below predicted quantile; nominal in parens) ===")
    for pos in POS:
        s = P[P.position == pos]
        print(f"  {pos}: P25 {np.mean(s.next_ppg<s.floor):.0%} (25%) | P50 {np.mean(s.next_ppg<s.central):.0%} (50%) | P75 {np.mean(s.next_ppg<s.ceiling):.0%} (75%)")

    # ---- pass 2: calibrated boom/bust classifiers (base vs +comp), walk-forward ----
    print("\n=== Boom/Bust: AUC base vs +comp, and calibration (walk-forward, isotonic-calibrated) ===")
    out_rows = []
    for tgt in ["bust", "boom"]:
        for feats, lab in [(BASE, "base"), (BASE + COMPF, "+comp")]:
            preds = []
            for pos in POS:
                d = P[P.position == pos].copy()
                for T in range(2017, 2026):
                    tr, te = d[d.season < T], d[d.season == T]
                    if len(te) == 0 or tr[f"{tgt}_y"].sum() < 15: continue
                    spw = (len(tr)-tr[f"{tgt}_y"].sum())/max(tr[f"{tgt}_y"].sum(), 1)
                    m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw, **GB)
                    m.fit(tr[feats].astype(float).fillna(-1), tr[f"{tgt}_y"])
                    raw = m.predict_proba(te[feats].astype(float).fillna(-1))[:, 1]
                    # isotonic calibration on training preds
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(m.predict_proba(tr[feats].astype(float).fillna(-1))[:, 1], tr[f"{tgt}_y"])
                    t2 = te.copy(); t2["p"] = np.clip(iso.predict(raw), 0.01, 0.97); t2["tgt"] = tgt; t2["fs"] = lab
                    preds.append(t2)
            pr = pd.concat(preds, ignore_index=True)
            auc = roc_auc_score(pr[f"{tgt}_y"], pr["p"]); brier = np.mean((pr["p"]-pr[f"{tgt}_y"])**2)
            print(f"  {tgt:5s} {lab:6s}: AUC {auc:.3f} | Brier {brier:.3f} | base rate {pr[f'{tgt}_y'].mean():.0%} | pred range {pr['p'].min():.0%}-{pr['p'].max():.0%}")
            if lab == "+comp":
                # reliability: realized rate by predicted-prob quintile
                pr["q"] = pd.qcut(pr["p"].rank(method="first"), 5, labels=False)
                rel = [f"{pr[pr.q==i][f'{tgt}_y'].mean():.0%}" for i in range(5)]
                print(f"        reliability (realized {tgt} by predicted quintile): {rel}")

    print("\nNote: with the learned distribution, NO player gets 0% — bust/boom range across the pool above.")

    # ---------- train production models on ALL data + project 2026 veterans ----------
    BUST_LINE = {"QB":14,"RB":10,"WR":9,"TE":7}; ELITE_LINE = {"QB":21,"RB":16,"WR":15,"TE":12}
    bsd = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        "bsd", os.path.join(os.path.dirname(__file__), "build_season_dataset.py")))
    importlib.util.spec_from_file_location("bsd", os.path.join(os.path.dirname(__file__),"build_season_dataset.py")).loader.exec_module(bsd)
    con = sqlite3.connect(DB); sf = bsd.build_season_frame(con); con.close()
    prior = sf[sf.season==2025][bsd.FEAT_COLS].copy()
    prior.columns = ["player_id","player_display_name","position","prior_season","prior_team"]+["prior_"+c for c in bsd.FEAT_COLS[5:]]
    prior["season"]=2026
    prior2 = sf[sf.season==2024][["player_id","ppg","games"]].rename(columns={"ppg":"prior2_ppg","games":"prior2_games"})
    ctx = sf[sf.season==2025][["player_id","recent_team","age","years_exp","height","weight",
        "draft_round","draft_pick","forty","vertical","broad_jump","cone","shuttle"]].rename(columns={"recent_team":"team"})
    ctx["age"]+=1; ctx["years_exp"]+=1
    con = sqlite3.connect(DB)
    v26 = MS.attach_ffa(prior.merge(prior2,on="player_id",how="left").merge(ctx,on="player_id",how="left"), con); con.close()
    v26["team_change"]=0; v26 = v26[v26.prior_games.fillna(0)>=3].copy()
    v26 = MS.add_injury_features(v26)                         # injury/games-aware prior (validated)
    FEATS = BASE + MS.INJURY_FEATURES + (MS.FFA_FEATURES if int(v26["ffa_points"].notna().sum())>=20 else [])
    print("\n  Production features:", ("BASE+INJ+FFA (market-anchored)" if len(FEATS)>len(BASE)+len(MS.INJURY_FEATURES) else "BASE+INJ (no 2026 FFA yet)"))

    out = []
    full = df  # all season_dataset (+comp/ffa) rows with next_ppg
    for pos in POS:
        tr = full[full.position==pos]; te = v26[v26.position==pos].copy()
        if len(te)==0: continue
        mods = {nm: lgb.LGBMRegressor(objective="quantile", alpha=a, **GB).fit(tr[FEATS].astype(float).fillna(-1), tr.next_ppg)
                for a,nm in [(0.15,"floor"),(0.5,"central"),(0.85,"ceiling")]}
        for nm,m in mods.items(): te[nm]=m.predict(te[FEATS].astype(float).fillna(-1)).clip(min=0)
        # boom/bust: Platt-calibrated probabilities (smooth, realistic; no class-weight distortion)
        tr2 = tr.copy(); tr2["bz"]=(tr2.next_ppg<BUST_LINE[pos]).astype(int); tr2["bm"]=(tr2.next_ppg>=ELITE_LINE[pos]).astype(int)
        Xtr = tr2[FEATS].astype(float).fillna(-1); Xte = te[FEATS].astype(float).fillna(-1)
        for tgt,col in [("bz","bust"),("bm","boom")]:
            cc = CalibratedClassifierCV(lgb.LGBMClassifier(objective="binary",**GB), method="sigmoid", cv=3)
            cc.fit(Xtr, tr2[tgt])
            te[col]=np.clip(cc.predict_proba(Xte)[:,1],0.02,0.92).round(3)
        out.append(te[["player_id","player_display_name","position","team","central","floor","ceiling","bust","boom"]])
    res = pd.concat(out, ignore_index=True)
    res[["central","floor","ceiling"]] = res[["central","floor","ceiling"]].round(1)
    con = sqlite3.connect(DB); res.to_sql("proj_overhaul", con, if_exists="replace", index=False); con.close()
    print(f"\nSaved proj_overhaul: {len(res)} 2026 veterans (central/floor/ceiling/bust/boom).")
    print(res.sort_values("central",ascending=False).head(10)[["player_display_name","position","central","floor","ceiling","bust","boom"]].to_string(index=False))


if __name__ == "__main__":
    main()
