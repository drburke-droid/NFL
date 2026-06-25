"""
Does adding the teammate-opportunity features to the MODEL improve projections
walk-forward — or is the +1.7 PPG subgroup edge already absorbed once it's a feature?

BASE vs BASE+OPP, season-blocked (train < T, predict T), per position:
  - central PPG MAE (LightGBM P50)
  - bust / boom AUC
Plus a focused check on the RB subgroup the signal lives in (vacated lead role):
  does the feature model actually lift those specific players toward their outcome?
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]; BASE = MS.FEATURES
OPP = ["vac_rb_carries", "inc_rb_carries", "net_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
BUST = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}; ELITE = {"QB": 21, "RB": 16, "WR": 15, "TE": 12}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr, vacated_role FROM nflv_opportunity", con)
    con.close()
    df = df.merge(opp, on=["player_id", "season"], how="left")
    df["net_rb_carries"] = df.vac_rb_carries - df.inc_rb_carries
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    cov = df.vac_rb_carries.notna().mean()
    print(f"Rows {len(df):,} | opportunity coverage {cov:.0%} (returning players)\n")

    def wf(feats, target=None, line=None):
        rows = []
        for pos in POS:
            d = df[df.position == pos].copy()
            if target: d["y"] = (d.next_ppg < line[pos]).astype(int) if target == "bust" else (d.next_ppg >= line[pos]).astype(int)
            for T in range(2017, 2026):
                tr, te = d[d.season < T], d[d.season == T]
                if len(te) == 0 or len(tr) < 80: continue
                if target:
                    if tr.y.sum() < 15: continue
                    m = lgb.LGBMClassifier(objective="binary", **GB); m.fit(tr[feats].astype(float).fillna(-1), tr.y)
                    te = te.copy(); te["p"] = m.predict_proba(te[feats].astype(float).fillna(-1))[:, 1]
                else:
                    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB); m.fit(tr[feats].astype(float).fillna(-1), tr.next_ppg)
                    te = te.copy(); te["pred"] = m.predict(te[feats].astype(float).fillna(-1))
                rows.append(te)
        return pd.concat(rows, ignore_index=True)

    print("=== Central PPG MAE (lower better) ===")
    print(f"{'pos':5s} {'BASE':>7s} {'+OPP':>8s} {'delta':>8s}")
    b_all, h_all = wf(BASE), wf(BASE + OPP)
    for pos in POS + ["ALL"]:
        b, h = b_all, h_all
        if pos != "ALL": b = b[b.position == pos]; h = h[h.position == pos]
        mb, mh = mean_absolute_error(b.next_ppg, b.pred), mean_absolute_error(h.next_ppg, h.pred)
        print(f"{pos:5s} {mb:7.3f} {mh:8.3f} {mh-mb:+8.3f}")

    print("\n  -- RB focus: the 'vacated lead role' players specifically --")
    rb_b = b_all[(b_all.position == "RB") & (b_all.vacated_role == 1)]
    rb_h = h_all[(h_all.position == "RB") & (h_all.vacated_role == 1)]
    print(f"  vacated-role RBs (n={len(rb_h)}): mean predicted PPG  BASE {rb_b.pred.mean():.1f} -> +OPP {rb_h.pred.mean():.1f}  | actual {rb_h.next_ppg.mean():.1f}")
    print(f"  vacated-role RBs MAE: BASE {mean_absolute_error(rb_b.next_ppg, rb_b.pred):.2f} -> +OPP {mean_absolute_error(rb_h.next_ppg, rb_h.pred):.2f}")

    print("\n=== Boom/Bust AUC (higher better) ===")
    for tgt, line in [("bust", BUST), ("boom", ELITE)]:
        b, h = wf(BASE, tgt, line), wf(BASE + OPP, tgt, line)
        yb = (b.next_ppg < b.position.map(line)).astype(int) if tgt == "bust" else (b.next_ppg >= b.position.map(line)).astype(int)
        yh = (h.next_ppg < h.position.map(line)).astype(int) if tgt == "bust" else (h.next_ppg >= h.position.map(line)).astype(int)
        print(f"  {tgt:5s} BASE {roc_auc_score(yb, b.p):.3f} | +OPP {roc_auc_score(yh, h.p):.3f} | delta {roc_auc_score(yh,h.p)-roc_auc_score(yb,b.p):+.3f}")


if __name__ == "__main__":
    main()
