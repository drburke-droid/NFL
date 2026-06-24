"""
Walk-forward test: does the half-season role/production TREND add value to the
season projection beyond what prior-year stats already capture?

Compares BASE vs BASE+half-trend, per position, season-blocked (train < T,
predict T), on:
  - central PPG MAE (LightGBM P50)
  - bust / boom AUC (absolute single-season outcomes, this league's lines)
Verdict only — does not touch the production model unless it earns its place.
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
POS = ["QB", "RB", "WR", "TE"]
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_h2_tgtsh", "ht_slope"]
BASE = MS.FEATURES
BUST = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}; ELITE = {"QB": 21, "RB": 16, "WR": 15, "TE": 12}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con); con.close()
    df = df.merge(ht, on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    cov = df.ht_h2_ppg.notna().mean()
    print(f"Rows {len(df):,} | half-trend coverage {cov:.0%}\n")

    def central(feats):
        rows = []
        for pos in POS:
            d = df[df.position == pos]
            for T in range(2017, 2026):
                tr, te = d[d.season < T], d[d.season == T]
                if len(te) == 0 or len(tr) < 80: continue
                m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB)
                m.fit(tr[feats].astype(float).fillna(-1), tr.next_ppg)
                t2 = te.copy(); t2["pred"] = m.predict(te[feats].astype(float).fillna(-1))
                rows.append(t2)
        return pd.concat(rows, ignore_index=True)

    def clf(feats, tgt, lines):
        rows = []
        for pos in POS:
            d = df[df.position == pos].copy(); d["y"] = (d.next_ppg < lines[pos]).astype(int) if tgt == "bust" else (d.next_ppg >= lines[pos]).astype(int)
            for T in range(2017, 2026):
                tr, te = d[d.season < T], d[d.season == T]
                if len(te) == 0 or tr.y.sum() < 15: continue
                m = lgb.LGBMClassifier(objective="binary", **GB)
                m.fit(tr[feats].astype(float).fillna(-1), tr.y)
                t2 = te.copy(); t2["p"] = m.predict_proba(te[feats].astype(float).fillna(-1))[:, 1]
                rows.append(t2)
        return pd.concat(rows, ignore_index=True)

    print("=== Central PPG MAE (lower better) ===")
    print(f"{'pos':4s} {'BASE':>7s} {'+trend':>8s} {'delta':>7s}")
    for pos in POS + ["ALL"]:
        b = central(BASE); h = central(BASE + HT)
        if pos != "ALL": b = b[b.position == pos]; h = h[h.position == pos]
        mb = mean_absolute_error(b.next_ppg, b.pred); mh = mean_absolute_error(h.next_ppg, h.pred)
        print(f"{pos:4s} {mb:7.3f} {mh:8.3f} {mh-mb:+7.3f}")

    print("\n=== Boom/Bust AUC (higher better) ===")
    for tgt, lines in [("bust", BUST), ("boom", ELITE)]:
        b = clf(BASE, tgt, lines); h = clf(BASE + HT, tgt, lines)
        bm = b.set_index(["player_id", "season"]); hm = h.set_index(["player_id", "season"])
        yb = (bm.next_ppg < bm.position.map(lines)).astype(int) if tgt == "bust" else (bm.next_ppg >= bm.position.map(lines)).astype(int)
        yh = (hm.next_ppg < hm.position.map(lines)).astype(int) if tgt == "bust" else (hm.next_ppg >= hm.position.map(lines)).astype(int)
        print(f"  {tgt:5s} BASE AUC {roc_auc_score(yb, b.p):.3f} | +trend AUC {roc_auc_score(yh, h.p):.3f} | delta {roc_auc_score(yh, h.p)-roc_auc_score(yb, b.p):+.3f}")

    print("\n=== Standalone signal: corr(half-trend, next_ppg) and next-PPG by trend quintile ===")
    for f in ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_d_tgtsh"]:
        s = df[df[f].notna()]
        r = np.corrcoef(s[f], s.next_ppg)[0, 1]
        q = pd.qcut(s[f].rank(method="first"), 5, labels=False)
        means = [f"{s[q==i].next_ppg.mean():.1f}" for i in range(5)]
        print(f"  {f:12s} corr {r:+.3f} | next_ppg by quintile (low->high): {means}")


if __name__ == "__main__":
    main()
