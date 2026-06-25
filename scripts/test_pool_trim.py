"""
What changes if we ignore players who can't be drafted? (12 teams x 16 = 192 spots.)

Two separate questions:
  (1) VALUE side — do replacement levels / VORP / auction $ / tiers change if we drop
      the undraftable tail? (They shouldn't: all are RANK-anchored to starter slots.)
  (2) MODEL side — if we also RETRAINED only on draftable-caliber players, would the
      projections / boom-bust for the players we care about change? (Walk-forward.)
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
BUST = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}; ELITE = {"QB": 21, "RB": 16, "WR": 15, "TE": 12}
# "draftable-caliber" prior-year PPG floor per position (roughly startable in a 12-team league)
FLOOR = {"QB": 12, "RB": 8, "WR": 8, "TE": 6}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def value_side():
    con = sqlite3.connect(DB)
    p = pd.read_sql("SELECT * FROM draft_board_2026", con) if _has(con, "draft_board_2026") else None
    con.close()
    # use the generated data.js instead (authoritative board the app shows)
    import json
    txt = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "data.js"), encoding="utf-8").read()
    pl = pd.DataFrame(json.loads(txt.split("const PLAYERS = ", 1)[1].rsplit(";", 1)[0]))
    pl = pl.sort_values("vorp", ascending=False).reset_index(drop=True)
    N = 192
    draftable, tail = pl.iloc[:N], pl.iloc[N:]
    print(f"Pool: {len(pl)} players. Top {N} (12x16) = 'draftable'; tail = {len(tail)}.")
    print(f"  Draftable VORP range:  {draftable.vorp.min():+.1f} .. {draftable.vorp.max():+.1f}")
    print(f"  Tail VORP range:       {tail.vorp.min():+.1f} .. {tail.vorp.max():+.1f}  (all <= replacement)")
    print(f"  Tail projected PPG:    median {tail.proj_ppg.median():.1f}, max {tail.proj_ppg.max():.1f}")
    print("  Replacement points are set at a FIXED RANK (starter slots), so dropping the")
    print("  tail leaves every replacement level, VORP, tier, and auction $ identical.")
    # prove the tail can't enter the auction: positive-VORP count vs spots
    posv = (pl.vorp > 0).sum()
    print(f"  Players with positive VORP (auction-relevant): {posv}. Tail positive-VORP: {(tail.vorp>0).sum()}")


def _has(con, t):
    return con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None


def model_side():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con); con.close()
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    df["relevant"] = df.apply(lambda r: r.prior_ppg >= FLOOR.get(r.position, 8), axis=1)

    def wf(train_relevant_only, target=None, line=None):
        rows = []
        for pos in POS:
            d = df[df.position == pos].copy()
            if target: d["y"] = (d.next_ppg < line[pos]).astype(int) if target == "bust" else (d.next_ppg >= line[pos]).astype(int)
            for T in range(2017, 2026):
                tr, te = d[d.season < T], d[d.season == T]
                if train_relevant_only: tr = tr[tr.relevant]
                te = te[te.relevant]                       # evaluate only on players we care about
                if len(te) == 0 or len(tr) < 60: continue
                if target:
                    if tr.y.sum() < 12: continue
                    m = lgb.LGBMClassifier(objective="binary", **GB); m.fit(tr[BASE].astype(float).fillna(-1), tr.y)
                    te = te.copy(); te["p"] = m.predict_proba(te[BASE].astype(float).fillna(-1))[:, 1]
                else:
                    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB); m.fit(tr[BASE].astype(float).fillna(-1), tr.next_ppg)
                    te = te.copy(); te["pred"] = m.predict(te[BASE].astype(float).fillna(-1))
                rows.append(te)
        return pd.concat(rows, ignore_index=True)

    print("\nEvaluated ONLY on draftable-caliber players (the ones we care about):")
    print(f"{'':28s} {'train ALL':>10s} {'train RELEVANT-only':>20s}")
    a, b = wf(False), wf(True)
    for pos in POS + ["ALL"]:
        aa, bb = a, b
        if pos != "ALL": aa = aa[aa.position == pos]; bb = bb[bb.position == pos]
        print(f"  central MAE {pos:5s}{'':10s} {mean_absolute_error(aa.next_ppg, aa.pred):>10.3f} {mean_absolute_error(bb.next_ppg, bb.pred):>20.3f}")
    for tgt, line in [("bust", BUST), ("boom", ELITE)]:
        a2, b2 = wf(False, tgt, line), wf(True, tgt, line)
        ya = (a2.next_ppg < a2.position.map(line)).astype(int) if tgt == "bust" else (a2.next_ppg >= a2.position.map(line)).astype(int)
        yb = (b2.next_ppg < b2.position.map(line)).astype(int) if tgt == "bust" else (b2.next_ppg >= b2.position.map(line)).astype(int)
        print(f"  {tgt} AUC{'':18s} {roc_auc_score(ya, a2.p):>10.3f} {roc_auc_score(yb, b2.p):>20.3f}")


def main():
    print("=" * 70, "\n(1) VALUE SIDE\n" + "=" * 70)
    value_side()
    print("\n" + "=" * 70, "\n(2) MODEL SIDE — retrain on draftable-caliber only?\n" + "=" * 70)
    model_side()


if __name__ == "__main__":
    main()
