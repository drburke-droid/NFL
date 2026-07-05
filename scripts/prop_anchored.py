"""Market-anchored prop probability model: baseline = de-vigged closing consensus P(over);
question = can our features IMPROVE that probability out-of-sample?
Train 2024 -> test 2025 (honest forward direction). Ladder of models:
  M0 raw novig                       (the market)
  M1 global shrink                   (novig + constant over-bias correction estimated on 2024)
  M2 market-only logistic            (novig + market id + line level)
  M3 anchored LGBM                   (novig + our mu/sig gap + all player/context features)
Metrics: Brier / log-loss vs realized over, calibration, then ROI betting only where the anchored
model disagrees enough with a book's actual price (EV > threshold at best and median book)."""
import os, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = pd.read_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
A = A.dropna(subset=["novig", "won_over"]).copy()
A["mkt_id"] = A.market.astype("category").cat.codes
A["z_gap"] = (A.mu - A.mline) / A.sig          # our model's standardized disagreement with the line
A["line_lvl"] = A.groupby("market").mline.transform(lambda x: (x - x.mean()) / (x.std() or 1))
FE_PANEL = [c for c in A.columns if c.startswith(("l3_", "l6_", "d_"))] + \
           ["gp", "itt", "spread", "gt", "arch_id", "dstyle_id", "pos_id", "p_vs_style"]
tr, te = A[A.season == 2024], A[A.season == 2025]
print(f"train 2024: {len(tr):,}   test 2025: {len(te):,}   (base over-rate train {tr.won_over.mean():.3f}, novig {tr.novig.mean():.3f})")
res = {}
def ev(name, p):
    p = np.clip(p, 0.01, 0.99)
    res[name] = (brier_score_loss(te.won_over, p), log_loss(te.won_over, p))
    return p
# M0 raw market
p0 = ev("M0 raw novig", te.novig.values)
# M1 global shrink (estimated on train only)
shift = tr.won_over.mean() - tr.novig.mean()
p1 = ev(f"M1 global shrink ({shift:+.3f})", te.novig.values + shift)
# M2 market-only logistic
X2tr = tr[["novig", "mkt_id", "line_lvl"]].values; X2te = te[["novig", "mkt_id", "line_lvl"]].values
m2 = LogisticRegression(C=1.0, max_iter=500).fit(X2tr, tr.won_over)
p2 = ev("M2 market-only logistic", m2.predict_proba(X2te)[:, 1])
# M3 anchored LGBM (market prob + our full feature set + model disagreement)
F3 = ["novig", "mkt_id", "line_lvl", "z_gap", "mu", "sig"] + FE_PANEL
GB = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15,
          min_child_samples=50, subsample=0.8, colsample_bytree=0.7, random_state=0, verbosity=-1)
m3 = lgb.LGBMClassifier(**GB).fit(tr[F3].astype(float).fillna(-1), tr.won_over)
p3 = ev("M3 anchored LGBM (full)", m3.predict_proba(te[F3].astype(float).fillna(-1))[:, 1])
print("\n=== Brier / log-loss on 2025 (lower = better) ===")
for k, (b, l) in res.items():
    print(f"  {k:32s} Brier {b:.5f}   logloss {l:.5f}")
imp = pd.DataFrame({"f": F3, "i": m3.feature_importances_}).sort_values("i", ascending=False)
print("  M3 top features: " + ", ".join(imp.head(8).f))
# calibration of M3
q = pd.qcut(p3, 8, labels=False, duplicates="drop")
cal = pd.DataFrame({"p": p3, "y": te.won_over.values, "q": q}).groupby("q").agg(p=("p", "mean"), real=("y", "mean"), n=("y", "size"))
print("\nM3 calibration:\n" + cal.round(3).to_string())
# ---------- EV: bet where anchored prob vs actual PRICE clears threshold ----------
print("\n=== ROI on 2025: bet when anchored-EV > threshold ===")
for lab, pcol_o, pcol_u in [("best-book", "best_over", "best_under")]:
    for name, p in [("M1 shrink", p1), ("M3 anchored", p3)]:
        for th in (0.02, 0.05):
            evo = p * te[pcol_o].values - 1
            evu = (1 - p) * te[pcol_u].values - 1
            side = np.where(evo > evu, 1, 0)
            best_ev = np.maximum(evo, evu)
            m = best_ev > th
            if m.sum() < 50: print(f"  {name} {lab} EV>{th:.0%}: n={m.sum()} too few"); continue
            win = (side[m] == te.won_over.values[m]).astype(int)
            dp = np.where(side[m] == 1, te[pcol_o].values[m], te[pcol_u].values[m])
            pnl = np.where(win == 1, dp - 1, -1.0)
            sh = side[m].mean()
            print(f"  {name:12s} {lab} EV>{th:.0%}: n={m.sum():5d}  hit {win.mean():.1%}  ROI {pnl.mean():+.1%}  (overs {sh:.0%})")
# median-book execution for the anchored model
med_o = te.novig / np.clip(1 - 0.045, 0.5, 1)   # approx median over price from novig + typical 4.5% two-way vig
# use actual median prices instead: reconstruct from novig -> price requires book data; approximate via best*0.965
for name, p in [("M3 anchored", p3)]:
    for th in (0.02, 0.05):
        bo = te.best_over.values * 0.965; bu = te.best_under.values * 0.965   # haircut best -> ~median
        evo = p * bo - 1; evu = (1 - p) * bu - 1
        side = np.where(evo > evu, 1, 0); best_ev = np.maximum(evo, evu); m = best_ev > th
        if m.sum() < 50: continue
        win = (side[m] == te.won_over.values[m]).astype(int)
        dp = np.where(side[m] == 1, bo[m], bu[m])
        pnl = np.where(win == 1, dp - 1, -1.0)
        print(f"  {name:12s} ~median (best x.965) EV>{th:.0%}: n={m.sum():5d}  hit {win.mean():.1%}  ROI {pnl.mean():+.1%}")
# where do M3's adjustments come from? size of deviation from market
dev = p3 - te.novig.values
print(f"\nM3 deviation from market: mean {dev.mean():+.3f}, sd {dev.std():.3f}, |dev|>5pp: {(np.abs(dev)>0.05).mean():.0%}")
out = te.copy(); out["p3"] = p3; out.to_pickle(os.path.join(ROOT, "outputs", "prop_anchored.pkl"))
