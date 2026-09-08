"""Props backtest v2 — fixed probability layer.

v1 read P(over) off the median-centred quantiles: on discrete receptions the
median bias (-0.5) turned into a permanent phantom under-edge. v2 keeps the
point model but fits P(over) DIRECTLY, walk-forward: logistic on
  z    = (Model_Burke_mean - line) / spread_width
  line = the line level (discreteness varies: a 1.5-line behaves differently
         from a 6.5-line)
trained on all strictly-prior weeks, refit each week. Bias and integer mass are
absorbed by the calibration instead of assumed away.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

SCRATCH = os.path.dirname(os.path.abspath(__file__))
# argv[1] = path to the model_burke package dir (the folder CONTAINING model_burke/)
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_BURKE_PKG", "")
if not PKG or not os.path.isdir(PKG):
    raise SystemExit("pass the model_burke package dir as argv[1] "
                     "(unzip model_burke_pkg.zip and point at its pkg/ folder)")
sys.path.insert(0, PKG)
from model_burke import pipeline

MARKETS = ["player_receptions", "player_reception_yds", "player_rush_yds"]

def amer_imp(o):
    o = float(o); return 100 / (o + 100) if o > 0 else -o / (-o + 100)
def amer_profit(o):
    o = float(o); return o / 100 if o > 0 else 100 / -o

all_bets = []
summary = []
for mkt in MARKETS:
    d = pd.read_parquet(os.path.join(SCRATCH, f"props_{mkt}.parquet"))
    d["market_proj"] = np.nan
    print(f"\n######## {mkt} ########")
    ev, rep = pipeline.run(d, verbose=False)

    ev = ev.dropna(subset=["Model_Burke_mean", "mb_p25", "mb_p75",
                           "actual_ppr", "baseline_proj"]).copy()
    ev["sd"] = ((ev.mb_p75 - ev.mb_p25) / 1.35).clip(lower=1e-3)
    ev["z"] = ((ev.Model_Burke_mean - ev.baseline_proj) / ev.sd).clip(-4, 4)
    ev["went_over"] = (ev.actual_ppr > ev.baseline_proj).astype(int)
    ev["is_push"] = ev.actual_ppr == ev.baseline_proj
    ev = ev.sort_values("t")

    # walk-forward logistic probability layer
    ev["p_over2"] = np.nan
    ts = sorted(ev.t.unique())
    for t in ts:
        tr = ev[(ev.t < t) & (~ev.is_push)]
        te = ev.index[ev.t == t]
        if len(tr) < 400 or not len(te):
            continue
        X = tr[["z", "baseline_proj"]].values
        lr = LogisticRegression(C=1.0).fit(X, tr.went_over.values)
        ev.loc[te, "p_over2"] = lr.predict_proba(ev.loc[te, ["z", "baseline_proj"]].values)[:, 1]

    e25 = ev[(ev.season == 2025) & ev.p_over2.notna()].copy()
    e25 = e25.dropna(subset=["over_price", "under_price"])
    e25 = e25[(e25.over_price.abs() >= 100) & (e25.under_price.abs() >= 100)]
    io = e25.over_price.map(amer_imp); iu = e25.under_price.map(amer_imp)
    e25["fair_over"] = io / (io + iu)
    e25["edge"] = e25.p_over2 - e25.fair_over
    e25["market"] = mkt
    # calibration check on 2025: predicted vs realised over-rate by decile
    cal = e25[~e25.is_push].copy()
    cal["bin"] = pd.qcut(cal.p_over2, 10, duplicates="drop")
    cc = cal.groupby("bin", observed=True).agg(pred=("p_over2", "mean"),
                                              real=("went_over", "mean"), n=("z", "size"))
    print("calibration (2025): pred vs realised over-rate")
    print(cc.round(3).to_string())

    for thr in (0.03, 0.05, 0.08, 0.12):
        b = e25[e25.edge.abs() >= thr].copy()
        if not len(b): continue
        b["side"] = np.where(b.edge > 0, "Over", "Under")
        b["win"] = np.where(b.side == "Over", b.actual_ppr > b.baseline_proj,
                            b.actual_ppr < b.baseline_proj)
        b["price"] = np.where(b.side == "Over", b.over_price, b.under_price)
        b["profit"] = np.where(b.is_push, 0.0,
                       np.where(b.win, b.price.map(amer_profit), -1.0))
        live = b[~b.is_push]
        summary.append({"market": mkt, "thr": thr, "bets": len(b),
                        "over%": (b.side == "Over").mean(),
                        "win%": live.win.mean(), "roi": b.profit.sum() / len(b),
                        "units": b.profit.sum()})
        if thr == 0.05: all_bets.append(b)

S = pd.DataFrame(summary)
print("\n================ 2025 SIM v2 (calibrated P(over)) ================")
print(S.round(3).to_string(index=False))
if all_bets:
    B = pd.concat(all_bets)
    live = B[~B.is_push]
    print(f"\nPOOLED @5%: {len(B)} bets, win {live.win.mean():.3f}, "
          f"roi {B.profit.sum()/len(B):+.4f} ({B.profit.sum():+.1f}u; "
          f"${25*B.profit.sum():+,.0f} at $25 flat)")
    print(B.groupby(["market", "side"]).agg(n=("win", "size"), win=("win", "mean"),
                                            roi=("profit", "mean")).round(3).to_string())
    B.to_parquet(os.path.join(SCRATCH, "props_bets_2025_v2.parquet"))
