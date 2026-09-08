"""Run Model_Burke on each prop market (baseline = the book's closing line) and
simulate betting the 2025 season from its calibrated distributions."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SCRATCH = os.path.dirname(os.path.abspath(__file__))
# argv[1] = path to the model_burke package dir (the folder CONTAINING model_burke/)
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_BURKE_PKG", "")
if not PKG or not os.path.isdir(PKG):
    raise SystemExit("pass the model_burke package dir as argv[1] "
                     "(unzip model_burke_pkg.zip and point at its pkg/ folder)")
sys.path.insert(0, PKG)
from model_burke import pipeline

MARKETS = ["player_reception_yds", "player_receptions", "player_rush_yds", "player_pass_yds"]
QCOLS = ["mb_p10", "mb_p25", "mb_p50", "mb_p75", "mb_p90"]
QLEV = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

def p_over(row, line):
    qs = np.array([row[c] for c in QCOLS], dtype=float)
    if np.isnan(qs).any(): return np.nan
    qs = np.maximum.accumulate(qs)
    if line <= qs[0]:
        slope = (qs[1] - qs[0]) / 0.15 if qs[1] > qs[0] else 1.0
        p = 0.10 - (qs[0] - line) / max(slope, 1e-6) * 1.0
        p = max(0.02, min(p, 0.10))
        return 1 - p
    if line >= qs[-1]:
        slope = (qs[-1] - qs[-2]) / 0.15 if qs[-1] > qs[-2] else 1.0
        p = 0.90 + (line - qs[-1]) / max(slope, 1e-6) * 1.0
        p = min(0.98, max(p, 0.90))
        return 1 - p
    cdf = np.interp(line, qs, QLEV)
    return 1 - cdf

def amer_imp(o):
    o = float(o)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)

def amer_profit(o):
    o = float(o)
    return o / 100 if o > 0 else 100 / -o

all_bets = []
summary = []
for mkt in MARKETS:
    d = pd.read_parquet(os.path.join(SCRATCH, f"props_{mkt}.parquet"))
    d["market_proj"] = np.nan            # baseline IS the market; skip the benchmark col
    print(f"\n######## {mkt} ({len(d):,} rows) ########")
    ev, rep = pipeline.run(d, verbose=False)
    pt = rep["point"].set_index("model") if "model" in rep["point"].columns else rep["point"]
    print(pt.round(4).to_string())
    print(f"80% coverage: {rep['interval_80_coverage']}")
    wr = rep.get("win_rate_vs_baseline", {})
    print(f"weekly win rate vs the line: {wr.get('win_rate', float('nan')):.1%} of {wr.get('weeks', 0)} weeks")

    e25 = ev[(ev.season == 2025) & ev.mb_p50.notna()].copy()
    e25["p_over"] = [p_over(r, r.baseline_proj) for _, r in e25.iterrows()]
    e25 = e25.dropna(subset=["p_over", "over_price", "under_price", "actual_ppr"])
    e25 = e25[(e25.over_price.abs() >= 100) & (e25.under_price.abs() >= 100)]
    e25["imp_over"] = [amer_imp(x) for x in e25.over_price]
    e25["imp_under"] = [amer_imp(x) for x in e25.under_price]
    tot_imp = e25.imp_over + e25.imp_under
    e25["fair_over"] = e25.imp_over / tot_imp
    e25["edge"] = e25.p_over - e25.fair_over
    e25["market"] = mkt

    for thr in (0.03, 0.05, 0.08, 0.12):
        b = e25[e25.edge.abs() >= thr].copy()
        if not len(b): continue
        b["side"] = np.where(b.edge > 0, "Over", "Under")
        b["push"] = b.actual_ppr == b.baseline_proj
        b["win"] = np.where(b.side == "Over", b.actual_ppr > b.baseline_proj,
                            b.actual_ppr < b.baseline_proj)
        b["price"] = np.where(b.side == "Over", b.over_price, b.under_price)
        b["profit"] = np.where(b.push, 0.0,
                       np.where(b.win, [amer_profit(x) for x in b.price], -1.0))
        live = b[~b.push]
        summary.append({"market": mkt, "thr": thr, "bets": len(b), "pushes": int(b.push.sum()),
                        "win%": live.win.mean(), "roi": b.profit.sum() / len(b),
                        "units": b.profit.sum()})
        if thr == 0.05: all_bets.append(b)

    # sanity: blind over / blind under at the same prices
    for side in ("Over", "Under"):
        w = (e25.actual_ppr > e25.baseline_proj) if side == "Over" else (e25.actual_ppr < e25.baseline_proj)
        push = e25.actual_ppr == e25.baseline_proj
        pr = e25.over_price if side == "Over" else e25.under_price
        prof = np.where(push, 0.0, np.where(w, [amer_profit(x) for x in pr], -1.0))
        print(f"  blind {side}: n={len(e25)}, win {w[~push].mean():.3f}, roi {prof.sum()/len(e25):+.3f}")

S = pd.DataFrame(summary)
print("\n================ 2025 BETTING SIM (model vs de-vigged close) ================")
print(S.round(3).to_string(index=False))
if all_bets:
    B = pd.concat(all_bets)
    print(f"\nPOOLED @5% edge: {len(B)} bets, win {B[~B.push].win.mean():.3f}, "
          f"roi {B.profit.sum()/len(B):+.4f} ({B.profit.sum():+.1f}u)")
    print(B.groupby("side").agg(n=("win", "size"), win=("win", "mean"),
                                roi=("profit", "mean")).round(3).to_string())
    print("\nby week (pooled, 5% edge):")
    wkr = B.groupby("week").agg(n=("profit", "size"), units=("profit", "sum")).round(1)
    print(wkr.T.to_string())
    B.to_parquet(os.path.join(SCRATCH, "props_bets_2025.parquet"))
