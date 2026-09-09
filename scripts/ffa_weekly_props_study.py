"""Does the weekly FFA consensus (rec_yds vs the DK line) add calibrated edge to
Model_Burke's P(over)?  Retro-train on the 2023-25 closing-line history now that
the home-PC weekly FFA backfill (data/ffanalytics/FFAn_weekly/raw_stats_S_wkW.csv,
2023 wk9 - 2025 wk20) has landed.

Walk-forward by (season, week): every logistic is fit only on earlier graded weeks.
Compares:  A  (z, line)                     -- what props_watch ships today
           B  A + ffa_gap (+ missing flag)   -- FFA-vs-line news feature
           C  B + injury flag
           F  (ffa_gap, line) alone          -- is FFA by itself a line-beater?
plus a leakage sanity check (FFA MAE vs closing-line MAE: a 'historical' scrape
that beat the closing line handily would be post-hoc).
Usage: python scripts/ffa_weekly_props_study.py <model_burke pkg dir>
"""
import os, re, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_BURKE_PKG", "")
sys.path.insert(0, PKG)
from model_burke import pipeline

def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
def amer_imp(o):
    o = np.asarray(o, float); return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))
def amer_profit(o):
    o = np.asarray(o, float); return np.where(o > 0, o / 100, 100 / -o)

# ---- weekly FFA history ----
rows = []
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    m = re.search(r"raw_stats_(\d{4})_wk(\d+)", f); s, w = int(m.group(1)), int(m.group(2))
    if s >= 2026: continue
    d = pd.read_csv(f, na_values=["NA"])
    d = d[d.avg_type == "weighted"] if "avg_type" in d.columns else d
    d = d[d.position.isin(["RB", "WR", "TE", "QB"])]
    d["nname"] = d.player.map(norm); d["season"], d["week"] = s, w
    rows.append(d[["nname", "season", "week", "rec_yds", "rec_yds_sd", "rec", "injury_status"]]
                .drop_duplicates("nname"))
ffa = pd.concat(rows, ignore_index=True).rename(
    columns={"rec_yds": "ffa_yds", "rec_yds_sd": "ffa_sd", "rec": "ffa_rec", "injury_status": "ffa_inj"})
print(f"weekly FFA: {len(ffa):,} player-weeks, {ffa.groupby('season').week.nunique().to_dict()} weeks/season")

# ---- history frame + Model_Burke ----
hist = pd.read_parquet(os.path.join(ROOT, "data", "props_frames", "props_player_reception_yds.parquet"))
hist["market_proj"] = np.nan
ev, _ = pipeline.run(hist, verbose=False)
ev["sd"] = ((ev.mb_p75 - ev.mb_p25) / 1.35).clip(lower=1e-3)
ev["z"] = ((ev.Model_Burke_mean - ev.baseline_proj) / ev.sd).clip(-4, 4)
ev["nname"] = ev.player.map(norm)
ev = ev.merge(ffa, on=["nname", "season", "week"], how="left")
g = ev.dropna(subset=["actual_ppr", "Model_Burke_mean", "z", "baseline_proj", "over_price", "under_price"]).copy()
g = g[g.actual_ppr != g.baseline_proj]
# multi-book averaged American prices can be illegal (|price|<100, even 0) on thin
# 2-book rows -> infinite 'profit'; same filter as model_burke_props_run2
g = g[(g.over_price.abs() >= 100) & (g.under_price.abs() >= 100)]
g["y"] = (g.actual_ppr > g.baseline_proj).astype(int)
g["has_ffa"] = g.ffa_yds.notna()
g["ffa_gap"] = (g.ffa_yds - g.baseline_proj).fillna(0.0)
g["ffa_gap_rel"] = (g.ffa_gap / g.baseline_proj.clip(lower=10)).fillna(0.0)
g["ffa_miss"] = (~g.has_ffa).astype(int)
g["inj_flag"] = g.ffa_inj.isin(["Q", "Questionable", "D", "Doubtful", "O", "Out", "IR"]).astype(int)
io, iu = amer_imp(g.over_price), amer_imp(g.under_price)
g["fair"] = io / (io + iu)
g["tkey"] = g.season * 100 + g.week
print(f"graded rows {len(g):,}; with FFA {g.has_ffa.sum():,} "
      f"({g.has_ffa.mean():.0%}); injury-tagged {g.inj_flag.sum()}")
print("FFA injury tags among priced players:", g.ffa_inj.value_counts().to_dict())

# ---- leakage / quality sanity ----
h = g[g.has_ffa]
print("\n== FFA vs closing line on the same player-weeks ==")
print(f"  MAE line->actual {np.abs(h.actual_ppr - h.baseline_proj).mean():.2f} | "
      f"FFA->actual {np.abs(h.actual_ppr - h.ffa_yds).mean():.2f} | "
      f"MB->actual {np.abs(h.actual_ppr - h.Model_Burke_mean).mean():.2f}")
print(f"  corr(ffa_gap, actual-line) = {np.corrcoef(h.ffa_gap, h.actual_ppr - h.baseline_proj)[0,1]:+.3f}"
      f"   corr(mb-line, actual-line) = {np.corrcoef(h.Model_Burke_mean - h.baseline_proj, h.actual_ppr - h.baseline_proj)[0,1]:+.3f}"
      f"   corr(ffa_gap, mb-line) = {np.corrcoef(h.ffa_gap, h.Model_Burke_mean - h.baseline_proj)[0,1]:+.3f}")
print(f"  mean ffa_gap {h.ffa_gap.mean():+.2f} (FFA systematically {'above' if h.ffa_gap.mean()>0 else 'below'} the line)")
for lo, hi in ((-99, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 99)):
    s = h[(h.ffa_gap >= lo) & (h.ffa_gap < hi)]
    if len(s): print(f"  ffa_gap [{lo:>3},{hi:>3}) n={len(s):4d}  over-rate {s.y.mean():.3f}  fair {s.fair.mean():.3f}")

# ---- walk-forward calibration ----
MODELS = {"A z+line": ["z", "baseline_proj"],
          "B +ffa_gap": ["z", "baseline_proj", "ffa_gap", "ffa_miss"],
          "B2 +ffa_gap_rel": ["z", "baseline_proj", "ffa_gap_rel", "ffa_miss"],
          "C +inj": ["z", "baseline_proj", "ffa_gap", "ffa_miss", "inj_flag"],
          "F ffa only": ["ffa_gap", "baseline_proj", "ffa_miss"]}
weeks = sorted(g.tkey.unique())
start = 202401  # 2023 gives the first training block
preds = {k: pd.Series(np.nan, index=g.index) for k in MODELS}
for t in weeks:
    if t < start: continue
    tr, te = g[g.tkey < t], g[g.tkey == t]
    if len(tr) < 300 or not len(te): continue
    for k, cols in MODELS.items():
        lr = LogisticRegression(C=1.0, max_iter=500).fit(tr[cols].values, tr.y.values)
        preds[k].loc[te.index] = lr.predict_proba(te[cols].values)[:, 1]
test = g[preds["A z+line"].notna()].copy()
print(f"\n== walk-forward 2024 wk1 -> 2025 (n={len(test):,}, FFA present {test.has_ffa.mean():.0%}) ==")
print(f"{'model':<18}{'logloss':>9}{'brier':>8}{'auc':>7} | {'@5% n':>6}{'roi':>7}{'@8% n':>6}{'roi':>7} | FFA-rows-only @5% n/roi")
def pocket(df, p, thr):
    b = df[(p - df.fair) >= thr]
    if not len(b): return 0, np.nan
    pr = np.where(b.y == 1, amer_profit(b.over_price), -1.0)
    return len(b), pr.mean()
for k in MODELS:
    p = preds[k].loc[test.index]
    n5, r5 = pocket(test, p, 0.05); n8, r8 = pocket(test, p, 0.08)
    tf = test[test.has_ffa]; nf, rf = pocket(tf, p.loc[tf.index], 0.05)
    print(f"{k:<18}{log_loss(test.y, p):9.4f}{brier_score_loss(test.y, p):8.4f}{roc_auc_score(test.y, p):7.3f} | "
          f"{n5:6d}{r5:+7.1%}{n8:6d}{r8:+7.1%} | {nf:4d} {rf:+.1%}")
print(f"{'market (fair)':<18}{log_loss(test.y, test.fair):9.4f}{brier_score_loss(test.y, test.fair):8.4f}{roc_auc_score(test.y, test.fair):7.3f}")
for k in ("A z+line", "B +ffa_gap", "C +inj"):
    for s in (2024, 2025):
        ts = test[test.season == s]; p = preds[k].loc[ts.index]
        n5, r5 = pocket(ts, p, 0.05); n8, r8 = pocket(ts, p, 0.08)
        print(f"  {k:<16} {s}: @5% n={n5:3d} roi={r5:+.1%}   @8% n={n8:3d} roi={r8:+.1%}")
pa, pb = preds["A z+line"].loc[test.index], preds["B +ffa_gap"].loc[test.index]
fa, fb = (pa - test.fair) >= 0.05, (pb - test.fair) >= 0.05
print(f"\nflag overlap @5%: A only {int((fa & ~fb).sum())}, both {int((fa & fb).sum())}, B only {int((~fa & fb).sum())}")
for lab, m in (("A-only", fa & ~fb), ("both", fa & fb), ("B-only", ~fa & fb)):
    s = test[m]
    if len(s): print(f"  {lab:<7} n={len(s):3d} over-rate {s.y.mean():.3f} roi {np.where(s.y==1, amer_profit(s.over_price), -1).mean():+.1%}  mean ffa_gap {s.ffa_gap.mean():+.1f}")
lr = LogisticRegression(C=1.0, max_iter=500).fit(g[MODELS["C +inj"]].values, g.y.values)
print("\nfull-sample coefs C:", dict(zip(MODELS["C +inj"], np.round(lr.coef_[0], 4))))
test.assign(pA=pa, pB=pb).to_parquet(os.path.join(ROOT, "outputs", "ffa_weekly_props_study_rows.parquet"))
