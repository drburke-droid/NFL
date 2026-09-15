"""Week-2 study, part 2: corrections ON TOP OF the incumbent, and decile diagnostics.

  (a) decile diagnostics: incumbent residual (actual - inc) by within-position decile of each week-1 signal,
      pooled week-2 rows 2017-2025.  A monotone trend = the incumbent under/over-reacts to that signal.
  (b) walk-forward ridge on the incumbent's residual, per position, trained on prior seasons' week-2 rows
      or weeks 2-4 (early-season regime), shrunk k = 0.5, alpha 30 / 100.
  (c) 2024-25: DK-implied points blended on top of the incumbent, inc + w * (mkt - ffa).
Reads week2_preds.parquet written by week2_study.py.  Appends to the same report.
"""
import os, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, binomtest, linregress
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True)
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "week2_study.md"))
A = ap.parse_args()
P = pd.read_parquet(os.path.join(A.scratch, "week2_preds.parquet"))
F = pd.read_parquet(os.path.join(A.scratch, "frame_full.parquet"))
POS = ["QB", "RB", "WR", "TE"]
SIG = ["pts_surp", "snap1", "rshare1", "tshare1", "car1", "tgt1", "pre_gap", "ffa_move", "impl_c", "ffa_lvl"]
P["r_inc"] = P.actual_ppr - P.inc
L = ["\n# Part 2: on top of the incumbent\n", "## (a) Incumbent residual by within-position decile of each week-1 signal (week 2, 2017-2025, played rows)\n",
     "Slope = OLS of residual on decile (PPR per decile), with p. A real under-reaction shows as a monotone positive slope.\n",
     "| signal | " + " | ".join(f"{p} slope (p)" for p in POS) + " |\n|---|" + "---|" * len(POS)]
for sg in SIG:
    cells = []
    for p in POS:
        d = P[(P.position == p) & (P.played1 == 1)] if sg not in ("pre_gap", "ffa_move", "impl_c", "ffa_lvl") else P[P.position == p]
        d = d[d[sg].notna()]
        if d[sg].nunique() < 10: cells.append("n/a"); continue
        dec = pd.qcut(d[sg].rank(method="first"), 10, labels=False)
        r = linregress(dec, d.r_inc); cells.append(f"{r.slope:+.2f} ({r.pvalue:.2f})")
    L.append(f"| {sg} | " + " | ".join(cells) + " |")
# top/bottom decile means for the two headline signals
L.append("\nTop vs bottom decile of the incumbent residual (mean actual - inc), played-week-1 rows:\n")
L.append("| signal | pos | bottom decile | top decile | n/decile |\n|---|---|---|---|---|")
for sg in ("pts_surp", "snap1", "pre_gap"):
    for p in POS:
        d = P[(P.position == p) & (P.played1 == 1)]; dec = pd.qcut(d[sg].rank(method="first"), 10, labels=False)
        L.append(f"| {sg} | {p} | {d.r_inc[dec == 0].mean():+.2f} | {d.r_inc[dec == 9].mean():+.2f} | {int((dec == 0).sum())} |")

# (b) walk-forward ridge on the incumbent residual
# need incumbent preds for weeks 3-4 too (early-season pool): rebuild the same features for weeks 2-4 from the frame
inc = pd.read_parquet(os.path.join(A.scratch, "incumbent_preds.parquet"))
F = F.join(inc); F["inc"] = F.inc_full
w1 = F[F.week == 1][["player_id", "season", "baseline_proj", "actual_ppr", "actual_active"]].rename(columns={"baseline_proj": "ffa1", "actual_ppr": "act1", "actual_active": "active1"})
E = F[(F.week.between(2, 4)) & (F.actual_active == 1) & F.inc.notna()].merge(w1, on=["player_id", "season"], how="left")
E["played1"] = (E.active1 == 1).astype(float); E["pts_surp"] = np.where(E.played1 == 1, (E.act1 - E.ffa1).fillna(0), 0.0)
E["ffa_move"] = (E.baseline_proj - E.ffa1).fillna(0); E["ffa_lvl"] = E.baseline_proj; E["impl_c"] = (E.ctx_implied - E.groupby(["season", "week"]).ctx_implied.transform("mean")).fillna(0)
E["tshare1"] = E.lag_target_share_l1.fillna(0); E["rshare1"] = E.rt_route_share_l1.fillna(0); E["car1"] = E.lag_carries_l1.fillna(0); E["tgt1"] = E.lag_targets_l1.fillna(0)
E["fp_l1"] = E.lag_fantasy_points_ppr_l1.fillna(0); E["r_inc"] = E.actual_ppr - E.inc
FE = ["pts_surp", "tshare1", "rshare1", "car1", "tgt1", "fp_l1", "ffa_move", "impl_c", "ffa_lvl", "played1"]


def ridge_on_inc(tr, te, alpha, k):
    out = te.inc.copy()
    for p in POS:
        a, b = tr[tr.position == p], te[te.position == p]
        if len(a) < 60 or len(b) == 0: continue
        mu, sd = a[FE].mean(), a[FE].std().replace(0, 1)
        m = Ridge(alpha=alpha).fit((a[FE] - mu) / sd, a.r_inc - a.r_inc.median())
        out.loc[b.index] = b.inc + k * m.predict((b[FE] - mu) / sd)
    return out


rows = []
for s in range(2018, 2026):
    te = E[(E.season == s) & (E.week == 2)].copy()
    for lab, tr in (("wk2", E[(E.season < s) & (E.week == 2)]), ("wk2-4", E[(E.season < s)])):
        for alpha in (30, 100):
            te[f"inc+ridge_{lab}_a{alpha}"] = ridge_on_inc(tr, te, alpha, 0.5)
    rows.append(te)
R = pd.concat(rows)
cols = [c for c in R.columns if c.startswith("inc+ridge")]
L.append("\n## (b) Ridge on the incumbent residual, k = 0.5 (week 2, test seasons 2018-2025, n = %d)\n" % len(R))
L.append("| model | MAE | dMAE vs inc | seasons better | t p | sign p |\n|---|---|---|---|---|---|")
ps = R.groupby("season").apply(lambda g: pd.Series({c: (g.actual_ppr - g[c]).abs().mean() for c in ["inc"] + cols}))
for c in cols:
    d = ps[c] - ps["inc"]; t = ttest_rel(ps[c], ps["inc"]); e1 = (R.actual_ppr - R[c]).abs(); e0 = (R.actual_ppr - R.inc).abs()
    w = int((e1 < e0).sum()); n = int((e1 != e0).sum())
    L.append(f"| {c} | {e1.mean():.3f} | {d.mean():+.3f} | {int((d < 0).sum())}/{len(d)} | {t.pvalue:.2f} | {binomtest(w, n).pvalue:.2f} |")
L.append(f"\nincumbent MAE on the same rows: {(R.actual_ppr - R.inc).abs().mean():.3f}\n")

# (c) DK on top of the incumbent
MP = pd.read_parquet(os.path.join(A.scratch, "week2_mkt_preds.parquet"))
L.append("## (c) DK-implied points on top of the incumbent, inc + w * (mkt - ffa) (2024-25 week 2, rows with a DK line, n = %d)\n" % len(MP))
L.append("| w | MAE | 2024 | 2025 | Spearman |\n|---|---|---|---|---|")
from scipy.stats import spearmanr
for w in (0, 0.2, 0.3, 0.4, 0.5, 0.7):
    MP["m"] = MP.inc + w * (MP.mkt2 - MP.ffa)
    rho = np.mean([spearmanr(g.m, g.actual_ppr)[0] for _, g in MP.groupby("season")])
    L.append(f"| {w} | {(MP.actual_ppr - MP.m).abs().mean():.3f} | " + " | ".join(f"{(g.actual_ppr - g.m).abs().mean():.3f}" for _, g in MP.groupby("season")) + f" | {rho:.3f} |")
# how does the same blend do at weeks 5-9 (where production already uses DK), for scale
open(A.out, "a", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
