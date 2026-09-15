"""Practice reports (nflverse injuries, final weekly report) vs availability and performance, 2017-2025.

  1. does the practice status (DNP / limited / full) + official designation predict who sits, beyond the
     FFA injury tag the model already carries?  play-rate tables and a walk-forward P(plays) model
     (log-loss / Brier / AUC by season), plus what the DOUBT path assumes vs what happened.
  2. when a listed player DOES play, how does he perform vs the model: median actual/projection relative
     to unlisted players of the same projection band, snap share vs his own norm, early-exit and bust rates.
  3. a walk-forward multiplicative haircut by (designation, practice) on played rows -> dMAE vs incumbent.

Usage: python scripts/practice_report_study.py --scratch DIR   (frame_full.parquet + incumbent_preds.parquet;
       data/nflverse_cache/injuries_<season>.parquet, snap_counts_<season>.parquet)
"""
import os, re, glob, argparse, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True)
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "practice_report_study.md"))
A = ap.parse_args()


def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)


f = pd.read_parquet(os.path.join(A.scratch, "frame_full.parquet")).join(pd.read_parquet(os.path.join(A.scratch, "incumbent_preds.parquet")))
f = f[f.season >= 2017].copy(); f["inc"] = f.inc_full.fillna(f.baseline_proj)
inj = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(ROOT, "data", "nflverse_cache", "injuries_*.parquet"))])
inj = inj[(inj.game_type == "REG") & inj.gsis_id.notna()].copy()
inj["prac"] = inj.practice_status.map({"Did Not Participate In Practice": "DNP", "Limited Participation in Practice": "LP", "Full Participation in Practice": "FP"}).fillna("listed, no practice status")
inj["desig"] = inj.report_status.fillna("no designation").replace({"Note": "no designation"})
inj["dow"] = pd.to_datetime(inj.date_modified, utc=True, errors="coerce").dt.day_name()
inj = inj.sort_values("date_modified").drop_duplicates(["season", "week", "gsis_id"], keep="last")
f = f.merge(inj[["season", "week", "gsis_id", "prac", "desig", "report_primary_injury", "dow"]].rename(columns={"gsis_id": "player_id"}), on=["season", "week", "player_id"], how="left")
f["prac"] = f.prac.fillna("not listed"); f["desig"] = f.desig.fillna("not listed")
f["ffa_tag"] = np.where(f.ffa_injury_o == 1, "O/D", np.where(f.ffa_injury_q == 1, "Q", "none"))
f["played"] = f.actual_active.astype(int)
# snap share vs own norm (early exits)
sn = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(ROOT, "data", "nflverse_cache", "snap_counts_*.parquet"))])
sn = sn[(sn.game_type == "REG") & sn.position.isin(["QB", "RB", "WR", "TE"])].copy(); sn["nname"] = sn.player.map(norm)
sn = sn.sort_values("offense_pct", ascending=False).drop_duplicates(["season", "week", "nname", "position"])[["season", "week", "nname", "position", "offense_pct"]]
f = f.merge(sn, on=["season", "week", "nname", "position"], how="left")
f["snap_norm"] = f.sort_values(["player_id", "season", "week"]).groupby("player_id").offense_pct.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
f["snap_ratio"] = f.offense_pct / f.snap_norm
f["proj_b"] = pd.cut(f.baseline_proj, [-1, 5, 10, 15, 99], labels=["<5", "5-10", "10-15", "15+"]).astype(str)
f["r"] = f.actual_ppr - f.inc
ORDER_P = ["not listed", "FP", "LP", "DNP", "listed, no practice status"]; ORDER_D = ["not listed", "no designation", "Questionable", "Doubtful", "Out"]
L = ["# Practice reports vs availability and performance (2017-2025)\n",
     f"Projected player-weeks (QB/RB/WR/TE with an FFA line): {len(f):,}; {f.played.mean():.1%} played. Injury report = the nflverse weekly final report (one row per player-week: designation + last practice status). "
     f"Report timestamps fall on {f[f.prac != 'not listed'].dow.value_counts(normalize=True).round(2).head(4).to_dict()}, i.e. it is on hand before the T-90 send. Listed players: {(f.prac != 'not listed').mean():.1%} of rows.\n"]

# ---------- 1. availability ----------
L.append("## 1. Who sits: play rate by designation x practice status\n")
tab = f.groupby(["desig", "prac"]).agg(n=("played", "size"), played=("played", "mean")).reset_index()
L.append("| designation | practice | rows | play rate |\n|---|---|---|---|")
for d in ORDER_D:
    for p in ORDER_P:
        x = tab[(tab.desig == d) & (tab.prac == p)]
        if len(x) and x.n.iloc[0] >= 30: L.append(f"| {d} | {p} | {int(x.n.iloc[0]):,} | {x.played.iloc[0]:.1%} |")
L.append("\nWhat the model sees today is the FFA tag alone. Play rate by FFA tag, then split by the practice status the model does NOT see:\n")
L.append("| FFA tag | practice | rows | play rate |\n|---|---|---|---|")
for t in ("none", "Q", "O/D"):
    x = f[f.ffa_tag == t]; L.append(f"| **{t}** | all | {len(x):,} | {x.played.mean():.1%} |")
    for p in ORDER_P:
        y = x[x.prac == p]
        if len(y) >= 30: L.append(f"| {t} | {p} | {len(y):,} | {y.played.mean():.1%} |")
# by position for Q + DNP / LP (the live decision)
L.append("\nQuestionable by practice status and position:\n\n| position | Q + FP | Q + LP | Q + DNP |\n|---|---|---|---|")
for pos in ("QB", "RB", "WR", "TE"):
    x = f[(f.position == pos) & (f.desig == "Questionable")]
    L.append(f"| {pos} | " + " | ".join(f"{x[x.prac == p].played.mean():.0%} (n={int((x.prac == p).sum())})" if (x.prac == p).sum() >= 15 else "n/a" for p in ("FP", "LP", "DNP")) + " |")
# walk-forward P(plays)
L.append("\nWalk-forward P(plays) (logistic, train = earlier seasons, test 2018-2025) on rows with any injury information (FFA tag or report listing):\n")
sub = f[(f.ffa_tag != "none") | (f.prac != "not listed")].copy()
def design(d, cols):
    X = pd.get_dummies(d[cols].astype(str), drop_first=False).astype(float); return X
res = []
for s in range(2018, 2026):
    tr, te = sub[sub.season < s], sub[sub.season == s]
    for lab, cols in (("FFA tag only", ["ffa_tag"]), ("+ designation", ["ffa_tag", "desig"]), ("+ designation + practice", ["ffa_tag", "desig", "prac"]), ("+ ... + position", ["ffa_tag", "desig", "prac", "position"])):
        X = design(tr, cols); Xt = design(te, cols).reindex(columns=X.columns, fill_value=0.0)
        m = LogisticRegression(C=1.0, max_iter=500).fit(X, tr.played); p = m.predict_proba(Xt)[:, 1]
        res.append({"season": s, "model": lab, "logloss": log_loss(te.played, p), "brier": brier_score_loss(te.played, p), "auc": roc_auc_score(te.played, p)})
R = pd.DataFrame(res).groupby("model", sort=False).mean(numeric_only=True)
L.append("| model | log-loss | Brier | AUC |\n|---|---|---|---|")
for lab, r in R.iterrows(): L.append(f"| {lab} | {r.logloss:.4f} | {r.brier:.4f} | {r.auc:.3f} |")
L.append(f"\nRows in that population: {len(sub):,} ({len(sub) / len(f):.0%} of all); base play rate {sub.played.mean():.1%}.\n")
# the DOUBT path today: P(plays)=0.2 for FFA O/D; what actually happens
od = f[f.ffa_tag == "O/D"]
L.append(f"DOUBT path today assumes P(plays) = 0.2 for FFA Out/Doubtful. Observed: {od.played.mean():.1%} over {len(od):,} rows; of those the official designation was Out {(od.desig == 'Out').mean():.0%} (played {od[od.desig == 'Out'].played.mean():.1%}), Doubtful {(od.desig == 'Doubtful').mean():.0%} (played {od[od.desig == 'Doubtful'].played.mean():.1%}), Questionable {(od.desig == 'Questionable').mean():.0%} (played {od[od.desig == 'Questionable'].played.mean():.1%}), nothing/none {(od.desig.isin(['not listed', 'no designation'])).mean():.0%} (played {od[od.desig.isin(['not listed', 'no designation'])].played.mean():.1%}).\n")

# ---------- 2. performance when they play ----------
L.append("## 2. When a listed player plays: performance vs the model\n")
pl = f[(f.played == 1) & f.inc_full.notna()].copy()
pl["ratio"] = pl.actual_ppr / pl.inc.clip(lower=1.0)
base = pl[pl.prac == "not listed"].groupby("proj_b").ratio.median()
pl["rel"] = pl.ratio / pl.proj_b.map(base)
L.append("Median actual/projection relative to unlisted players in the same projection band (1.00 = performs like a healthy player at that projection), with mean residual, snap share vs own norm, early exits (<50% of usual snaps) and bust rate (residual <= -10):\n")
L.append("| designation | practice | played rows | relative output | mean residual | median snap ratio | early exit | bust rate |\n|---|---|---|---|---|---|---|---|")
for d in ORDER_D:
    for p in ORDER_P:
        x = pl[(pl.desig == d) & (pl.prac == p)]
        if len(x) >= 40:
            L.append(f"| {d} | {p} | {len(x):,} | {x.rel.median():.2f} | {x.r.mean():+.2f} | {x.snap_ratio.median():.2f} | {(x.snap_ratio < 0.5).mean():.0%} | {(x.r <= -10).mean():.1%} |")
L.append("\nBy position, Questionable players who played:\n\n| position | Q + FP output | Q + LP output | Q + DNP output |\n|---|---|---|---|")
for pos in ("QB", "RB", "WR", "TE"):
    x = pl[(pl.position == pos) & (pl.desig == "Questionable")]
    L.append(f"| {pos} | " + " | ".join(f"{x[x.prac == p].rel.median():.2f} (n={int((x.prac == p).sum())})" if (x.prac == p).sum() >= 20 else "n/a" for p in ("FP", "LP", "DNP")) + " |")
# by injury type for DNP/LP who played
L.append("\nBy primary injury (listed players who played, LP or DNP, n >= 60):\n\n| injury | played rows | relative output | early exit | play rate when listed |\n|---|---|---|---|---|")
lp = pl[pl.prac.isin(["LP", "DNP"])]
for injn, x in lp.groupby("report_primary_injury"):
    if len(x) < 60: continue
    allr = f[(f.prac.isin(["LP", "DNP"])) & (f.report_primary_injury == injn)]
    L.append(f"| {injn} | {len(x)} | {x.rel.median():.2f} | {(x.snap_ratio < 0.5).mean():.0%} | {allr.played.mean():.0%} |")
# is the projection already discounting them? compare FFA line vs the player's own recent mean
pl["ffa_vs_r3"] = pl.baseline_proj - pl.lag_fantasy_points_ppr_r3
L.append("\nDoes FFA already discount listed players? FFA line minus the player's trailing-3 average (played rows):\n\n| practice | not listed | FP | LP | DNP |\n|---|---|---|---|---|")
L.append("| FFA - trailing 3 (mean) | " + " | ".join(f"{pl[pl.prac == p].ffa_vs_r3.mean():+.2f}" for p in ("not listed", "FP", "LP", "DNP")) + " |")

# ---------- 3. walk-forward haircut on played rows ----------
L.append("\n## 3. Would a practice-status haircut improve the projection for players who play?\n")
pl["cell"] = pl.desig + "|" + pl.prac
rows = []
for s in range(2018, 2026):
    tr, te = pl[pl.season < s], pl[pl.season == s].copy()
    medc = tr.groupby("cell").r.median(); cnt = tr.groupby("cell").size(); glob_m = tr.r.median()
    off = (medc - glob_m) * cnt / (cnt + 200)          # shrunk median offset per cell
    te["adj"] = te.inc + te.cell.map(off).fillna(0.0)
    mult = (tr.groupby("cell").ratio.median() / tr.ratio.median()); mult = 1 + (mult - 1) * cnt / (cnt + 200)
    te["adj_mult"] = te.inc * te.cell.map(mult).fillna(1.0)
    rows.append(te)
P = pd.concat(rows); listed = P[P.prac != "not listed"]
for lab, d in (("all played rows", P), ("listed players only", listed), ("LP or DNP only", P[P.prac.isin(["LP", "DNP"])])):
    e0 = (d.actual_ppr - d.inc).abs(); e1 = (d.actual_ppr - d.adj).abs(); e2 = (d.actual_ppr - d.adj_mult).abs()
    ps = pd.DataFrame({"a": e1, "m": e2, "b": e0, "s": d.season}).groupby("s").mean()
    L.append(f"- {lab} (n = {len(d):,}): incumbent MAE {e0.mean():.3f}; additive haircut {e1.mean():.3f} ({(ps.a < ps.b).sum()}/{len(ps)} seasons better); multiplicative {e2.mean():.3f} ({(ps.m < ps.b).sum()}/{len(ps)} seasons better).")
os.makedirs(os.path.dirname(A.out), exist_ok=True); open(A.out, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
