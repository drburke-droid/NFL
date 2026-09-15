"""Week-2-only tweak study (Sept 2026).

Question: is there a correction to the weekly projection that works specifically at week 2, when the
only in-season evidence is one game?  Candidates, each evaluated walk-forward by season on played
week-2 rows (train = earlier seasons' week-2 rows unless stated), against FFA and the incumbent:

  pos_offset    per-position median week-2 residual from prior seasons (is week 2 biased by position?)
  surprise      per-position ridge on week-1 surprise: points surprise (act1 - ffa1), usage (wk1 target
                share / route share / snap share / carries), FFA's own reaction (ffa2 - ffa1), distance from
                the preseason per-game line, Vegas implied total.  Shrunk (k) toward zero.
  surprise_all  the same ridge fitted on weeks 2-18 of prior seasons (is the effect week-2-specific?)
  mkt_blend     2023-25 only: DK-implied PPR (props -> points) blended with FFA at week 2, weight fitted
                on prior seasons' weeks 1-9 vs weeks 2 only.

Usage: python scripts/week2_study.py --scratch DIR   (needs frame_full.parquet + incumbent_preds.parquet
from the scratch build: build_full_frame.py)
"""
import os, re, sys, glob, sqlite3, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, ttest_rel, binomtest
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True)
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "week2_study.md"))
A = ap.parse_args()


def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)


f = pd.read_parquet(os.path.join(A.scratch, "frame_full.parquet"))
ip = os.path.join(A.scratch, "incumbent_preds.parquet")
if os.path.exists(ip): f = f.join(pd.read_parquet(ip))
else: f["inc_full"] = np.nan; f["inc_2023"] = np.nan

# ---------- week-1 self-features ----------
w1 = f[f.week == 1][["player_id", "season", "baseline_proj", "actual_ppr", "actual_active", "actual_targets", "actual_carries", "actual_attempts"]]
w1 = w1.rename(columns={"baseline_proj": "ffa1", "actual_ppr": "act1", "actual_active": "active1", "actual_targets": "tgt1", "actual_carries": "car1", "actual_attempts": "att1"})
f = f.merge(w1, on=["player_id", "season"], how="left")
# snap share week 1 (nflverse snap counts, name+position+season match)
sn = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(ROOT, "data", "nflverse_cache", "snap_counts_*.parquet"))])
sn = sn[(sn.week == 1) & (sn.game_type == "REG") & sn.position.isin(["QB", "RB", "WR", "TE"])].copy(); sn["nname"] = sn.player.map(norm)
sn = sn.sort_values("offense_pct", ascending=False).drop_duplicates(["season", "nname", "position"])[["season", "nname", "position", "offense_pct"]].rename(columns={"offense_pct": "snap1"})
f = f.merge(sn, on=["season", "nname", "position"], how="left")
# preseason FFA per-game line
pre = []
for p in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_league", "projections_*_wk0.csv")):
    s_ = int(re.search(r"projections_(\d{4})_wk0", p).group(1)); d = pd.read_csv(p, low_memory=False)
    d = d[d.position.isin(["QB", "RB", "WR", "TE"])].copy(); d["nname"] = d.player.map(norm); d["season"] = s_; d["pre_pg"] = d.points / 17.0
    pre.append(d.drop_duplicates(["nname", "position"])[["season", "nname", "position", "pre_pg"]])
f = f.merge(pd.concat(pre), on=["season", "nname", "position"], how="left")
# DK-implied PPR (props -> points), 2023-25 weeks 1-9
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
mk = pd.read_sql("SELECT season, week, player_id, proj_pts mkt2 FROM weekly_proj_props", con); con.close()
for c in ("season", "week"): mk[c] = pd.to_numeric(mk[c], errors="coerce")
mk = mk.dropna(subset=["season", "week", "player_id"]); mk["season"] = mk.season.astype(int); mk["week"] = mk.week.astype(int)
f = f.merge(mk.drop_duplicates(["player_id", "season", "week"]), on=["player_id", "season", "week"], how="left")

f["res"] = f.actual_ppr - f.baseline_proj
f["played1"] = (f.active1 == 1).astype(float)
f["pts_surp"] = np.where(f.played1 == 1, (f.act1 - f.ffa1).fillna(0), 0.0)
f["ffa_move"] = (f.baseline_proj - f.ffa1).fillna(0.0)
f["pre_gap"] = (f.baseline_proj - f.pre_pg).fillna(0.0)
f["tshare1"] = np.where(f.played1 == 1, f.lag_target_share_l1.fillna(0), 0.0)
f["rshare1"] = np.where(f.played1 == 1, f.rt_route_share_l1.fillna(0), 0.0)
f["snap1"] = np.where(f.played1 == 1, f.snap1.fillna(0), 0.0)
f["car1"] = np.where(f.played1 == 1, f.car1.fillna(0), 0.0)
f["tgt1"] = np.where(f.played1 == 1, f.tgt1.fillna(0), 0.0)
f["wopr1"] = np.where(f.played1 == 1, f.lag_wopr_l1.fillna(0), 0.0)
f["impl"] = f.ctx_implied.fillna(f.ctx_implied.median())
f["impl_c"] = f.impl - f.groupby(["season", "week"]).impl.transform("mean")
f["ffa_lvl"] = f.baseline_proj
FEATS = ["pts_surp", "tshare1", "rshare1", "snap1", "car1", "tgt1", "wopr1", "ffa_move", "pre_gap", "impl_c", "played1", "ffa_lvl"]
USAGE = ["tshare1", "rshare1", "snap1", "car1", "tgt1", "wopr1", "played1", "ffa_lvl"]
PTS = ["pts_surp", "ffa_move", "pre_gap", "played1", "ffa_lvl"]
VEG = ["impl_c", "ffa_lvl", "played1"]
POS = ["QB", "RB", "WR", "TE"]


def metr(d, col):
    e = d.actual_ppr - d[col]
    rho = np.mean([spearmanr(g[col], g.actual_ppr)[0] for _, g in d.groupby(["season", "week"]) if len(g) > 10])
    return dict(mae=e.abs().mean(), rmse=np.sqrt((e ** 2).mean()), rho=rho, bias=-e.mean())


def fit_ridge(tr, te, alpha, feats=FEATS):
    out = pd.Series(0.0, index=te.index)
    for p in POS:
        a, b = tr[tr.position == p], te[te.position == p]
        if len(a) < 60 or len(b) == 0: continue
        mu, sd = a[feats].mean(), a[feats].std().replace(0, 1)
        m = Ridge(alpha=alpha).fit((a[feats] - mu) / sd, a.res - a.res.median())
        out.loc[b.index] = m.predict((b[feats] - mu) / sd)
    return out


def pos_offset(tr, te):
    off = tr.groupby("position").res.median(); return te.position.map(off).fillna(0.0)


SEASONS = list(range(2017, 2026)); per_season = []
played = f[f.actual_active == 1]
for s in SEASONS:
    te = played[(played.season == s) & (played.week == 2)].copy()
    tr2 = played[(played.season < s) & (played.week == 2)]
    trall = played[(played.season < s) & (played.week >= 2)]
    te["ffa"] = te.baseline_proj
    te["ffa_off"] = te.baseline_proj + tr2.res.median()          # global median shade (the incumbent's own lever)
    te["pos_off"] = te.baseline_proj + pos_offset(tr2, te)
    te["pos_off_all"] = te.baseline_proj + pos_offset(trall, te)
    for k in (0.5, 1.0):
        te[f"surp_k{k}"] = te.pos_off + k * fit_ridge(tr2, te, alpha=30.0)
        te[f"surp_all_k{k}"] = te.pos_off_all + k * fit_ridge(trall, te, alpha=30.0)
    te["surp_usage"] = te.pos_off + fit_ridge(tr2, te, alpha=30.0, feats=USAGE)
    te["surp_pts"] = te.pos_off + fit_ridge(tr2, te, alpha=30.0, feats=PTS)
    te["surp_vegas"] = te.pos_off + fit_ridge(tr2, te, alpha=30.0, feats=VEG)
    per_season.append(te)
P = pd.concat(per_season)
P["inc"] = P.inc_full.fillna(P.ffa); P["inc23"] = P.inc_2023.fillna(P.inc)   # 2023 rows: inc_2023 undefined (min_train) -> inc_full

# ---------- market blend, 2023-25 week 2 ----------
M = played[(played.season >= 2023) & (played.week <= 9) & played.mkt2.notna()].copy()
mrows = []
for s in (2024, 2025):
    te = P[(P.season == s) & P.mkt2.notna()].copy()
    for lab, trm in (("w_all19", M[(M.season < s)]), ("w_wk2", M[(M.season < s) & (M.week == 2)])):
        best = min(np.arange(0, 1.01, 0.05), key=lambda w: (trm.actual_ppr - (w * trm.mkt2 + (1 - w) * trm.baseline_proj)).abs().mean())
        te[f"mkt_{lab}"] = best * te.mkt2 + (1 - best) * te.baseline_proj + te.pos_off - te.baseline_proj
        te[f"mkt_{lab}_w"] = best
    te["mkt_w0.5"] = 0.5 * te.mkt2 + 0.5 * te.baseline_proj + te.pos_off - te.baseline_proj
    te["mkt_w0.5_surp"] = te["mkt_w0.5"] + (te["surp_k0.5"] - te.pos_off)
    mrows.append(te)
MP = pd.concat(mrows)

# ---------- report ----------
L = ["# Week-2-only tweak study\n",
     f"Played week-2 rows, walk-forward by season (test seasons {SEASONS[0]}-{SEASONS[-1]}, n = {len(P):,}). Train = earlier seasons' week-2 rows unless the label says `_all` (weeks 2-18). "
     "Incumbent = the Model_Burke package refit on this frame (played rows, FFA baseline, no market), `inc` = trained from 2016, `inc23` = trained from 2023 (production window; 2023 rows fall back to `inc`).\n"]
cols = ["ffa", "ffa_off", "inc", "inc23", "pos_off", "pos_off_all", "surp_k0.5", "surp_k1.0", "surp_all_k0.5", "surp_all_k1.0", "surp_usage", "surp_pts", "surp_vegas"]
T = pd.DataFrame({c: metr(P, c) for c in cols}).T
L.append("## All seasons pooled\n\n| model | MAE | RMSE | Spearman | bias |\n|---|---|---|---|---|")
for c, r in T.iterrows(): L.append(f"| {c} | {r.mae:.3f} | {r.rmse:.3f} | {r.rho:.3f} | {r.bias:+.2f} |")
L.append("\n## Per-season MAE (week 2)\n")
ps = P.groupby("season").apply(lambda g: pd.Series({c: (g.actual_ppr - g[c]).abs().mean() for c in cols}))
L.append("| season | n | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
for s, r in ps.iterrows(): L.append(f"| {s} | {int((P.season == s).sum())} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
L.append("\n## Paired tests vs the incumbent (per-season MAE differences, negative = better)\n\n| model | mean dMAE | seasons better | t-test p (9 seasons) | row-level sign p |\n|---|---|---|---|---|")
for c in cols:
    if c == "inc": continue
    d = ps[c] - ps["inc"]; t = ttest_rel(ps[c], ps["inc"])
    e1 = (P.actual_ppr - P[c]).abs(); e0 = (P.actual_ppr - P.inc).abs(); w = int((e1 < e0).sum()); n = int((e1 != e0).sum())
    L.append(f"| {c} | {d.mean():+.3f} | {int((d < 0).sum())}/{len(d)} | {t.pvalue:.3f} | {binomtest(w, n).pvalue:.3f} ({w}/{n}) |")
L.append("\n## What the week-2 surprise ridge learned (fit on every week-2 row 2016-2025, standardised coefficients, PPR per 1 sd)\n")
tr = played[played.week == 2]
L.append("| feature | " + " | ".join(POS) + " |\n|---|" + "---|" * len(POS))
co = {}
for p in POS:
    a = tr[tr.position == p]; mu, sd = a[FEATS].mean(), a[FEATS].std().replace(0, 1)
    co[p] = pd.Series(Ridge(alpha=30.0).fit((a[FEATS] - mu) / sd, a.res - a.res.median()).coef_, index=FEATS)
for ft in FEATS: L.append(f"| {ft} | " + " | ".join(f"{co[p][ft]:+.2f}" for p in POS) + " |")
L.append("\n## Week-2 position offsets by season (median residual actual - FFA, played rows)\n")
po = played[played.week == 2].groupby(["season", "position"]).res.median().unstack().round(2)
L.append("| season | " + " | ".join(po.columns) + " |\n|---|" + "---|" * len(po.columns))
for s, r in po.iterrows(): L.append(f"| {s} | " + " | ".join(f"{r[c]:+.2f}" for c in po.columns) + " |")
po5 = played[played.week >= 5].groupby("position").res.median().round(2)
L.append("\nWeeks 5+ position medians for reference: " + ", ".join(f"{p} {po5[p]:+.2f}" for p in po5.index) + "\n")
L.append("## DraftKings-implied points at week 2 (2024-25 test, rows with a DK line)\n")
mc = ["ffa", "inc", "pos_off", "surp_k0.5", "mkt_w_all19", "mkt_w_wk2", "mkt_w0.5", "mkt_w0.5_surp"]
L.append(f"n = {len(MP)}; fitted DK weights: all weeks 1-9 -> {MP.groupby('season').mkt_w_all19_w.first().to_dict()}, week 2 only -> {MP.groupby('season').mkt_w_wk2_w.first().to_dict()}\n")
L.append("| model | MAE | RMSE | Spearman | bias |\n|---|---|---|---|---|")
for c in mc:
    r = metr(MP, c); L.append(f"| {c} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['rho']:.3f} | {r['bias']:+.2f} |")
L.append("\nPer season:\n\n| season | n | " + " | ".join(mc) + " |\n|---|---|" + "---|" * len(mc))
for s, g in MP.groupby("season"): L.append(f"| {s} | {len(g)} | " + " | ".join(f"{(g.actual_ppr - g[c]).abs().mean():.3f}" for c in mc) + " |")
cov = P[P.season >= 2023].groupby("position").mkt2.apply(lambda s: s.notna().mean()).round(2).to_dict()
L.append(f"\nDK line coverage among played week-2 rows 2023-25 by position: {cov}\n")
os.makedirs(os.path.dirname(A.out), exist_ok=True); open(A.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
P.to_parquet(os.path.join(A.scratch, "week2_preds.parquet")); MP.to_parquet(os.path.join(A.scratch, "week2_mkt_preds.parquet"))
print("\n".join(L))
