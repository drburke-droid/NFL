"""Anatomy of the largest misses (2017-2025, played rows, residual = actual - incumbent).

  1. what a big miss is: the top 5% of |residual| each season, split into booms (actual >> model) and
     busts (actual << model); size, direction, share of total absolute error
  2. who they are: position / tier / week / Q-tag / volatility / experience composition vs the base
     rate (lift), and repeat offenders vs what their volatility predicts
  3. what happened on the day: booms decomposed into touchdown points vs volume vs efficiency;
     busts into volume collapse vs early exit (snap share vs the player's norm) vs efficiency
  4. can they be seen coming: walk-forward classifier for P(big miss) from pre-game features, AUC by
     season, and the big-miss rate by the strongest single flags; 2023+ market disagreement

Usage: python scripts/big_miss_anatomy.py --scratch DIR  (frame_full.parquet + incumbent_preds.parquet)
"""
import os, re, glob, argparse, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True)
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "big_miss_anatomy.md"))
A = ap.parse_args()


def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)


f = pd.read_parquet(os.path.join(A.scratch, "frame_full.parquet")).join(pd.read_parquet(os.path.join(A.scratch, "incumbent_preds.parquet")))
f = f[(f.actual_active == 1) & f.inc_full.notna() & (f.season >= 2017)].copy()
f["inc"] = f.inc_full; f["r"] = f.actual_ppr - f.inc; f["ar"] = f.r.abs()
# snap share for every week (nflverse), by name + position + season + week
sn = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(ROOT, "data", "nflverse_cache", "snap_counts_*.parquet"))])
sn = sn[(sn.game_type == "REG") & sn.position.isin(["QB", "RB", "WR", "TE"])].copy(); sn["nname"] = sn.player.map(norm)
sn = sn.sort_values("offense_pct", ascending=False).drop_duplicates(["season", "week", "nname", "position"])[["season", "week", "nname", "position", "offense_pct"]]
f = f.merge(sn, on=["season", "week", "nname", "position"], how="left")
g = f.sort_values(["player_id", "season", "week"]).groupby("player_id").offense_pct
f["snap_norm"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
f["snap_ratio"] = f.offense_pct / f.snap_norm
# points decomposition: yards + receptions are observable; the rest (TDs, 2pt, fumbles, INT) is the residual bucket
f["yard_pts"] = 0.1 * (f.actual_rec_yds + f.actual_rush_yds) + 0.04 * f.actual_pass_yds + f.actual_receptions
f["td_pts"] = f.actual_ppr - f.yard_pts
f["touches"] = f.actual_targets + f.actual_carries + f.actual_attempts
f["touch_norm"] = (f.lag_targets_r3.fillna(0) + f.lag_carries_r3.fillna(0) + f.lag_attempts_r3.fillna(0))
f["tier"] = f.groupby(["season", "week", "position"]).baseline_proj.rank(ascending=False, method="first")
f["tier_b"] = pd.cut(f.tier, [0, 6, 12, 24, 36, 999], labels=["top6", "7-12", "13-24", "25-36", "37+"]).astype(str)
f["wk_b"] = np.where(f.week <= 2, "wk1-2", np.where(f.week <= 4, "wk3-4", np.where(f.week <= 13, "wk5-13", "wk14+")))
f["q"] = np.where(f.ffa_injury_q == 1, "Q", "not Q")
f["vol"] = np.where(f.lag_fantasy_points_ppr_std3.isna(), "no history", pd.qcut(f.lag_fantasy_points_ppr_std3.rank(method="first"), 3, labels=["steady", "mid", "volatile"]).astype(str))
f["exp"] = np.where(f.lag_games_played.fillna(0) < 6, "new", "established")
f["proj_b"] = pd.cut(f.baseline_proj, [-1, 5, 10, 15, 20, 99], labels=["<5", "5-10", "10-15", "15-20", "20+"]).astype(str)

# ---------- 1. what a big miss is ----------
thr = f.groupby("season").ar.quantile(0.95); f["big"] = f.ar >= f.season.map(thr)
f["kind"] = np.where(~f.big, "normal", np.where(f.r > 0, "boom", "bust"))
B = f[f.big]
L = ["# Anatomy of the largest misses (2017-2025)\n",
     f"Played rows with an incumbent prediction: {len(f):,}. A big miss = top 5% of |actual - model| within each season "
     f"(threshold {thr.min():.1f}-{thr.max():.1f} points). {len(B):,} big misses: {int((B.r > 0).sum()):,} booms (actual above the model) and {int((B.r < 0).sum()):,} busts.\n",
     f"- They are {len(B) / len(f):.1%} of rows and {B.ar.sum() / f.ar.sum():.1%} of all absolute error; mean size {B.ar.mean():.1f} points (booms {B[B.r > 0].ar.mean():.1f}, busts {B[B.r < 0].ar.mean():.1f}).",
     f"- Booms outnumber busts {(B.r > 0).mean():.0%} to {(B.r < 0).mean():.0%}: the right tail is where the big errors live. The biggest single misses: "
     + "; ".join(f"{r.nname} {int(r.season)} wk{int(r.week)} {r.actual_ppr:.0f} vs {r.inc:.0f}" for r in B.sort_values("ar", ascending=False).head(5).itertuples()) + ".",
     f"- FFA misses the same rows: the incumbent and FFA share {(f[f.big].ar > 0).mean():.0%} of these (|actual - FFA| >= threshold on {((f.actual_ppr - f.baseline_proj).abs() >= f.season.map(thr))[f.big].mean():.0%} of them).\n"]

# ---------- 2. who they are ----------
L.append("## Who the big misses are (share of big misses vs share of all rows; lift > 1 = over-represented)\n")
for lab, col in (("position", "position"), ("projection", "proj_b"), ("tier within position", "tier_b"), ("week", "wk_b"), ("injury tag", "q"), ("recent volatility", "vol"), ("experience", "exp")):
    base = f[col].value_counts(normalize=True); boom = f[f.kind == "boom"][col].value_counts(normalize=True); bust = f[f.kind == "bust"][col].value_counts(normalize=True)
    order = [x for x in (["QB", "RB", "WR", "TE"] if col == "position" else ["<5", "5-10", "10-15", "15-20", "20+"] if col == "proj_b" else ["top6", "7-12", "13-24", "25-36", "37+"] if col == "tier_b" else ["wk1-2", "wk3-4", "wk5-13", "wk14+"] if col == "wk_b" else base.index) if x in base.index]
    L.append(f"**{lab}**\n\n| {lab} | rows | boom lift | bust lift | big-miss rate |\n|---|---|---|---|---|")
    for x in order:
        rate = f[f[col] == x].big.mean()
        L.append(f"| {x} | {base[x]:.0%} | {boom.get(x, 0) / base[x]:.2f} | {bust.get(x, 0) / base[x]:.2f} | {rate:.1%} |")
    L.append("")
# repeat offenders: players with the most big misses vs expected from their row count x position-tier rate
exp_rate = f.groupby(["position", "proj_b"]).big.mean()
f["exp_big"] = pd.MultiIndex.from_frame(f[["position", "proj_b"]]).map(exp_rate).values if hasattr(pd.MultiIndex.from_frame(f[["position", "proj_b"]]), "map") else f.set_index(["position", "proj_b"]).index.map(exp_rate).values
pl = f.groupby(["player_id", "nname"]).agg(n=("big", "size"), big=("big", "sum"), exp=("exp_big", "sum"), boom=("kind", lambda s: (s == "boom").sum())).reset_index()
pl = pl[pl.n >= 30]; pl["excess"] = pl.big - pl.exp; pl["rate"] = pl.big / pl.n
L.append("**Repeat offenders** (players with 30+ rows, ranked by big misses above what their projection level predicts):\n\n| player | rows | big misses | expected | of which booms |\n|---|---|---|---|---|")
for r in pl.sort_values("excess", ascending=False).head(10).itertuples(): L.append(f"| {r.nname} | {r.n} | {int(r.big)} | {r.exp:.1f} | {int(r.boom)} |")
# is a player's big-miss rate persistent season to season?
ps = f.groupby(["player_id", "season"]).agg(n=("big", "size"), rate=("big", "mean")).reset_index(); ps = ps[ps.n >= 8]
ps["prev"] = ps.sort_values("season").groupby("player_id").rate.shift(1); pp = ps.dropna(subset=["prev"])
L.append(f"\nSeason-to-season correlation of a player's big-miss rate (players with 8+ games both seasons, n = {len(pp):,}): {np.corrcoef(pp.prev, pp.rate)[0, 1]:+.2f}. "
         f"Players in the top volatility tercile last season have a big-miss rate of {f[f.vol == 'volatile'].big.mean():.1%} vs {f[f.vol == 'steady'].big.mean():.1%} for the steadiest.\n")

# ---------- 3. what happened on the day ----------
L.append("## What happened on the day\n")
bo = f[f.kind == "boom"]; bu = f[f.kind == "bust"]; nm = f[f.kind == "normal"]
L.append("**Booms** (actual far above the model): where the extra points came from, per row, vs a normal row of the same projection band.\n")
L.append("| projection | n booms | actual | model | TD+other pts (boom) | TD+other pts (normal) | touches (boom) | touches (normal) | yards/touch (boom) | (normal) |\n|---|---|---|---|---|---|---|---|---|---|")
for b in ["<5", "5-10", "10-15", "15-20", "20+"]:
    x = bo[bo.proj_b == b]; y = nm[nm.proj_b == b]
    if len(x) < 20: continue
    ypt = lambda d: ((d.actual_rec_yds + d.actual_rush_yds + 0.4 * d.actual_pass_yds) / d.touches.replace(0, np.nan)).median()
    L.append(f"| {b} | {len(x)} | {x.actual_ppr.mean():.1f} | {x.inc.mean():.1f} | {x.td_pts.mean():.1f} | {y.td_pts.mean():.1f} | {x.touches.mean():.1f} | {y.touches.mean():.1f} | {ypt(x):.1f} | {ypt(y):.1f} |")
share_td = ((bo.td_pts - bo.groupby("proj_b").td_pts.transform(lambda s: nm.td_pts[nm.proj_b == s.name].mean() if False else 0)) ).mean()
td_excess = (bo.td_pts.values - bo.proj_b.map(nm.groupby("proj_b").td_pts.mean()).values)
L.append(f"\nAcross all booms the excess over a normal row of the same band is {bo.r.mean():.1f} points, of which touchdowns/other account for {td_excess.mean():.1f} ({td_excess.mean() / bo.r.mean():.0%}); the rest is volume and yardage. "
         f"{(bo.td_pts >= 12).mean():.0%} of booms had two or more touchdowns' worth of TD points.\n")
L.append("**Busts** (actual far below the model): volume collapse, early exit, or efficiency?\n")
L.append("| projection | n busts | model | actual | touches (bust) | touches (normal) | snap share vs own norm (bust) | (normal) | left early (<50% of norm) | zero-touch |\n|---|---|---|---|---|---|---|---|---|---|")
for b in ["5-10", "10-15", "15-20", "20+"]:
    x = bu[bu.proj_b == b]; y = nm[nm.proj_b == b]
    if len(x) < 20: continue
    L.append(f"| {b} | {len(x)} | {x.inc.mean():.1f} | {x.actual_ppr.mean():.1f} | {x.touches.mean():.1f} | {y.touches.mean():.1f} | {x.snap_ratio.median():.2f} | {y.snap_ratio.median():.2f} | {(x.snap_ratio < 0.5).mean():.0%} | {(x.touches <= 2).mean():.0%} |")
L.append(f"\nBusts with a snap-share record: {bu.snap_ratio.notna().mean():.0%}. Of those, {(bu.snap_ratio < 0.5).mean():.0%} played under half their usual snaps (an in-game injury or a benching the pre-game data could not see) and {(bu.snap_ratio.between(0.5, 0.9)).mean():.0%} were between 50-90%; {(bu.snap_ratio >= 0.9).mean():.0%} played a full complement and simply produced nothing.\n")

# ---------- 4. can they be seen coming ----------
L.append("## Can a big miss be seen coming?\n")
FEATS = ["baseline_proj", "lag_fantasy_points_ppr_std3", "lag_fantasy_points_ppr_r3", "ffa_injury_q", "week", "ctx_spread", "ctx_total", "ctx_home",
         "lag_games_played", "rt_deep_share_r6", "rt_route_share_r3", "lag_target_share_r3", "ffa_rec_yds_sd", "ffa_rush_yds_sd", "ffa_pass_yds_sd", "snap_norm"]
rows = []
for s in range(2019, 2026):
    tr = f[f.season < s]; te = f[f.season == s]
    med = tr[FEATS].median(); X = tr[FEATS].fillna(med).fillna(0); Xt = te[FEATS].fillna(med).fillna(0)
    mu, sd = X.mean(), X.std().replace(0, 1)
    for tgt, lab in (("big", "any big miss"), ("boom", "boom"), ("bust", "bust")):
        y = (tr.kind == tgt) if tgt != "big" else tr.big; yt = (te.kind == tgt) if tgt != "big" else te.big
        m = LogisticRegression(C=0.3, max_iter=500).fit((X - mu) / sd, y); p = m.predict_proba((Xt - mu) / sd)[:, 1]
        base = LogisticRegression(C=0.3, max_iter=500).fit(((X - mu) / sd)[["baseline_proj"]], y).predict_proba(((Xt - mu) / sd)[["baseline_proj"]])[:, 1]
        rows.append({"season": s, "target": lab, "auc": roc_auc_score(yt, p), "auc_proj_only": roc_auc_score(yt, base), "top-decile rate": yt[p >= np.quantile(p, 0.9)].mean(), "base rate": yt.mean()})
R = pd.DataFrame(rows)
L.append("Walk-forward logistic regression on pre-game features (projection, volatility, recent points, Q-tag, week, spread/total/home, experience, deep share, route share, target share, FFA stat spreads, usual snap share), 2019-2025:\n")
L.append("| target | AUC (all features) | AUC (projection only) | big-miss rate in the model's top decile | base rate |\n|---|---|---|---|---|")
for lab, x in R.groupby("target"): L.append(f"| {lab} | {x.auc.mean():.3f} | {x.auc_proj_only.mean():.3f} | {x['top-decile rate'].mean():.1%} | {x['base rate'].mean():.1%} |")
# strongest single flags
L.append("\nBig-miss rate by the strongest single pre-game flags (all rows):\n\n| flag | rows | big-miss rate | boom rate | bust rate |\n|---|---|---|---|---|")
flags = [("projection 20+", f.baseline_proj >= 20), ("projection 15-20", f.baseline_proj.between(15, 20)), ("projection < 5", f.baseline_proj < 5),
         ("volatile (top tercile std3)", f.vol == "volatile"), ("steady (bottom tercile)", f.vol == "steady"), ("Q tag", f.q == "Q"), ("weeks 1-2", f.week <= 2),
         ("game total 50+", f.ctx_total >= 50), ("game total < 40", f.ctx_total < 40), ("spread |7|+ underdog", f.ctx_spread >= 7), ("favourite by 7+", f.ctx_spread <= -7),
         ("new (< 6 games)", f.exp == "new"), ("usual snap share < 50%", f.snap_norm < 0.5), ("usual snap share 90%+", f.snap_norm >= 0.9)]
for lab, m in flags:
    x = f[m.fillna(False)]
    if len(x) < 200: continue
    L.append(f"| {lab} | {len(x):,} | {x.big.mean():.1%} | {(x.kind == 'boom').mean():.1%} | {(x.kind == 'bust').mean():.1%} |")
# market disagreement 2023+
mk = f[(f.season >= 2023) & f.mkt_rec_yds_line.notna() & f.ffa_rec_yds.notna()].copy()
if len(mk) > 500:
    mk["gap"] = mk.mkt_rec_yds_line - mk.ffa_rec_yds; mk["gap_b"] = pd.cut(mk.gap, [-99, -10, -4, 4, 10, 99], labels=["DK 10+ under FFA", "DK 4-10 under", "agree", "DK 4-10 over", "DK 10+ over FFA"]).astype(str)
    L.append(f"\nMarket disagreement (2023-25 rows with a DK receiving-yards line, n = {len(mk):,}): big-miss rate and which way the miss went.\n\n| DK line vs FFA rec yds | rows | big-miss rate | boom | bust | mean residual vs model |\n|---|---|---|---|---|---|")
    for b in ["DK 10+ under FFA", "DK 4-10 under", "agree", "DK 4-10 over", "DK 10+ over FFA"]:
        x = mk[mk.gap_b == b]
        if len(x) < 30: continue
        L.append(f"| {b} | {len(x):,} | {x.big.mean():.1%} | {(x.kind == 'boom').mean():.1%} | {(x.kind == 'bust').mean():.1%} | {x.r.mean():+.2f} |")
os.makedirs(os.path.dirname(A.out), exist_ok=True); open(A.out, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
