"""Segment heterogeneity study (Sept 2026): does the projection error have structure by team, player
archetype, tier, context, or player that a segment-specific correction could exploit?

Every scheme is tested the same way so the number of segments cannot flatter it:
  * residual r = actual - incumbent (Model_Burke refit on the frame; FFA baseline + correction),
    played rows 2017-2025 (the incumbent needs a season of burn-in).
  * in-sample screen: between-segment variance of r vs a permutation null (segment labels shuffled
    within season) -> one p-value per scheme, so 40 schemes do not manufacture a hit.
  * out-of-sample test: empirical-Bayes shrunk segment offsets (James-Stein: m_g * n_g/(n_g+k),
    k = within/between variance) fitted on earlier seasons, applied to the test season, walk-forward
    2018-2025. dMAE vs the incumbent, seasons better, paired t over seasons. Pure noise shrinks to ~0
    and scores ~0.000; a real segment effect scores negative.
  * within-season schemes (team / player running hot or cold vs the model) are walk-forward by week.
  * blend heterogeneity (2023-25 rows with DK stat lines): the per-stat DK weight fitted globally vs
    per position x tier vs per star/non-star, walk-forward by season.

Usage: python scripts/segment_hetero_study.py --scratch DIR  (frame_full.parquet + incumbent_preds.parquet)
"""
import os, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True)
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "segment_hetero_study.md"))
ap.add_argument("--perms", type=int, default=200)
A = ap.parse_args()
rng = np.random.default_rng(17)

f = pd.read_parquet(os.path.join(A.scratch, "frame_full.parquet")).join(pd.read_parquet(os.path.join(A.scratch, "incumbent_preds.parquet")))
f = f[(f.actual_active == 1) & f.inc_full.notna() & (f.season >= 2017)].copy()
f["inc"] = f.inc_full; f["r"] = f.actual_ppr - f.inc; f["r_ffa"] = f.actual_ppr - f.baseline_proj
f["early"] = np.where(f.week <= 2, "wk1-2", np.where(f.week <= 4, "wk3-4", "wk5+"))
f["tier"] = f.groupby(["season", "week", "position"]).baseline_proj.rank(ascending=False, method="first")
f["tier_b"] = pd.cut(f.tier, [0, 6, 12, 24, 36, 999], labels=["top6", "7-12", "13-24", "25-36", "37+"]).astype(str)
f["star"] = np.where(f.tier <= {"QB": 6, "RB": 12, "WR": 12, "TE": 6}.get("WR", 12), "star", "rest")
f["star"] = np.where(f.tier <= f.position.map({"QB": 6, "RB": 12, "WR": 12, "TE": 6}), "star", "rest")
f["fav"] = np.where(f.ctx_spread.isna(), "na", np.where(f.ctx_spread < -3, "fav", np.where(f.ctx_spread > 3, "dog", "pickem")))
f["tot_b"] = np.where(f.ctx_total.isna(), "na", np.where(f.ctx_total >= 48, "hi", np.where(f.ctx_total <= 42, "lo", "mid")))
f["home"] = np.where(f.ctx_home == 1, "home", "away")
f["qtag"] = np.where(f.ffa_injury_q == 1, "Q", "ok")
rec_share = f.lag_targets_r6 / (f.lag_targets_r6 + f.lag_carries_r6).replace(0, np.nan)
f["rb_arch"] = np.where(f.position != "RB", "na", pd.cut(rec_share, [-0.01, 0.15, 0.3, 1.01], labels=["ground", "mixed", "pass-catch"]).astype(str))
f["wr_deep"] = np.where(f.position != "WR", "na", pd.qcut(f.rt_deep_share_r6.rank(method="first"), 3, labels=["short", "mid", "deep"]).astype(str) if f.rt_deep_share_r6.notna().any() else "na")
f["qb_scr"] = np.where(f.position != "QB", "na", pd.qcut(f.qb_scramble_rate_r6.rank(method="first"), 3, labels=["pocket", "mid", "scrambler"]).astype(str))
f["exp"] = np.where(f.lag_games_played.fillna(0) < 6, "new(<6 gms)", np.where(f.lag_games_played < 30, "mid", "vet"))
f["vol"] = np.where(f.lag_fantasy_points_ppr_std3.isna(), "na", pd.qcut(f.lag_fantasy_points_ppr_std3.rank(method="first"), 3, labels=["steady", "mid", "volatile"]).astype(str))
f["opp_press"] = np.where(f.opp_press_r6_z.isna(), "na", np.where(f.opp_press_r6_z > 0.5, "hi-press", np.where(f.opp_press_r6_z < -0.5, "lo-press", "mid")))

SCHEMES = {
    "team": ["team"], "team x early": ["team", "early"], "team x pos": ["team", "position"],
    "opponent": ["opponent_team"], "opponent x pos": ["opponent_team", "position"],
    "pos x tier": ["position", "tier_b"], "pos x tier x early": ["position", "tier_b", "early"],
    "pos x star x early": ["position", "star", "early"],
    "pos x early": ["position", "early"],
    "RB archetype": ["rb_arch"], "RB archetype x early": ["rb_arch", "early"],
    "WR deep-share": ["wr_deep"], "QB scramble": ["qb_scr"],
    "pos x experience": ["position", "exp"], "pos x volatility": ["position", "vol"],
    "pos x home": ["position", "home"], "pos x favourite": ["position", "fav"], "pos x total": ["position", "tot_b"],
    "pos x Q-tag": ["position", "qtag"], "pos x opp pressure": ["position", "opp_press"],
    "pos x tier x favourite": ["position", "tier_b", "fav"], "pos x tier x total": ["position", "tier_b", "tot_b"],
    "player (cross-season)": ["player_id"], "player x early": ["player_id", "early"],
    "team x season-phase x pos": ["team", "early", "position"],
}


def eb_offsets(tr, keys):
    """James-Stein shrunk segment means of the residual."""
    g = tr.groupby(keys).r.agg(["mean", "count"])
    if len(g) < 2: return pd.Series(dtype=float), 0.0
    within = tr.r.var()
    between = max(((g["mean"] - tr.r.mean()) ** 2 * g["count"]).sum() / g["count"].sum() - within / g["count"].mean(), 0.0)
    k = within / between if between > 0 else np.inf
    shrink = g["count"] / (g["count"] + k) if np.isfinite(k) else g["count"] * 0.0
    med = tr.groupby(keys).r.median() - tr.r.median()      # MAE is minimised by the median: shade segments by their median, not their mean
    return {"mean": g["mean"] * shrink, "med": med * shrink}, (0.0 if not np.isfinite(k) else between / (between + within / g["count"].median()))


def apply_off(te, off, keys):
    if len(off) == 0: return pd.Series(0.0, index=te.index)
    idx = pd.MultiIndex.from_frame(te[keys]) if len(keys) > 1 else te[keys[0]]
    return pd.Series(off.reindex(idx).fillna(0.0).values, index=te.index)


def perm_p(d, keys, nperm):
    """between-segment variance explained vs labels shuffled within season."""
    def bss(x):
        g = x.groupby(keys).r.agg(["mean", "count"]); return float(((g["mean"] - x.r.mean()) ** 2 * g["count"]).sum())
    obs = bss(d); null = []
    for _ in range(nperm):
        x = d.copy()
        for k in keys:
            if k != "position": x[k] = x.groupby("season")[k].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        null.append(bss(x))
    null = np.array(null); return obs, float((null >= obs).mean()), float(null.mean())


SEASONS = list(range(2018, 2026)); L = ["# Segment heterogeneity study\n",
    f"Played rows 2017-2025 with an incumbent prediction (n = {len(f):,}); residual = actual - incumbent. Out-of-sample = EB-shrunk segment offsets fitted on earlier seasons, applied walk-forward {SEASONS[0]}-{SEASONS[-1]}. "
    "Perm p = probability that shuffled labels (within season) explain as much residual variance as the real labels; shrink = how much of a typical segment mean survives shrinkage (0 = nothing but noise).\n",
    "## Cross-season schemes\n", "| scheme | segments | perm p | var explained (obs / null, pts^2) | shrink | OOS dMAE vs inc (median offsets) | seasons better | t p | OOS dRMSE (mean offsets) | biggest surviving median offsets (train = all seasons) |", "|---|---|---|---|---|---|---|---|---|---|"]
res_rows = []
for name, keys in SCHEMES.items():
    d = f.dropna(subset=keys)
    obs, p, nullm = perm_p(d, keys, A.perms if "player" not in name else 40)
    preds = []
    for s in SEASONS:
        tr, te = d[d.season < s], d[d.season == s].copy()
        off, shr = eb_offsets(tr, keys); te["adj"] = te.inc + apply_off(te, off["med"], keys); te["adj_mean"] = te.inc + apply_off(te, off["mean"], keys); preds.append(te)
    P = pd.concat(preds); e1 = (P.actual_ppr - P.adj).abs(); e0 = (P.actual_ppr - P.inc).abs()
    ps = pd.DataFrame({"a": e1, "b": e0, "s": P.season}).groupby("s").mean(); dd = ps.a - ps.b
    rs = pd.DataFrame({"a": (P.actual_ppr - P.adj_mean) ** 2, "b": (P.actual_ppr - P.inc) ** 2, "s": P.season}).groupby("s").mean() ** 0.5; dr = rs.a - rs.b
    offs, shr_all = eb_offsets(d, keys); off_all = offs["med"]
    top = off_all.reindex(off_all.abs().sort_values(ascending=False).index).head(4)
    topstr = "; ".join(f"{'/'.join(map(str, i)) if isinstance(i, tuple) else i} {v:+.2f}" for i, v in top.items()) if len(top) else ""
    if "player" in name and len(top):
        nm = d.drop_duplicates("player_id").set_index("player_id").nname
        topstr = "; ".join(f"{nm.get(i[0] if isinstance(i, tuple) else i, '?')}{('/' + i[1]) if isinstance(i, tuple) else ''} {v:+.2f}" for i, v in top.items())
    L.append(f"| {name} | {d.groupby(keys).ngroups} | {p:.3f} | {obs/len(d):.3f} / {nullm/len(d):.3f} | {shr_all:.2f} | {dd.mean():+.4f} | {int((dd < 0).sum())}/{len(dd)} | {ttest_rel(ps.a, ps.b).pvalue:.2f} | {dr.mean():+.4f} ({int((dr < 0).sum())}/{len(dr)}) | {topstr} |")
    res_rows.append((name, dd.mean(), p))

# ---------- within-season running offsets (hot / cold vs the model), walk-forward by week ----------
L += ["\n## Within-season running offsets (walk-forward by week; offset = EB-shrunk mean residual of the segment in the season's earlier weeks)\n",
      "| scheme | min prior weeks | OOS dMAE vs inc | seasons better | t p |", "|---|---|---|---|---|"]
for name, keys, minw in (("team, this season", ["team"], 2), ("team, this season", ["team"], 4), ("player, this season", ["player_id"], 2), ("player, this season", ["player_id"], 4),
                         ("team x pos, this season", ["team", "position"], 3), ("player, prior + this season", ["player_id"], 0)):
    preds = []
    for s in range(2017, 2026):
        d = f[f.season == s]
        for w in sorted(d.week.unique()):
            te = d[d.week == w].copy()
            tr = d[d.week < w] if "prior" not in name else f[(f.season < s) | ((f.season == s) & (f.week < w))]
            if tr.groupby("week").ngroups < minw or (tr.empty): te["adj"] = te.inc
            else:
                off, _ = eb_offsets(tr, keys); te["adj"] = te.inc + apply_off(te, off["med"], keys)
            preds.append(te)
    P = pd.concat(preds); P = P[P.season >= 2018]
    ps = pd.DataFrame({"a": (P.actual_ppr - P.adj).abs(), "b": (P.actual_ppr - P.inc).abs(), "s": P.season}).groupby("s").mean(); dd = ps.a - ps.b
    L.append(f"| {name} | {minw} | {dd.mean():+.4f} | {int((dd < 0).sum())}/{len(dd)} | {ttest_rel(ps.a, ps.b).pvalue:.2f} |")

# ---------- the two named examples, in-sample, for the record ----------
L += ["\n## The two examples, for the record (in-sample means of actual - incumbent)\n"]
cin = f[(f.team == "CIN") & (f.week <= 2)]
L.append(f"- Bengals weeks 1-2, all seasons: mean residual {cin.r.mean():+.2f} on {len(cin)} rows; by season " + ", ".join(f"{s} {g.r.mean():+.1f} (n={len(g)})" for s, g in cin.groupby("season")))
tm2 = f[f.week <= 2].groupby(["team"]).r.agg(["mean", "count"]); tm2 = tm2[tm2["count"] >= 60]
L.append(f"- Every team's weeks 1-2 mean residual is inside {tm2['mean'].min():+.2f} .. {tm2['mean'].max():+.2f}; the EB shrinkage on team x early keeps {eb_offsets(f[f.week <= 2], ['team'])[1]:.0%} of a typical team mean.")
st = f.groupby(["position", "star"]).apply(lambda g: pd.Series({"n": len(g), "inc MAE": (g.actual_ppr - g.inc).abs().mean(), "FFA MAE": (g.actual_ppr - g.baseline_proj).abs().mean(), "bias vs inc": g.r.mean()}))
L.append("- Stars vs the rest (star = top-6 QB/TE, top-12 RB/WR by FFA that week):\n\n| pos | group | n | inc MAE | FFA MAE | mean residual |\n|---|---|---|---|---|---|")
for (p, s), r in st.iterrows(): L.append(f"| {p} | {s} | {int(r.n)} | {r['inc MAE']:.3f} | {r['FFA MAE']:.3f} | {r['bias vs inc']:+.2f} |")

# ---------- blend heterogeneity: DK weight by segment (2023-25, rows with a DK stat line) ----------
L += ["\n## DraftKings weight heterogeneity (2023-25 played rows with a DK line; PPR rebuilt from FFA stats with the DK line blended per stat)\n"]
g = pd.read_parquet(os.path.join(A.scratch, "frame_full.parquet")); g = g[(g.actual_active == 1) & (g.season >= 2023)].copy()
for mk, col in (("reception_yds", "l_rec_yds"), ("receptions", "l_rec"), ("rush_yds", "l_rush_yds")):
    p = pd.read_parquet(os.path.join(ROOT, "data", "props_frames", f"props_player_{mk}.parquet")).dropna(subset=["player_id"]).drop_duplicates(["player_id", "season", "week"])
    g = g.merge(p[["player_id", "season", "week", "baseline_proj"]].rename(columns={"baseline_proj": col}), on=["player_id", "season", "week"], how="left")
g = g[g[["l_rec_yds", "l_rec", "l_rush_yds"]].notna().any(axis=1)].copy()
g["tier"] = g.groupby(["season", "week", "position"]).baseline_proj.rank(ascending=False, method="first")
g["star"] = np.where(g.tier <= g.position.map({"QB": 6, "RB": 12, "WR": 12, "TE": 6}), "star", "rest")
g["tier_b"] = pd.cut(g.tier, [0, 6, 12, 24, 36, 999], labels=["top6", "7-12", "13-24", "25-36", "37+"]).astype(str)
g["early"] = np.where(g.week <= 2, "wk1-2", "wk3+")
z = lambda c: g[c].fillna(0)
base = z("ffa_pass_yds") * .04 + z("ffa_pass_tds") * 4 - z("ffa_pass_int") * 2 + z("ffa_rush_tds") * 6 + z("ffa_rec_tds") * 6 - z("ffa_fumbles_lost") * 2
def ppr_w(d, wy, wr):
    ry = np.where(d.l_rec_yds.notna(), wy * d.l_rec_yds + (1 - wy) * d.ffa_rec_yds.fillna(0), d.ffa_rec_yds.fillna(0))
    ru = np.where(d.l_rush_yds.notna(), wy * d.l_rush_yds + (1 - wy) * d.ffa_rush_yds.fillna(0), d.ffa_rush_yds.fillna(0))
    rc = np.where(d.l_rec.notna(), wr * d.l_rec + (1 - wr) * d.ffa_rec.fillna(0), d.ffa_rec.fillna(0))
    return base.loc[d.index] + ry * .1 + ru * .1 + rc
GRID = [(wy, wr) for wy in (0.5, 0.7, 0.9, 1.0) for wr in (0.0, 0.3, 0.6)]
def best_w(d): return min(GRID, key=lambda w: (d.actual_ppr - ppr_w(d, *w)).abs().mean())
L += ["| weight scheme | 2024 MAE | 2025 MAE | pooled | fitted weights (train 2023-24, yards/rec) |", "|---|---|---|---|---|"]
for name, keys in (("global", []), ("by position", ["position"]), ("by star", ["star"]), ("by position x star", ["position", "star"]), ("by tier", ["tier_b"]), ("by early", ["early"]), ("by position x early", ["position", "early"])):
    out = []; fitted = {}
    for s in (2024, 2025):
        tr, te = g[g.season < s], g[g.season == s].copy(); te["pred"] = np.nan
        if not keys:
            w = best_w(tr); te["pred"] = ppr_w(te, *w); fitted["all"] = w
        else:
            for key, sub in te.groupby(keys):
                trs = tr[(tr[keys] == pd.Series(key if isinstance(key, tuple) else (key,), index=keys)).all(axis=1)]
                w = best_w(trs) if len(trs) >= 150 else best_w(tr); te.loc[sub.index, "pred"] = ppr_w(sub, *w); fitted[key] = w
        out.append(te)
    P = pd.concat(out); m = {s: (x.actual_ppr - x.pred).abs().mean() for s, x in P.groupby("season")}
    L.append(f"| {name} | {m[2024]:.3f} | {m[2025]:.3f} | {(P.actual_ppr - P.pred).abs().mean():.3f} | {'; '.join(f'{k}:{v[0]}/{v[1]}' for k, v in list(fitted.items())[:8])} |")
L.append(f"\nFFA-only MAE on the same rows: 2024 {(g[g.season==2024].actual_ppr - ppr_w(g[g.season==2024], 0, 0)).abs().mean():.3f}, 2025 {(g[g.season==2025].actual_ppr - ppr_w(g[g.season==2025], 0, 0)).abs().mean():.3f}. Rows: {len(g):,}.\n")

# ---------- summary ----------
best = sorted(res_rows, key=lambda x: x[1])[:5]
L.insert(3, "**Best cross-season schemes by out-of-sample dMAE (negative = better than the incumbent):** " + "; ".join(f"{n} {d:+.4f} (perm p {p:.2f})" for n, d, p in best) + "\n")
open(A.out, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
