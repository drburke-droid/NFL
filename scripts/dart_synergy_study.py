"""
DART FEATURE SYNERGY — do the cheap-pool signals STACK, or merely add?

Same ex-ante pool construction as bench_darts_backtest.py (RB/WR/TE veterans,
FFA AAV<=8 or unlisted, 2016-2025), signals from T-1 only:
  ALPHA (skill >=80th pctl) . VAC (vacated role) . HEIR (out-ran own lead RB)
  FLASH (best-4-wk >=12) . H2 (2nd-half surge) . BUZZ (Aug wiki views: spike
  vs own Jan-May baseline, top quartile of pool)

For every pair: observed hit rate of A-and-B vs the MULTIPLICATIVE-independence
expectation (base x liftA x liftB). synergy > 1 = the combo beats what the two
signals independently imply; Fisher exact vs pool for significance; logistic
reliable ~ A + B + A:B interaction p as the stricter test.

Output: outputs/reports/dart_synergy.md
"""
import os, sqlite3, itertools
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
S = pd.read_sql("""SELECT player_id, player_display_name name, position, season, recent_team team,
                          games, carries, rushing_yards, rushing_epa, fantasy_points_ppr
                   FROM nflv_season WHERE position IN ('RB','WR','TE')""", con)
W = pd.read_sql("""SELECT player_id, position, season, week, fantasy_points_ppr pts
                   FROM nflv_weekly WHERE season_type='REG' AND position IN ('RB','WR','TE')""", con)
A = pd.read_sql("SELECT player_id, position, season, alpha_skill FROM player_skill_alpha", con)
OPP = pd.read_sql("SELECT player_id, season, vacated_role FROM nflv_opportunity", con)
HT = pd.read_sql("SELECT player_id, season, ht_d_ppg, ht_d_snap FROM nflv_half_trend", con)
FFA = pd.read_sql("""SELECT season, player_id, ffa_aav FROM nflv_ffa_league
                     WHERE player_id IS NOT NULL""", con).drop_duplicates(["season", "player_id"])
WB = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
con.close()

A["apct"] = A.groupby(["position", "season"])["alpha_skill"].rank(pct=True)
START_RANK = {"RB": 24, "WR": 24, "TE": 12}
W["wk_rank"] = W.groupby(["season", "week", "position"])["pts"].rank(ascending=False)
W["starter_wk"] = W.apply(lambda r: r.wk_rank <= START_RANK[r.position], axis=1)
out = W.groupby(["player_id", "position", "season"]).agg(starter_wks=("starter_wk", "sum")).reset_index()
ppg = S.assign(ppg=S.fantasy_points_ppr / S.games.clip(lower=1))
ppg["pos_rank"] = ppg[ppg.games >= 10].groupby(["season", "position"])["ppg"].rank(ascending=False)
out = out.merge(ppg[["player_id", "season", "pos_rank"]], on=["player_id", "season"], how="left")
out["reliable"] = out.starter_wks >= 6
out["star"] = out.pos_rank <= 12

# august buzz spike per (player, draft-year T): Aug(T) views vs own Jan-May(T) median
WB["y"] = WB.ym.str[:4].astype(int); WB["m"] = WB.ym.str[-2:].astype(int)
aug = WB[WB.m == 8].rename(columns={"views": "augv"})[["player_id", "y", "augv"]]
bs = WB[WB.m.between(1, 5)].groupby(["player_id", "y"]).views.median().rename("basev").reset_index()
BZ = aug.merge(bs, on=["player_id", "y"], how="left")
BZ["spike"] = np.log((BZ.augv + 100) / (BZ.basev.fillna(0) + 100))


def heirs(t1):
    rb = S[(S.position == "RB") & (S.season == t1) & (S.carries >= 40)].copy()
    rb["ypc"] = rb.rushing_yards / rb.carries; rb["epc"] = rb.rushing_epa / rb.carries
    ids = set()
    for tm, g in rb.groupby("team"):
        if len(g) < 2: continue
        g = g.sort_values("carries", ascending=False); lead = g.iloc[0]
        for _, b in g.iloc[1:].iterrows():
            if b.ypc > lead.ypc and b.epc > lead.epc: ids.add(b.player_id)
    return ids


pools = []
for T in range(2016, 2026):
    prev = S[(S.season == T - 1) & (S.games >= 4)][["player_id", "name", "position"]]
    aav = FFA[FFA.season == T].set_index("player_id")["ffa_aav"]
    pool = prev.copy(); pool["aav"] = pool.player_id.map(aav)
    pool = pool[(pool.aav.fillna(0) <= 8)]
    w1 = W[W.season == T - 1].sort_values("pts", ascending=False).groupby("player_id").head(4) \
          .groupby("player_id")["pts"].mean()
    pool["FLASH"] = (pool.player_id.map(w1).fillna(0) >= 12).astype(int)
    pool["HEIR"] = pool.player_id.isin(heirs(T - 1)).astype(int)
    pool["VAC"] = (pool.player_id.map(OPP[OPP.season == T].set_index("player_id")["vacated_role"]).fillna(0) > 0).astype(int)
    pool["ALPHA"] = (pool.player_id.map(A[A.season == T - 1].set_index("player_id")["apct"]).fillna(0) >= 0.80).astype(int)
    ht = HT[HT.season == T - 1].set_index("player_id")
    pool["H2"] = ((pool.player_id.map(ht["ht_d_ppg"]).fillna(0) > 1.5) &
                  (pool.player_id.map(ht["ht_d_snap"]).fillna(0) > 5)).astype(int)
    bz = BZ[BZ.y == T].set_index("player_id")["spike"]
    pool["_spike"] = pool.player_id.map(bz)
    q = pool._spike.quantile(0.75)
    pool["BUZZ"] = ((pool._spike >= q) & pool._spike.notna()).astype(int)
    oT = out[out.season == T].set_index("player_id")
    pool["reliable"] = pool.player_id.map(oT["reliable"]).fillna(False).astype(bool)
    pool["star"] = pool.player_id.map(oT["star"]).fillna(False).astype(bool)
    pool["season"] = T
    pools.append(pool)
P = pd.concat(pools, ignore_index=True)
FEATS = ["ALPHA", "VAC", "HEIR", "FLASH", "H2", "BUZZ"]
_out = []


def pr(*a):
    s = " ".join(str(x) for x in a); _out.append(s); print(s)


base_r, base_s = P.reliable.mean(), P.star.mean()
pr(f"pool n={len(P)} (2016-25) . base reliable {base_r:.1%} . base star {base_s:.1%}\n")
pr(f"{'signal':22s} {'n':>5s} {'rel%':>6s} {'liftR':>6s} {'star%':>6s} {'liftS':>6s}")
lift = {}
for f in FEATS:
    d = P[P[f] == 1]; lr = d.reliable.mean() / base_r; ls = d.star.mean() / base_s
    lift[f] = (lr, ls)
    pr(f"{f:22s} {len(d):5d} {d.reliable.mean():6.1%} {lr:5.1f}x {d.star.mean():6.1%} {ls:5.1f}x")

pr("\nPAIRS - observed vs multiplicative-independence expectation (reliable):")
pr(f"{'pair':22s} {'n':>5s} {'obs%':>6s} {'exp%':>6s} {'synergy':>8s} {'fisher_p':>9s} {'logit_int_p':>11s}")
def logit_ll(X, y, l2=1e-6):
    """tiny IRLS logistic; returns max log-likelihood (for LR tests)"""
    w = np.zeros(X.shape[1])
    for _ in range(60):
        z = X @ w; p_ = 1 / (1 + np.exp(-z))
        g = X.T @ (y - p_) - l2 * w
        H = (X * (p_ * (1 - p_))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        w += step
        if np.abs(step).max() < 1e-8: break
    z = X @ w; p_ = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
    return float(y @ np.log(p_) + (1 - y) @ np.log(1 - p_))


for a, b in itertools.combinations(FEATS, 2):
    d = P[(P[a] == 1) & (P[b] == 1)]
    nm = a + "+" + b
    if len(d) < 20:
        pr(f"{nm:22s} {len(d):5d}   (too thin)"); continue
    obs = d.reliable.mean()
    exp = min(base_r * lift[a][0] * lift[b][0], 0.95)
    both = int(d.reliable.sum()); rest = P[~((P[a] == 1) & (P[b] == 1))]
    _, fp = stats.fisher_exact([[both, len(d) - both],
                                [int(rest.reliable.sum()), len(rest) - int(rest.reliable.sum())]])
    # LR test: does the interaction term improve the additive logit?
    y = P.reliable.values.astype(float)
    X0 = np.column_stack([np.ones(len(P)), P[a].values, P[b].values]).astype(float)
    X1 = np.column_stack([X0, (P[a] * P[b]).values.astype(float)])
    ip = 1 - stats.chi2.cdf(2 * (logit_ll(X1, y) - logit_ll(X0, y)), df=1)
    syn = obs / exp if exp > 0 else np.nan
    pr(f"{nm:22s} {len(d):5d} {obs:6.1%} {exp:6.1%} {syn:7.2f}x {fp:9.4f} {ip:11.4f}")

pr("\nSIGNAL-COUNT stack (how many of the 6 fire):")
P["nsig"] = P[FEATS].sum(axis=1)
for k in (0, 1, 2, 3):
    d = P[P.nsig == k] if k < 3 else P[P.nsig >= 3]
    lbl = str(k) if k < 3 else "3+"
    pr(f"  {lbl} signals: n={len(d):5d}  reliable {d.reliable.mean():6.1%} ({d.reliable.mean()/base_r:4.1f}x)"
       f"  star {d.star.mean():5.1%} ({d.star.mean()/base_s:4.1f}x)")

open(os.path.join(ROOT, "outputs", "reports", "dart_synergy.md"), "w", encoding="utf-8").write(
    "# Dart feature synergy - do the cheap-pool signals stack?\n\n"
    "Generated by `scripts/dart_synergy_study.py`. Same ex-ante pool as\n"
    "`bench_darts_backtest.py` + an August wiki buzz-spike flag. 'synergy' = observed\n"
    "hit rate of the pair over the multiplicative-independence expectation; the logit\n"
    "interaction p is the stricter test (small cells lack power - read n first).\n\n```\n"
    + "\n".join(_out) + "\n```\n")
print("\nwrote outputs/reports/dart_synergy.md")
