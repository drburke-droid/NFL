"""Does true skill add value to WEEKLY player props beyond the closing market?

Base: outputs/prop_ev_backtest.pkl (17K matched closing props 2024-25, de-vigged
consensus novig, actual outcome won_over, per-market line mline, model mu/sig).
Skill joined causally:
  - skill_true_causal, alpha_skill through season S-1 (player_skill_true/alpha)
  - within-season EPA/play to date (nflv_pbp_skill_wk, weeks < w, play-weighted),
    z-scored within position
Tests:
  1. Logistic won_over ~ novig + skill, per market group (does skill shift P(over)?)
  2. Partial corr: skill vs standardized (actual - line) residual | novig, line level
  3. Brier ladder train 2024 -> test 2025: M1 shrink vs +skill vs anchored LGBM +/- skill
  4. ROI at best book for the skill-augmented model if anything survives
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")

A = pd.read_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
A = A.dropna(subset=["novig", "won_over"]).copy()
con = sqlite3.connect(DB)
st = pd.read_sql("SELECT player_id, season, skill_true_causal FROM player_skill_true", con)
al = pd.read_sql("SELECT player_id, season, alpha_skill FROM player_skill_alpha", con)
sk = st.merge(al, on=["player_id", "season"], how="outer")
sk["season"] += 1
A = A.merge(sk, on=["player_id", "season"], how="left")

# within-season EPA/play to date (causal: weeks strictly before the prop week)
wk = pd.read_sql("""SELECT gsis_id player_id, season, week, plays, epa_play
                    FROM nflv_pbp_skill_wk WHERE season >= 2023""", con)
con.close()
wk = wk.sort_values(["player_id", "season", "week"])
g = wk.groupby(["player_id", "season"])
wk["cum_wepa"] = (g["epa_play"].apply(lambda s: (s * wk.loc[s.index, "plays"]).cumsum().shift(1))
                  .reset_index(level=[0, 1], drop=True))
wk["cum_plays"] = g["plays"].cumsum().shift(1).where(g.cumcount() > 0)
wk["epa_td"] = wk["cum_wepa"] / wk["cum_plays"]
A = A.merge(wk[["player_id", "season", "week", "epa_td", "cum_plays"]],
            on=["player_id", "season", "week"], how="left")
A.loc[A["cum_plays"] < 50, "epa_td"] = np.nan
A["epa_td_z"] = A.groupby("position")["epa_td"].transform(lambda s: (s - s.mean()) / (s.std() or 1))

A["mkt_id"] = A.market.astype("category").cat.codes
A["line_lvl"] = A.groupby("market").mline.transform(lambda x: (x - x.mean()) / (x.std() or 1))
A["resid_z"] = (A["y"] - A["mline"]) / A["sig"].clip(lower=1e-6)
A["resid_z"] = A["resid_z"].clip(-4, 4)
SKC = ["skill_true_causal", "alpha_skill", "epa_td_z"]
print(f"props: {len(A):,}  with skill S-1: {A.skill_true_causal.notna().sum():,}  "
      f"alpha: {A.alpha_skill.notna().sum():,}  epa-to-date: {A.epa_td_z.notna().sum():,}")
print("markets:", A.market.value_counts().to_dict())

print("\n=== 1. Logistic won_over ~ novig + skill (per market) ===")
print(f"{'market':26s} {'n':>5s} | coef(skill) p | coef(alpha) p | coef(epa_td) p")
def lcoef(d, col):
    d = d.dropna(subset=[col, "novig", "won_over"])
    if len(d) < 300: return None
    X = np.column_stack([d["novig"], d["line_lvl"], (d[col] - d[col].mean()) / (d[col].std() or 1)])
    m = LogisticRegression(C=10.0, max_iter=1000).fit(X, d["won_over"])
    b = m.coef_[0][2]
    # crude z via bootstrap-free Wald approx: refit on halves for stability check instead
    p = np.nan
    try:
        import statsmodels.api as sm
        r = sm.Logit(d["won_over"], sm.add_constant(pd.DataFrame(X, index=d.index,
              columns=["novig", "lvl", "sk"]))).fit(disp=0)
        b, p = r.params["sk"], r.pvalues["sk"]
    except Exception: pass
    return b, p, len(d)
for mkt, d in A.groupby("market"):
    row = f"{mkt:26s} {len(d):5d} |"
    for col in SKC:
        r = lcoef(d, col)
        row += f" {r[0]:+.3f} {r[1]:.2f} |" if r else "   n/a      |"
    print(row)
row = f"{'ALL (pooled+mkt_id)':26s} {len(A):5d} |"
for col in SKC:
    r = lcoef(A, col)
    row += f" {r[0]:+.3f} {r[1]:.2f} |" if r else "   n/a      |"
print(row)

print("\n=== 2. Partial corr: skill vs (actual-line)/sig | novig, line_lvl ===")
def partial(d, xcol):
    d = d.dropna(subset=[xcol, "resid_z", "novig", "line_lvl"])
    if len(d) < 300: return None
    M = np.column_stack([np.ones(len(d)), d["novig"], d["line_lvl"]])
    rx = d[xcol] - M @ np.linalg.lstsq(M, d[xcol], rcond=None)[0]
    ry = d["resid_z"] - M @ np.linalg.lstsq(M, d["resid_z"], rcond=None)[0]
    r, p = pearsonr(rx, ry)
    return r, p, len(d)
print(f"{'market':26s} | skill_causal | alpha | epa_td")
for mkt, d in A.groupby("market"):
    row = f"{mkt:26s} |"
    for col in SKC:
        r = partial(d, col)
        row += f" r={r[0]:+.3f} p={r[1]:.2f} |" if r else " n/a |"
    print(row)

print("\n=== 3. Brier ladder: train 2024 -> test 2025 ===")
tr, te = A[A.season == 2024], A[A.season == 2025]
res = {}
def ev(name, p):
    p = np.clip(p, 0.01, 0.99)
    res[name] = (brier_score_loss(te.won_over, p), log_loss(te.won_over, p))
    return p
p0 = ev("M0 raw novig", te.novig.values)
shift = tr.won_over.mean() - tr.novig.mean()
p1 = ev("M1 global shrink", te.novig.values + shift)
BASE_L = ["novig", "mkt_id", "line_lvl"]
def logi(feats, name):
    trd = tr.dropna(subset=feats); X = trd[feats].values
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, trd["won_over"])
    Xt = te[feats].fillna(te[feats].median()).values
    return ev(name, m.predict_proba(Xt)[:, 1])
p2 = logi(BASE_L, "M2 market logistic")
p2s = logi(BASE_L + SKC, "M2 + skill features")
A["z_gap"] = (A.mu - A.mline) / A.sig
FE = [c for c in A.columns if c.startswith(("l3_", "l6_", "d_"))] + \
     ["gp", "itt", "spread", "gt", "arch_id", "dstyle_id", "pos_id", "p_vs_style"]
F3 = ["novig", "mkt_id", "line_lvl", "z_gap", "mu", "sig"] + FE
GB = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15,
          min_child_samples=50, subsample=0.8, colsample_bytree=0.7, random_state=0, verbosity=-1)
tr2, te2 = A[A.season == 2024], A[A.season == 2025]
m3 = lgb.LGBMClassifier(**GB).fit(tr2[F3].astype(float).fillna(-1), tr2.won_over)
p3 = ev("M3 anchored LGBM", m3.predict_proba(te2[F3].astype(float).fillna(-1))[:, 1])
m3s = lgb.LGBMClassifier(**GB).fit(tr2[F3 + SKC].astype(float).fillna(-1), tr2.won_over)
p3s = ev("M3 + skill features", m3s.predict_proba(te2[F3 + SKC].astype(float).fillna(-1))[:, 1])
print(f"{'model':28s} {'Brier':>8s} {'logloss':>9s}")
for k, (b, l) in res.items():
    print(f"  {k:26s} {b:.5f}  {l:.5f}")
imp = pd.DataFrame({"f": F3 + SKC, "i": m3s.feature_importances_}).sort_values("i", ascending=False)
print("  M3+skill: skill feature ranks:",
      {c: int((imp.f == c).idxmax() and imp.reset_index(drop=True).index[imp.reset_index(drop=True).f == c][0] + 1) for c in SKC})

print("\n=== 4. ROI 2025, best book, skill-augmented model ===")
for name, p in [("M1 shrink", p1), ("M2+skill", p2s), ("M3+skill", p3s)]:
    for th in (0.02, 0.05):
        evo = p * te.best_over.values - 1
        evu = (1 - p) * te.best_under.values - 1
        side = np.where(evo > evu, 1, 0); best_ev = np.maximum(evo, evu); m = best_ev > th
        if m.sum() < 50: print(f"  {name} EV>{th:.0%}: n={m.sum()} too few"); continue
        win = (side[m] == te.won_over.values[m]).astype(int)
        dp = np.where(side[m] == 1, te.best_over.values[m], te.best_under.values[m])
        pnl = np.where(win == 1, dp - 1, -1.0)
        print(f"  {name:10s} EV>{th:.0%}: n={m.sum():5d}  hit {win.mean():.1%}  ROI {pnl.mean():+.1%}  (overs {side[m].mean():.0%})")
