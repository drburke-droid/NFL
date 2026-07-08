"""
Recalibrate the prop model's p_over — it under-predicts overs by ~5 probability points
(sentiment_prop_model_boost.py finding), which poisons side-selection before any signal
can help. Season-out protocol: fit ONLY on the train season, judge on the other.

Candidates (nested, all fit by MLE/grid on train only):
  SHIFT  logit(p') = logit(p) + b                (fix the level)
  PLATT  logit(p') = a*logit(p) + b              (fix level + overconfidence)
  BLEND  p'' = w*platt(p) + (1-w)*novig          (shrink toward the market)
Metrics: Brier / log-loss vs the market's novig; reliability deciles; then the betting
sims (bet side when |p - novig| >= t at best price) with and without the tone veto.
Appends to outputs/reports/sentiment_props.md.
"""
import os, re, sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

EPS = 1e-6
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
inv = lambda z: 1 / (1 + np.exp(-z))
ll = lambda y, p: -np.mean(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
brier = lambda y, p: np.mean((y - p) ** 2)

d = pd.read_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
d = d.dropna(subset=["won_over", "novig", "p_over", "best_over", "best_under"]).copy()
d["nm"] = d.player_name.map(norm)
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
gd = pd.read_sql("SELECT name nm, ym, articles, avg_tone FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)
d["month"] = np.select([d.week <= 4, d.week <= 8, d.week <= 13, d.week <= 17], [9, 10, 11, 12], 1)
d["sy"] = d.season; d.loc[d.month == 1, "sy"] = d.season + 1
pr = gd[["nm", "y", "m", "articles", "avg_tone"]].copy()
pr["month"] = pr.m + 1; pr.loc[pr.m == 12, "month"] = 1
pr["sy"] = pr.y; pr.loc[pr.m == 12, "sy"] += 1
d = d.merge(pr[["nm", "sy", "month", "articles", "avg_tone"]]
            .rename(columns={"articles": "prev_n", "avg_tone": "prev_tone"}), on=["nm", "sy", "month"], how="left")
d["bad"] = ((d.prev_tone < -2) & (d.prev_n >= 100)).fillna(False)

def fit(train):
    y, z = train.won_over.values, logit(train.p_over.values)
    b = minimize_scalar(lambda b: ll(y, inv(z + b)), bounds=(-2, 2), method="bounded").x
    ab = minimize(lambda ab: ll(y, inv(ab[0] * z + ab[1])), x0=[1.0, b], method="Nelder-Mead").x
    pz = inv(ab[0] * z + ab[1])
    ws = np.linspace(0, 1, 41)
    w = ws[np.argmin([ll(y, w * pz + (1 - w) * train.novig.values) for w in ws])]
    return {"b": b, "a": ab[0], "ab_b": ab[1], "w": w}

def apply(sub, F, kind):
    z = logit(sub.p_over.values)
    if kind == "raw":   return sub.p_over.values
    if kind == "shift": return inv(z + F["b"])
    if kind == "platt": return inv(F["a"] * z + F["ab_b"])
    if kind == "blend": return F["w"] * inv(F["a"] * z + F["ab_b"]) + (1 - F["w"]) * sub.novig.values

def sims(sub, p, veto):
    e = p - sub.novig.values
    out = []
    for t in (0.03, 0.05, 0.08):
        keep_o = (e >= t) & ~(veto & sub.bad.values)
        keep_u = e <= -t
        bo, bu = sub[keep_o], sub[keep_u]
        ret = np.concatenate([np.where(bo.won_over == 1, bo.best_over - 1, -1.0),
                              np.where(bu.won_over == 0, bu.best_under - 1, -1.0)])
        out.append((t, len(ret), ret.mean() if len(ret) else 0.0))
    return out

for tr_y, te_y in ((2024, 2025), (2025, 2024)):
    tr, te = d[d.season == tr_y], d[d.season == te_y]
    F = fit(tr)
    print(f"=== fit {tr_y} -> test {te_y} ===")
    print(f"   params: shift b={F['b']:+.3f} | platt a={F['a']:.3f} b={F['ab_b']:+.3f} | blend w={F['w']:.2f} (model weight)")
    y = te.won_over.values
    print(f"   {'':<8}{'brier':>8}{'logloss':>9}")
    for kind in ("raw", "shift", "platt", "blend"):
        p = apply(te, F, kind)
        print(f"   {kind:<8}{brier(y, p):>8.4f}{ll(y, p):>9.4f}")
    print(f"   {'market':<8}{brier(y, te.novig.values):>8.4f}{ll(y, te.novig.values):>9.4f}")
    print(f"   betting sims on {te_y} (n, ROI at t=3/5/8%):")
    for kind in ("raw", "blend"):
        p = apply(te, F, kind)
        for veto in (False, True):
            lab = f"{kind}{' +veto' if veto else '':<6}"
            cells = sims(te, p, veto)
            print(f"     {lab:<12}" + "".join(f"  n={n:>5} {r:+.1%}" for t, n, r in cells))
    print()

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Prop-model calibration fix (`prop_model_calibration.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
