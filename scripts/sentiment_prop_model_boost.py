"""
Can sentiment improve the EXISTING prop model (mu/sig/p_over in prop_ev_backtest.pkl)?

The model already sees L3/L6 trailing stats — if a bad-news month only matters through
depressed recent production, the model absorbed it and tone adds nothing. Tests:
  A. does prior-month tone predict the MODEL's residual (won_over − p_over)?
  B. betting sims at edge thresholds: baseline (bet the model's side at best price when
     |p_over − novig| ≥ t) vs (i) TONE VETO: skip model-OVER bets on bad-news players,
     (ii) TONE TILT: subtract δ from p_over for bad-news players before picking sides
     (δ = the study's L2 residual, 0.02 — set a priori, not fit).
  C. season split (2024 vs 2025).
Appends to outputs/reports/sentiment_props.md.
"""
import os, re, sqlite3
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

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
d["bad"] = (d.prev_tone < -2) & (d.prev_n >= 100)
d.bad = d.bad.fillna(False)
print(f"props with model + sentiment: {d.prev_tone.notna().sum()} of {len(d)}; "
      f"bad-news cell: {d.bad.sum()}")

# A. does tone predict the MODEL's residual?
ok = d.dropna(subset=["prev_tone"]).copy()
ok["mresid"] = ok.won_over - ok.p_over
r, p = stats.spearmanr(ok.prev_tone, ok.mresid)
print(f"\nA. tone vs MODEL residual (won_over − p_over): r={r:+.3f} (p={p:.4f})")
for lab, sub in (("bad-news cell", ok[ok.bad]), ("rest", ok[~ok.bad])):
    se = sub.mresid.std() / np.sqrt(len(sub))
    print(f"   {lab:<14} n={len(sub):>5}  model resid {sub.mresid.mean():+.3f} (±{2*se:.3f})  "
          f"(negative = overs do worse than the model thinks)")

# B/C. betting sims
def sim(sub, po_col):
    """bet model side at best price when |edge| >= t"""
    rows = []
    for t in (0.03, 0.05, 0.08):
        e = sub[po_col] - sub.novig
        bo = sub[e >= t]; bu = sub[e <= -t]
        ret = np.concatenate([
            np.where(bo.won_over == 1, bo.best_over - 1, -1.0),
            np.where(bu.won_over == 0, bu.best_under - 1, -1.0)])
        rows.append((t, len(bo) + len(bu), ret.mean() if len(ret) else 0.0))
    return rows

ok["p_veto"] = ok.p_over                                     # veto: implemented as a filter below
ok["p_tilt"] = np.where(ok.bad, ok.p_over - 0.02, ok.p_over) # a-priori δ from the study's L2

print("\nB. betting sims (bet model side at best price when |p_over − novig| ≥ t):")
print(f"{'':>22}{'t=3%':>16}{'t=5%':>16}{'t=8%':>16}")
def row(lab, sub, col, veto=False):
    if veto:
        sub = sub[~(sub.bad & (sub[col] - sub.novig > 0))]   # drop model-OVER bets on bad-news players
    cells = sim(sub, col)
    print(f"   {lab:<19}" + "".join(f"  n={n:>5} {r:+.1%}" for t, n, r in cells))
row("baseline model", ok, "p_over")
row("+ tone VETO", ok, "p_over", veto=True)
row("+ tone TILT (−.02)", ok, "p_tilt")
print("\nC. season split:")
for ssn in sorted(ok.season.unique()):
    s = ok[ok.season == ssn]
    print(f"  {ssn}:")
    row("baseline model", s, "p_over")
    row("+ tone VETO", s, "p_over", veto=True)
    row("+ tone TILT (−.02)", s, "p_tilt")

# what does the veto actually remove?
vetoed = ok[ok.bad & (ok.p_over - ok.novig >= 0.03)]
ret = np.where(vetoed.won_over == 1, vetoed.best_over - 1, -1.0)
print(f"\nvetoed bets (model-OVER on bad-news players, t=3%): n={len(vetoed)}, "
      f"their ROI would have been {ret.mean():+.1%}" if len(vetoed) else "\nno vetoed bets at t=3%")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Sentiment × the existing prop model (`sentiment_prop_model_boost.py`)\n\n```\n"
             + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
