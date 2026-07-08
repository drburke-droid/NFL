"""
Weekly-grain sequel to sentiment_props_study.py: does the fade-bad-news-unders edge
SHARPEN when sentiment is measured over the 1-2 GDELT weeks immediately before the game
(instead of the prior calendar month)?

Alignment: props -> games.commence_time via event_id; sentiment = nflv_gdelt_wk rows whose
week-start falls 3-16 days before kickoff (the last two COMPLETED Tue-anchored weeks).
Volume floors scaled from monthly (100/mo ~ 25/wk). Report: outputs/reports/sentiment_props.md (appended)
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
d = d.dropna(subset=["won_over", "novig", "mline"]).copy()
d["nm"] = d.player_name.map(norm)
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
g = pd.read_sql("SELECT event_id, commence_time FROM games", con)
d = d.merge(g, on="event_id", how="left")
d["gdate"] = pd.to_datetime(d.commence_time).dt.tz_localize(None).dt.normalize()

wk = pd.read_sql("SELECT name nm, wk, articles, avg_tone FROM nflv_gdelt_wk", con)
wk["wkd"] = pd.to_datetime(wk.wk)

# for each prop: volume-weighted tone over GDELT weeks starting 3-16 days before kickoff
d = d.reset_index(drop=True); d["pi"] = d.index
m = d[["pi", "nm", "gdate"]].merge(wk, on="nm", how="left")
m["lag"] = (m.gdate - m.wkd).dt.days
m = m[m.lag.between(3, 16)]
agg = m.groupby("pi").apply(lambda x: pd.Series({
    "wk_n": x.articles.sum(),
    "wk_tone": np.average(x.avg_tone, weights=x.articles) if x.articles.sum() > 0 else np.nan}),
    include_groups=False).reset_index()
d = d.merge(agg, on="pi", how="left")
d["resid"] = d.won_over - d.novig
ok = d.dropna(subset=["wk_tone"]).copy()
print(f"props with prior-2-week sentiment: {len(ok)} of {len(d)}")
print(f"baseline resid (over result - market prob): {ok.resid.mean():+.3f}\n")

print("L2 weekly — resid by prior-2wk tone bucket:")
for lab, sub in (("tone < -3", ok[ok.wk_tone < -3]), ("-3..-2", ok[ok.wk_tone.between(-3, -2)]),
                 ("-2..0", ok[ok.wk_tone.between(-2, 0)]), ("tone > 0", ok[ok.wk_tone > 0])):
    if len(sub) > 30:
        se = sub.resid.std() / np.sqrt(len(sub))
        print(f"   {lab:<10} n={len(sub):>5}  resid {sub.resid.mean():+.3f} (±{2*se:.3f})")
r, p = stats.spearmanr(ok.wk_tone, ok.resid)
print(f"   Spearman wk_tone vs resid: r={r:+.3f} (p={p:.4f})")

def roi(sub, side):
    if not len(sub): return 0, 0
    ret = np.where(sub.won_over == (0 if side == "under" else 1),
                   (sub.best_under if side == "under" else sub.best_over) - 1, -1.0)
    return ret.mean(), len(sub)

print("\nL3 weekly — ROI (flat 1u, best price):")
cells = [
    ("ALL unders baseline", ok, "under"),
    ("UNDER wk_tone < -2", ok[ok.wk_tone < -2], "under"),
    ("UNDER wk_tone < -2 & wk_n>=25", ok[(ok.wk_tone < -2) & (ok.wk_n >= 25)], "under"),
    ("UNDER wk_tone < -3 & wk_n>=25", ok[(ok.wk_tone < -3) & (ok.wk_n >= 25)], "under"),
    ("UNDER wk_tone < -2 & wk_n>=100", ok[(ok.wk_tone < -2) & (ok.wk_n >= 100)], "under"),
    ("OVER  wk_tone < -2 & wk_n>=25 (sanity)", ok[(ok.wk_tone < -2) & (ok.wk_n >= 25)], "over"),
]
for lab, sub, side in cells:
    mn, n = roi(sub, side)
    print(f"   {lab:<40} n={n:>5}  ROI {mn:+.1%}")

print("\nseason split for UNDER wk_tone<-2 & wk_n>=25:")
cell = ok[(ok.wk_tone < -2) & (ok.wk_n >= 25)]
for ssn in sorted(ok.season.unique()):
    mn, n = roi(cell[cell.season == ssn], "under"); mb, nb = roi(ok[ok.season == ssn], "under")
    print(f"   {ssn}: cell n={n:>4} ROI {mn:+.1%}   | all-unders n={nb} ROI {mb:+.1%}")
print("\nby market:")
for mk, sub in cell.groupby("market"):
    mn, n = roi(sub, "under")
    print(f"   {mk:<28} n={n:>4}  ROI {mn:+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as f:
    f.write("\n\n## Weekly-grain sequel (`sentiment_props_weekly.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
