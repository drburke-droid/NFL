"""Does a first-week-of-September wiki spike carry signal for dart hits,
beyond the validated August spike? Cheap pool 2016-2025.
Clean window: Sep 1 .. min(Sep 7, opener-1). Aug 25-31 dailies kept separately."""
import sqlite3, os
import numpy as np, pandas as pd

root = r"C:\Users\drbur\Documents\GitHub\NFL"
con = sqlite3.connect(os.path.join(root, "db", "nfl_odds.db"))

OPENER = {2016: "0908", 2017: "0907", 2018: "0906", 2019: "0905", 2020: "0910",
          2021: "0909", 2022: "0908", 2023: "0907", 2024: "0905", 2025: "0904"}

frame = pd.read_pickle(os.path.join(root, r"outputs\models\late_breakout_frame.pkl"))
pool = frame[frame.cheap == 1][["player_id", "season", "name", "position", "hit"]].copy()

d = pd.read_sql("SELECT * FROM nflv_wiki_buzz_daily_sep", con)
d["season"] = d.date.str[:4].astype(int)
d["mmdd"] = d.date.str[4:]

rows = []
for (pid, y), g in d.groupby(["player_id", "season"]):
    stop = OPENER.get(y, "0908")
    sep = g[(g.mmdd >= "0901") & (g.mmdd < stop) & (g.mmdd <= "0907")]
    cut = g[(g.mmdd >= "0825") & (g.mmdd <= "0831")]
    rows.append({"player_id": pid, "season": y,
                 "sep_rate": sep.views.mean() if len(sep) else np.nan,
                 "sep_days": len(sep),
                 "cut_rate": cut.views.mean() if len(cut) else np.nan})
sig = pd.DataFrame(rows)

# monthly base (Jan-May median) + August, same construction as buzz_features
wb = pd.read_sql("SELECT * FROM nflv_wiki_buzz", con)
wb["year"] = wb.ym.str[:4].astype(int); wb["month"] = wb.ym.str[4:6].astype(int)
wb = wb[wb.year.between(2016, 2025)]
aug = wb[wb.month == 8].rename(columns={"year": "season", "views": "aug_views"})[
    ["player_id", "season", "aug_views"]]
base = (wb[wb.month.between(1, 5)].groupby(["player_id", "year"]).views.median()
        .rename("base_views").reset_index().rename(columns={"year": "season"}))
sig = sig.merge(aug, on=["player_id", "season"], how="left").merge(
    base, on=["player_id", "season"], how="left")

sig["sep_spike"] = np.log((sig.sep_rate * 31 + 10) / (sig.base_views + 10))
sig["aug_spike"] = np.log((sig.aug_views + 10) / (sig.base_views + 10))
sig["sep_vs_aug"] = np.log((sig.sep_rate + 1) / (sig.aug_views / 31 + 1))
sig["cut_vs_aug"] = np.log((sig.cut_rate + 1) / (sig.aug_views / 31 + 1))

m = pool.merge(sig, on=["player_id", "season"], how="inner").dropna(subset=["sep_spike"])
for c in ["sep_spike", "aug_spike", "sep_vs_aug", "cut_vs_aug"]:
    m[c + "_p"] = m.groupby("season")[c].rank(pct=True)
print(f"pool with Sep data: {len(m)} player-seasons, base hit {m.hit.mean():.3f}")
print("seasons:", m.groupby("season").size().to_dict())

print("\n[1] hit rate by SEP spike quartile (vs Jan-May base):")
q = pd.cut(m.sep_spike_p, [0, .25, .5, .75, .9, 1.0])
print(m.groupby(q).agg(n=("hit", "size"), hit=("hit", "mean")).round(3).to_string())
print("\n    same for AUG spike (reference):")
qa = pd.cut(m.aug_spike_p, [0, .25, .5, .75, .9, 1.0])
print(m.groupby(qa).agg(n=("hit", "size"), hit=("hit", "mean")).round(3).to_string())

print("\n[4] corr(aug_spike_p, sep_spike_p) =", round(m.aug_spike_p.corr(m.sep_spike_p), 3))

print("\n[2a] logit hit ~ aug_spike_p + sep_vs_aug_p:")
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
ml = m.dropna(subset=["aug_spike_p", "sep_vs_aug_p"]).copy()
print(f"    (logit rows: {len(ml)})")
X = ml[["aug_spike_p", "sep_vs_aug_p"]].values; yv = ml.hit.values
lr = LogisticRegression().fit(X, yv)
print("    coefs:", dict(zip(["aug_spike_p", "sep_vs_aug_p"], lr.coef_[0].round(3))))
# bootstrap the sep coefficient
rng = np.random.default_rng(0)
cs = []
for _ in range(500):
    ix = rng.integers(0, len(ml), len(ml))
    cs.append(LogisticRegression().fit(X[ix], yv[ix]).coef_[0][1])
cs = np.array(cs)
print(f"    sep_vs_aug_p coef bootstrap: mean {cs.mean():.3f}, P(coef<=0) = {(cs<=0).mean():.3f}")

print("\n[2b] the actionable cell: Aug-QUIET (<75th) x Sep-SURGE (>=90th sep_vs_aug):")
m["aug_quiet"] = (m.aug_spike_p < 0.75)
m["sep_surge"] = (m.sep_vs_aug_p >= 0.90)
cells = m.groupby(["aug_quiet", "sep_surge"]).agg(n=("hit", "size"), hit=("hit", "mean")).round(3)
print(cells.to_string())
cell = m[m.aug_quiet & m.sep_surge]
rest = m[m.aug_quiet & ~m.sep_surge]
if len(cell):
    p = st.binomtest(int(cell.hit.sum()), len(cell), rest.hit.mean()).pvalue
    print(f"    Aug-quiet+Sep-surge: {cell.hit.mean():.3f} (n={len(cell)}) vs Aug-quiet rest "
          f"{rest.hit.mean():.3f}; binom p={p:.3f}")
    print("    per-season hits in that cell:")
    print(cell.groupby("season").agg(n=("hit", "size"), hits=("hit", "sum")).to_string())

print("\n[3] per-season top-10 by sep_vs_aug (cheap pool): mean hit")
def top10(col):
    t = m.dropna(subset=[col]).sort_values(col, ascending=False).groupby("season").head(10)
    return t.groupby("season").hit.mean()
t10 = top10("sep_vs_aug")
print(t10.round(2).to_string(), "| pooled:", round(t10.mean(), 3))
print("    top-10 by sep_spike:", round(top10("sep_spike").mean(), 3),
      "| top-10 by aug_spike:", round(top10("aug_spike").mean(), 3))

print("\n[bonus] cutdown week (Aug25-31 rate vs Aug avg), same cell test:")
m["cut_surge"] = (m.cut_vs_aug_p >= 0.90)
cellc = m[m.aug_quiet & m.cut_surge]
print(f"    Aug-quiet+cut-surge hit: {cellc.hit.mean():.3f} (n={len(cellc)})")

print("\nexamples: top 15 Aug-quiet Sep-surgers of all time (did they hit?):")
ex = m[m.aug_quiet].nlargest(15, "sep_vs_aug")[
    ["season", "name", "position", "sep_rate", "aug_views", "sep_vs_aug", "hit"]]
print(ex.round(2).to_string(index=False))
con.close()
