"""
Does pre-draft news/attention buzz predict late breakouts?

Buzz features per player-season Y (from nflv_wiki_buzz monthly pageviews):
  aug_views    raw August views (camp + preseason, pre-fantasy-draft)
  buzz_spike   log((aug+10)/(base+10)), base = median Jan-May views
  aug_pctl     percentile of aug_views within season (level of attention)
  spike_pctl   percentile of buzz_spike within season

Tests on the cheap pool (adp>120, aav<=$2, skill positions):
  1. hit rate by spike/level decile (base rate 2.3%)
  2. walk-forward model with vs without buzz features -> delta AUC / p@10
  3. sanity: Puka '23, Kyren '23, ARSB '21
Leakage note: August is measured as the full month; drafts in mid-August may
precede a sliver of it. Wiki coverage floor 2015-07 -> seasons 2016+.
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "models")
DB = os.path.join(ROOT, "db", "nfl_odds.db")

con = sqlite3.connect(DB)
wb = pd.read_sql("SELECT * FROM nflv_wiki_buzz", con)
wb["year"] = wb.ym.str[:4].astype(int)
wb["month"] = wb.ym.str[4:6].astype(int)

rows = []
for (pid, y), g in wb[wb.year >= 2016].groupby(["player_id", "year"]):
    m = dict(zip(g.month, g.views))
    aug = m.get(8, np.nan)
    base = np.nanmedian([m.get(k, np.nan) for k in range(1, 6)])
    rows.append({"player_id": pid, "season": y, "aug_views": aug,
                 "base_views": base,
                 "buzz_spike": np.log((aug + 10) / (base + 10)) if aug == aug and base == base else np.nan})
buzz = pd.DataFrame(rows)
buzz["aug_pctl"] = buzz.groupby("season").aug_views.rank(pct=True)
buzz["spike_pctl"] = buzz.groupby("season").buzz_spike.rank(pct=True)

df = pd.read_pickle(os.path.join(OUT, "late_breakout_frame.pkl"))
df = pd.concat([df, pd.get_dummies(df.position, prefix="pos")], axis=1)
df = df.merge(buzz, on=["player_id", "season"], how="left")
print(f"buzz coverage in frame: {df[df.season.between(2016,2025)].aug_views.notna().mean():.1%}")

for nm, yr in [("Puka Nacua", 2023), ("Kyren Williams", 2023), ("Amon-Ra St. Brown", 2021)]:
    r = df[(df.name == nm) & (df.season == yr)]
    if len(r):
        r = r.iloc[0]
        print(f"  {nm} {yr}: aug={r.aug_views}, base={r.base_views}, spike={r.buzz_spike:.2f}, "
              f"spike_pctl={r.spike_pctl:.2f}, aug_pctl={r.aug_pctl:.2f}, hit={r.hit}")

ch = df[(df.cheap == 1) & (df.season.between(2016, 2025)) & (df.position != "QB")].copy()
print(f"\ncheap pool with buzz: {ch.buzz_spike.notna().sum()}/{len(ch)}; base hit {ch.hit.mean():.3f}")

for col in ["spike_pctl", "aug_pctl"]:
    ch["dec"] = pd.qcut(ch[col], 10, labels=False, duplicates="drop")
    tab = ch.groupby("dec").agg(n=("hit", "size"), hit=("hit", "mean"))
    print(f"\nhit rate by {col} decile:\n", (tab.assign(lift=tab.hit / ch.hit.mean())).round(3).to_string())

# --- walk-forward add-on test ------------------------------------------------
CORE = ["age", "years_exp", "is_rookie", "draft_round", "draft_pick",
        "prior_games", "prior_ppg", "prior2_ppg", "prior_cv", "prior_snap_pct",
        "prior_target_share", "prior_wopr", "prior_receiving_epa", "prior_rushing_epa",
        "prior_carries_pg", "prior_targets_pg", "prior_tds", "team_change",
        "ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_d_tch", "ht_slope",
        "vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets",
        "pos_RB", "pos_TE", "pos_WR"]
MKT = ["adp_f", "aav_f", "in_ffa", "ffa_points", "ffa_upside", "ffa_uncertainty"]
BUZZ = ["aug_views", "buzz_spike", "aug_pctl", "spike_pctl"]
for c in CORE + MKT + BUZZ:
    if c not in df.columns: df[c] = np.nan

allp = df[(df.season <= 2025) & (df.position != "QB")]
pool = allp[allp.cheap == 1]
P = dict(n_estimators=350, learning_rate=0.03, num_leaves=31, min_child_samples=40,
         subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def wf(feats):
    preds = []
    for Y in range(2019, 2026):
        tr = allp[allp.season < Y]; te = pool[pool.season == Y]
        m = lgb.LGBMClassifier(**P).fit(tr[feats], tr.hit)
        t = te.copy(); t["prob"] = m.predict_proba(te[feats])[:, 1]; preds.append(t)
    bt = pd.concat(preds)
    p10 = np.mean([g.sort_values("prob", ascending=False).head(10).hit.mean()
                   for _, g in bt.groupby("season")])
    p20 = np.mean([g.sort_values("prob", ascending=False).head(20).hit.mean()
                   for _, g in bt.groupby("season")])
    return roc_auc_score(bt.hit, bt.prob), p10, p20, bt


for label, feats in [("core+mkt        ", MKT + CORE),
                     ("core+mkt+buzz   ", MKT + CORE + BUZZ),
                     ("buzz only       ", BUZZ)]:
    auc, p10, p20, bt = wf(feats)
    print(f"{label} AUC {auc:.3f}  p@10 {p10:.3f}  p@20 {p20:.3f}")

# feature importance with buzz included
m = lgb.LGBMClassifier(**P).fit(allp[MKT + CORE + BUZZ], allp.hit)
imp = pd.Series(m.feature_importances_, index=MKT + CORE + BUZZ).sort_values(ascending=False)
print("\ntop features (with buzz):\n", imp.head(12).to_string())
