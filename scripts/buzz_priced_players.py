"""
Does August Wikipedia buzz predict performance for PRICED players — not just $1 darts?

The late-breakout study (outputs/models/LATE_BREAKOUTS.md §7) validated buzz only inside
the cheap pool (ADP>120 / AAV<=$2). Here we test the DRAFTED pool (FFA ADP<=120): does
August attention — level or spike vs the player's own Jan-May baseline — carry signal for
(a) beating the price-implied VORP expectation, and (b) beating the FFA projection?

Method: within-season-and-ADP-bucket comparisons (attention level correlates with stardom,
so raw views must be normalized to price peers), outcome = VORP residual vs the bucket
median and act_pts − ffa_points. Buckets: ADP 1-36 / 37-72 / 73-120.
"""
import os, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
wb = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
wb["y"] = wb.ym.str[:4].astype(int); wb["m"] = wb.ym.str[4:].astype(int)

# per player-season: August views + Jan-May baseline of the SAME calendar year
aug = wb[wb.m == 8].rename(columns={"views": "aug_views", "y": "season"})[["player_id", "season", "aug_views"]]
base = (wb[wb.m.between(1, 5)].groupby(["player_id", "y"]).views.median()
        .rename("base_views").reset_index().rename(columns={"y": "season"}))
df = df.merge(aug, on=["player_id", "season"], how="left").merge(base, on=["player_id", "season"], how="left")
df["buzz_spike"] = np.log((df.aug_views + 10) / (df.base_views + 10))

# PRICED pool: in FFA with ADP <= 120
P = df[(df.in_ffa == 1) & (df.adp_f <= 120) & df.aug_views.notna()].copy()
P["bucket"] = pd.cut(P.adp_f, [0, 36, 72, 120], labels=["ADP 1-36", "ADP 37-72", "ADP 73-120"])
print(f"priced pool with buzz coverage: {len(P)} player-seasons "
      f"({P.season.min()}-{P.season.max()}); coverage {len(P)/((df.in_ffa==1)&(df.adp_f<=120)).sum():.0%}")

# outcomes: VORP residual vs same-season-bucket-position median; projection residual
P["vorp_resid"] = P.vorp - P.groupby(["season", "bucket", "position"], observed=True).vorp.transform("median")
P["proj_resid"] = P.act_pts - P.ffa_points
# attention normalized WITHIN season x bucket (stars always have more raw views)
P["lvl_pct"] = P.groupby(["season", "bucket"], observed=True).aug_views.rank(pct=True)
P["spk_pct"] = P.groupby(["season", "bucket"], observed=True).buzz_spike.rank(pct=True)

def table(col, label):
    print(f"\n=== {label} (quintiles within season x ADP bucket) ===")
    P["q"] = pd.cut(P[col], [0, .2, .4, .6, .8, 1.0], labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    for b in P.bucket.cat.categories:
        sub = P[P.bucket == b]
        rows = []
        for q in sub.q.cat.categories:
            s = sub[sub.q == q]
            rows.append(f"{q}: vorpΔ {s.vorp_resid.mean():+6.1f} | projΔ {s.proj_resid.mean():+6.1f} | hit {s.hit.mean():.0%} (n={len(s)})")
        print(f"  {b}:  " + "   ".join(rows))
    # rank correlations, pooled
    from scipy import stats
    for out in ("vorp_resid", "proj_resid"):
        ok = P[[col, out]].dropna()
        r, p = stats.spearmanr(ok[col], ok[out])
        print(f"  pooled Spearman {col} vs {out}: r={r:+.3f} (p={p:.3f}, n={len(ok)})")

table("lvl_pct", "August attention LEVEL")
table("spk_pct", "August attention SPIKE vs own Jan-May baseline")

# does a big spike among ALREADY-PRICED players flag injury/news noise instead of upside?
print("\n=== top-decile spikes among priced players: who are they? (sample, 2022-25) ===")
top = P[(P.spk_pct >= .9) & (P.season >= 2022)].nlargest(14, "buzz_spike")
for _, r in top.iterrows():
    print(f"  {int(r.season)} {r['name']:24s} {r.position} ADP {r.adp_f:5.0f} spike {r.buzz_spike:+.2f} "
          f"vorpΔ {r.vorp_resid:+6.1f} projΔ {r.proj_resid:+6.1f}")
