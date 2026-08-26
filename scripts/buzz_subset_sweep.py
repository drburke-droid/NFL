"""
Buzz/GDELT subset sweep — where might signal hide under the pooled noise?

Pooled results so far: Aug wiki spike works ONLY in the cheap pool; GDELT tone alone
dead; buzz on priced players dead (r=-0.01); RB handcuff-storm interaction live;
QB->WR interaction weak-positive. This sweep PRE-REGISTERS 12 subset tests with a
mechanism prior each, instead of dredging every slice. With ~12 tests at alpha .05,
expect ~0.6 false positives — anything without BOTH a decent p and per-season sign
consistency is reported as noise.

Subset proxies for archetypes (historical size/speed data doesn't exist -> use
draft capital x experience x usage):
  post-hype   = draft round <=2, years_exp 2-4, now cheap
  satellite   = RB with prior target share >= 8%
  fame        = high GDELT article volume (recognisable name)

Tests (signal x subset -> outcome; cheap = dart pool, priced = FFA aav >= $10):
   1 buzz x post-hype cheap            -> hit      (crowd re-finding a talent)
   2 buzz x vacated targets, cheap WR  -> hit      (buzz + real opportunity vacuum)
   3 buzz x vacated carries, cheap RB  -> hit      (roster-vacancy analog of the storm)
   4 buzz x satellite cheap RB         -> hit      (PPR role the market underprices)
   5 buzz x young priced (age<=24)     -> beat     (hype on ascenders may be real)
   6 buzz x team-change priced         -> beat     (new-team buzz = role news)
   7 buzz x mid-tier $3-8              -> hit      (the fill bin between pools)
   8 neg-news x priced age>=28         -> overperf (does the flag bite harder old?)
   9 neg-news x priced age<=25         -> overperf (or younger?)
  10 buzz x cheap TE                   -> hit      (TE-specific dart lens)
  11 fame (articles>=P75) x cheap      -> hit      (name-brand cheap vets)
  12 buzz x prior H2 riser, cheap      -> hit      (buzz confirming a real late surge)
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "reports", "buzz_subset_sweep.md")
rng = np.random.default_rng(23)
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))

wb = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
wb["year"] = wb.ym.str[:4].astype(int); wb["month"] = wb.ym.str[4:6].astype(int)
rows = []
for (pid, y), g in wb[wb.year.between(2016, 2025)].groupby(["player_id", "year"]):
    m = dict(zip(g.month, g.views))
    aug = m.get(8, np.nan); base = np.nanmedian([m.get(k, np.nan) for k in range(1, 6)])
    rows.append({"player_id": pid, "season": y,
                 "spike": np.log((aug + 10) / (base + 10)) if aug == aug and base == base else np.nan})
buzz = pd.DataFrame(rows); buzz["spike_pctl"] = buzz.groupby("season").spike.rank(pct=True)

gd = pd.read_sql("SELECT player_id, ym, articles, avg_tone, avg_neg FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["year"] = gd.ym.str[:4].astype(int); gd["month"] = gd.ym.str[4:6].astype(int)
aug_news = (gd[(gd.month == 8) & gd.year.between(2016, 2025)]
            .rename(columns={"year": "season"})[["player_id", "season", "articles", "avg_tone", "avg_neg"]])

df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[df.season.between(2016, 2025)].copy()
df["overperf"] = df.act_pts - df.ffa_points
df["beat"] = (df.act_pts > df.ffa_points).astype(float)
df.loc[df.ffa_points.isna() | df.act_pts.isna(), ["overperf", "beat"]] = np.nan
df = df.merge(buzz, on=["player_id", "season"], how="left").merge(aug_news, on=["player_id", "season"], how="left")
df["articles"] = df.articles.fillna(0)
df["buzz_hi"] = (df.spike_pctl >= 0.75).astype(float).where(df.spike_pctl.notna())
df["neg_hi"] = ((df.articles >= 3) & (df.avg_neg >= df[df.articles >= 3].avg_neg.quantile(.75))).astype(int)
df["fame_hi"] = (df.articles >= df.groupby("season").articles.transform(lambda s: s.quantile(.75))).astype(int)
df["priced"] = (df.ffa_aav.fillna(0) >= 10).astype(int)
df["mid"] = df.ffa_aav.fillna(0).between(3, 8).astype(int)
df["h2riser"] = ((df.ht_d_ppg.fillna(0) > 1.5)).astype(int)

def cell_test(name, sub, flag, outcome, n_boot=4000):
    s = sub.dropna(subset=[flag, outcome])
    a, b = s[s[flag] == 1][outcome].values, s[s[flag] == 0][outcome].values
    if len(a) < 15 or len(b) < 15:
        return {"test": name, "n1": len(a), "n0": len(b), "verdict": "TOO THIN"}
    obs = np.nanmean(a) - np.nanmean(b)
    pool = np.concatenate([a, b]); la = len(a)
    null = []
    for _ in range(n_boot):
        pm = rng.permutation(pool); null.append(np.nanmean(pm[:la]) - np.nanmean(pm[la:]))
    p = float((np.abs(null) >= abs(obs)).mean())
    per = s.groupby("season").apply(lambda g: np.nanmean(g[g[flag] == 1][outcome]) - np.nanmean(g[g[flag] == 0][outcome]))
    pos = int((per > 0).sum()); tot = int(per.notna().sum())
    return {"test": name, "n1": len(a), "n0": len(b),
            "rate1": round(float(np.nanmean(a)), 3), "rate0": round(float(np.nanmean(b)), 3),
            "delta": round(float(obs), 3), "p": round(p, 3), "seasons+": f"{pos}/{tot}",
            "verdict": "CANDIDATE" if (p < 0.05 and pos >= 0.65 * tot) else ("weak" if p < 0.15 else "noise")}

cheap = df[df.cheap == 1]; priced = df[df.priced == 1]
TESTS = [
    ("1 buzz x post-hype cheap", cheap[(cheap.draft_round <= 2) & (cheap.years_exp.between(2, 4))], "buzz_hi", "hit"),
    ("2 buzz x vacated-targets cheap WR", cheap[(cheap.position == "WR") & (cheap.vac_pc_targets.fillna(0) >= 80)], "buzz_hi", "hit"),
    ("3 buzz x vacated-carries cheap RB", cheap[(cheap.position == "RB") & (cheap.vac_rb_carries.fillna(0) >= 80)], "buzz_hi", "hit"),
    ("4 buzz x satellite cheap RB", cheap[(cheap.position == "RB") & (cheap.prior_target_share.fillna(0) >= 0.08)], "buzz_hi", "hit"),
    ("5 buzz x young priced (age<=24)", priced[priced.age <= 24], "buzz_hi", "beat"),
    ("6 buzz x team-change priced", priced[priced.team_change == 1], "buzz_hi", "beat"),
    ("7 buzz x mid-tier $3-8", df[df.mid == 1], "buzz_hi", "hit"),
    ("8 neg-news x priced age>=28", priced[priced.age >= 28], "neg_hi", "overperf"),
    ("9 neg-news x priced age<=25", priced[priced.age <= 25], "neg_hi", "overperf"),
    ("10 buzz x cheap TE", cheap[cheap.position == "TE"], "buzz_hi", "hit"),
    ("11 fame x cheap", cheap, "fame_hi", "hit"),
    ("12 buzz x H2-riser cheap", cheap[cheap.h2riser == 1], "buzz_hi", "hit"),
]
res = pd.DataFrame([cell_test(*t) for t in TESTS])
print(res.to_string(index=False))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("# Buzz/GDELT subset sweep — 12 pre-registered tests\n\n"
            "delta = flagged-minus-unflagged mean of the outcome; p = permutation two-sided;\n"
            "seasons+ = seasons with positive delta. With 12 tests expect ~0.6 false positives\n"
            "at p<.05 — only CANDIDATE verdicts (p<.05 AND >=65% season consistency) matter,\n"
            "and even those need a mechanism story before touching the board.\n\n```\n"
            + res.to_string(index=False) + "\n```\n")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
