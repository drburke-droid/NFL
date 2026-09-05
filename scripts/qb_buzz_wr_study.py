"""
QB buzz/GDELT -> WR performance study (2016-2025).

Does the QB1's August attention profile predict his WRs' seasons?

  Q1  QB1 NEGATIVE news (GDELT gkg tone/neg, Aug) -> team WRs underperform FFA?
  Q2  QB1 wiki HYPE (Aug spike pctl) -> team WRs over/underperform?
  Q3  Interaction: QB1 news storm x WR's OWN buzz (the handcuff shape, cross-position).
  Q4  Where does any effect live — priced WRs (WR1 / aav>=$10) vs cheap darts?

Same data spine as gdelt_buzz_interaction_study.py: late_breakout_frame (team, FFA
expectation, act_pts, hit, cheap), nflv_gdelt_bq gkg monthly Aug rows, nflv_wiki_buzz
monthly spike. QB1 = team's top QB by FFA AAV (fallback ADP, prior PPG).

Outcomes: overperf = act_pts - ffa_points (points vs projection), beat = overperf>0,
hit (late-breakout label, meaningful only for the cheap pool).

Honesty: every cell reports n, notable deltas get per-season signs + bootstrap p,
and the number of cuts tested is stated up front (multiplicity).
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
OUT = os.path.join(ROOT, "outputs", "reports", "qb_buzz_wr.md")
rng = np.random.default_rng(11)
con = sqlite3.connect(DB)

# ---- signals (identical construction to the RB study) ----
wb = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
wb["year"] = wb.ym.str[:4].astype(int); wb["month"] = wb.ym.str[4:6].astype(int)
rows = []
for (pid, y), g in wb[wb.year.between(2016, 2025)].groupby(["player_id", "year"]):
    m = dict(zip(g.month, g.views))
    aug = m.get(8, np.nan)
    base = np.nanmedian([m.get(k, np.nan) for k in range(1, 6)])
    rows.append({"player_id": pid, "season": y,
                 "spike": np.log((aug + 10) / (base + 10)) if aug == aug and base == base else np.nan})
buzz = pd.DataFrame(rows)
buzz["spike_pctl"] = buzz.groupby("season").spike.rank(pct=True)

gd = pd.read_sql("SELECT player_id, ym, articles, avg_tone, avg_neg FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["year"] = gd.ym.str[:4].astype(int); gd["month"] = gd.ym.str[4:6].astype(int)
aug_news = (gd[(gd.month == 8) & gd.year.between(2016, 2025)]
            .rename(columns={"year": "season"})[["player_id", "season", "articles", "avg_tone", "avg_neg"]])

df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[df.season.between(2016, 2025)].copy()
df["exp_aav"] = df.ffa_aav.fillna(0); df["exp_adp"] = df.ffa_adp.fillna(300)
df["overperf"] = df.act_pts - df.ffa_points
df["beat"] = (df.act_pts > df.ffa_points).astype(float)
df.loc[df.ffa_points.isna() | df.act_pts.isna(), ["overperf", "beat"]] = np.nan
df = df.merge(buzz, on=["player_id", "season"], how="left").merge(aug_news, on=["player_id", "season"], how="left")

# ---- QB1 per team-season with his August signals ----
qb = df[df.position == "QB"].copy()
qb["_k"] = [(-a, b, -c) for a, b, c in zip(qb.exp_aav, qb.exp_adp, qb.prior_ppg.fillna(0))]
qb1 = qb.sort_values("_k").groupby(["team", "season"]).head(1)
qb1 = qb1[(qb1.exp_aav >= 1) | (qb1.exp_adp <= 200)]        # must be a real expected starter
qb1 = qb1[["team", "season", "name", "articles", "avg_tone", "avg_neg", "spike", "spike_pctl",
           "years_exp", "is_rookie", "team_change"]].rename(
    columns={c: "qb_" + c for c in ["name", "articles", "avg_tone", "avg_neg", "spike", "spike_pctl",
                                    "years_exp", "is_rookie", "team_change"]})
qb1["qb_articles"] = qb1.qb_articles.fillna(0)

wr = df[df.position == "WR"].merge(qb1, on=["team", "season"], how="inner")
wr["wr1"] = wr.sort_values("exp_aav", ascending=False).groupby(["team", "season"]).cumcount().eq(0) if False else 0
wr["wr_rank"] = wr.sort_values("exp_aav", ascending=False).groupby(["team", "season"]).cumcount() + 1
wr["priced"] = (wr.exp_aav >= 10).astype(int)
print(f"WR-seasons with a QB1 attached: {len(wr)} ({wr.season.nunique()} seasons, "
      f"{wr.groupby(['team','season']).ngroups} team-seasons)")

REPORT = ["# QB buzz/GDELT -> WR performance", "",
          f"WR player-seasons 2016-25 joined to their team's QB1 August signals: n={len(wr)}. "
          "Outcome overperf = act_pts - FFA projection. Cuts tested: 4 QB-signal quartile tables "
          "x 3 WR pools + 2 interactions (state multiplicity when reading p-values).", ""]

def qtab(sub, col, label, w=None):
    s = sub.dropna(subset=[col, "overperf"]).copy()
    if len(s) < 80: return f"{label}: n={len(s)} too thin"
    s["q"] = pd.qcut(s.groupby("season")[col].rank(pct=True), 4, labels=False, duplicates="drop")
    t = s.groupby("q").agg(n=("overperf", "size"), over=("overperf", "mean"),
                           beat=("beat", "mean"), hit=("hit", "mean")).round(3)
    c = s[[col, "overperf"]].corr().iloc[0, 1]
    return f"--- {label} (n={len(s)}, corr {c:+.3f}) ---\n" + t.to_string()

def boot_delta(a, b, n=4000):
    """mean(a)-mean(b) with permutation p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 8 or len(b) < 8: return np.nan, np.nan
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); la = len(a)
    null = [pool[rng.permutation(len(pool))][:la].mean() - pool[rng.permutation(len(pool))][la:].mean()
            for _ in range(n)]
    return obs, float((np.abs(null) >= abs(obs)).mean())

POOLS = [("ALL WRs", wr), ("priced WRs (aav>=$10)", wr[wr.priced == 1]),
         ("cheap WRs (dart pool)", wr[wr.cheap == 1])]

print("\n============ Q1/Q2: WR overperf by QB1 signal quartile ============")
for sig, lbl in [("qb_avg_tone", "QB1 Aug news TONE (q0=most negative)"),
                 ("qb_avg_neg", "QB1 Aug news NEG-dimension (q3=most negative)"),
                 ("qb_articles", "QB1 Aug article VOLUME"),
                 ("qb_spike_pctl", "QB1 Aug wiki SPIKE pctl")]:
    print(f"\n### {lbl}")
    REPORT.append(f"\n## {lbl}")
    for pl, sub in POOLS:
        t = qtab(sub, sig, pl)
        print(t); REPORT += ["```", t, "```"]

print("\n============ Q3: QB1 storm x WR own buzz (handcuff shape) ============")
REPORT.append("\n## QB1 storm x WR own buzz")
w = wr.dropna(subset=["spike_pctl"]).copy()
elig = w[w.qb_articles >= 3]
storm = (w.qb_articles >= 3) & (w.qb_avg_neg >= elig.qb_avg_neg.quantile(.75))
w["qb_storm"] = storm
w["own_buzz"] = w.spike_pctl >= 0.75
for pl, sub in [("ALL WRs", w), ("cheap WRs", w[w.cheap == 1])]:
    cells = sub.groupby(["qb_storm", "own_buzz"]).agg(
        n=("overperf", "size"), over=("overperf", "mean"), beat=("beat", "mean"), hit=("hit", "mean")).round(3)
    print(f"\n--- {pl} ---"); print(cells.to_string())
    try:
        m = sub.groupby(["qb_storm", "own_buzz"]).overperf.mean()
        inter = (m[True, True] - m[True, False]) - (m[False, True] - m[False, False])
        d, p = boot_delta(sub[sub.qb_storm & sub.own_buzz].overperf, sub[sub.qb_storm & ~sub.own_buzz].overperf)
        print(f"interaction (overperf): {inter:+.1f} | storm cell buzz-vs-quiet delta {d:+.1f} (p={p:.3f})")
        REPORT += ["```", f"{pl}:", cells.to_string(), f"interaction {inter:+.1f}", "```"]
    except KeyError:
        pass

print("\n============ Q4 extras: QB context flags ============")
REPORT.append("\n## QB context flags")
for flag, lbl in [("qb_is_rookie", "rookie QB1"), ("qb_team_change", "QB1 changed teams")]:
    for pl, sub in POOLS:
        a = sub[sub[flag] == 1].overperf; b = sub[sub[flag] == 0].overperf
        d, p = boot_delta(a, b)
        line = f"{lbl} x {pl}: n={a.notna().sum()} vs {b.notna().sum()}, WR overperf delta {d:+.1f} (p={p:.3f})"
        print(line); REPORT += [line, ""]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT) + "\n")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
