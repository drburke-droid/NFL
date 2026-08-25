"""
GDELT x wiki-buzz INTERACTION study — do combined signals work where the singles died?

Prior results (buzz_signal_scope): the August wiki spike works in the $1 dart pool
only; GDELT tone alone, lead-lag, and priced-pool buzz are all dead. This tests the
next layer — cross-player and team-level interactions:

  H1  "Handcuff storm": negative August news on a team's RB1 x high wiki buzz on the
      same team's RB2 -> RB2 overperforms (hit / beats FFA projection).
  H2  Same shape at WR (WR1 bad news x WR2 buzz).
  H3  Team-level August buzz (sum of skill-player wiki spikes) -> the team's players
      collectively overperform FFA, or the team's cheap players hit more.

Definitions
  RB1/RB2 by preseason FFA AAV within team-season (fallback ADP, then prior PPG).
  Bad news (Aug of season Y, pre-draft): nflv_gdelt_bq gkg avg_neg / avg_tone /
      article volume; several reasonable cuts tested, all reported (multiplicity!).
  Buzz: log((aug+10)/(jan-may median+10)) monthly wiki spike, season percentile.
  Outcomes: `hit` (late-breakout label), overperf = act_pts - ffa_points, and
      beat = act_pts > ffa_points.

Honesty rules: every cell reports n; interaction tested per-season for sign
consistency; bootstrap p on the interaction delta; no walk-forward claim is made
for anything that isn't at least directionally stable across seasons.
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
OUT = os.path.join(ROOT, "outputs", "reports", "gdelt_buzz_interactions.md")
rng = np.random.default_rng(7)

con = sqlite3.connect(DB)

# ---------------- signals ----------------
wb = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
wb["year"] = wb.ym.str[:4].astype(int); wb["month"] = wb.ym.str[4:6].astype(int)
rows = []
for (pid, y), g in wb[wb.year.between(2016, 2025)].groupby(["player_id", "year"]):
    m = dict(zip(g.month, g.views))
    aug = m.get(8, np.nan)
    base = np.nanmedian([m.get(k, np.nan) for k in range(1, 6)])
    rows.append({"player_id": pid, "season": y, "aug_views": aug,
                 "spike": np.log((aug + 10) / (base + 10)) if aug == aug and base == base else np.nan})
buzz = pd.DataFrame(rows)
buzz["spike_pctl"] = buzz.groupby("season").spike.rank(pct=True)

gd = pd.read_sql("""SELECT player_id, ym, articles, avg_tone, avg_neg
                    FROM nflv_gdelt_bq WHERE source='gkg'""", con)
gd["year"] = gd.ym.str[:4].astype(int); gd["month"] = gd.ym.str[4:6].astype(int)
aug_news = (gd[(gd.month == 8) & gd.year.between(2016, 2025)]
            .rename(columns={"year": "season"})[
                ["player_id", "season", "articles", "avg_tone", "avg_neg"]])

# ---------------- frame + depth chart ----------------
df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[df.season.between(2016, 2025)].copy()
df["exp_aav"] = df.ffa_aav.fillna(0)
df["exp_adp"] = df.ffa_adp.fillna(300)
df["overperf"] = df.act_pts - df.ffa_points
df["beat"] = (df.act_pts > df.ffa_points).astype(float)
df.loc[df.ffa_points.isna() | df.act_pts.isna(), ["overperf", "beat"]] = np.nan
df = df.merge(buzz, on=["player_id", "season"], how="left") \
       .merge(aug_news, on=["player_id", "season"], how="left")

REPORT = ["# GDELT x wiki-buzz interaction study", "",
          f"Universe: late-breakout frame 2016-2025 ({len(df)} player-seasons). "
          "Signals measured in August of the season, pre-draft. "
          "Outcomes: `hit` (late-breakout label), `overperf` = actual pts - FFA projection.", ""]

def depth_pairs(pos):
    """(team, season) -> (P1 row, P2 row) ranked by preseason expectation."""
    p = df[(df.position == pos)].copy()
    p["rankkey"] = list(zip(-p.exp_aav, p.exp_adp, -p.prior_ppg.fillna(0)))
    out = []
    for (t, s), g in p.groupby(["team", "season"]):
        g = g.sort_values(["exp_aav", "prior_ppg"], ascending=False) \
             .sort_values("rankkey", key=lambda k: k)  # tuple sort
        g = p.loc[g.index]
        g = g.assign(_k=[(-a, b, -c) for a, b, c in zip(g.exp_aav, g.exp_adp, g.prior_ppg.fillna(0))]) \
             .sort_values("_k")
        if len(g) < 2: continue
        p1, p2 = g.iloc[0], g.iloc[1]
        if p1.exp_aav < 3 and p1.exp_adp > 150: continue   # no real starter expectation -> skip team
        out.append((p1, p2))
    return out

def boot_p(delta_fn, n=4000):
    obs = delta_fn(None)
    if obs != obs: return obs, np.nan
    null = [delta_fn(rng) for _ in range(n)]
    null = np.array([x for x in null if x == x])
    if not len(null): return obs, np.nan
    return obs, float((np.abs(null) >= abs(obs)).mean())

def interaction_test(pos, news_cut):
    pairs = depth_pairs(pos)
    rec = []
    for p1, p2 in pairs:
        rec.append({"season": p1.season, "team": p1.team,
                    "p1_articles": p1.articles, "p1_tone": p1.avg_tone, "p1_neg": p1.avg_neg,
                    "p2_spike_pctl": p2.spike_pctl, "p2_hit": p2.hit,
                    "p2_over": p2.overperf, "p2_beat": p2.beat,
                    "p2_exp_aav": p2.exp_aav, "p2_cheap": p2.cheap})
    r = pd.DataFrame(rec).dropna(subset=["p2_spike_pctl"])
    # news availability: missing GDELT row for P1 = no articles that month -> neutral, not bad
    r["p1_articles"] = r.p1_articles.fillna(0)
    lbl, bad = news_cut(r)
    r["bad1"] = bad.astype(bool)
    r["buzz2"] = r.p2_spike_pctl >= 0.75
    cells = r.groupby(["bad1", "buzz2"]).agg(n=("p2_hit", "size"), hit=("p2_hit", "mean"),
                                             over=("p2_over", "mean"), beat=("p2_beat", "mean")).round(3)
    # interaction delta on hit: (bad&buzz - bad&nobuzz) - (good&buzz - good&nobuzz)
    def delta(rg):
        rr = r if rg is None else r.assign(bad1=rng.permutation(r.bad1.values))
        try:
            m = rr.groupby(["bad1", "buzz2"]).p2_hit.mean()
            return (m[True, True] - m[True, False]) - (m[False, True] - m[False, False])
        except KeyError:
            return np.nan
    obs, p = boot_p(delta)
    per = r[r.bad1 & r.buzz2].groupby("season").p2_hit.agg(["size", "mean"])
    return lbl, r, cells, obs, p, per

NEWS_CUTS = [
    ("neg-tone Q1 (tone <= 25th pctl of P1s, >=3 articles)",
     lambda r: ("toneQ1", (r.p1_articles >= 3) & (r.p1_tone <= r[r.p1_articles >= 3].p1_tone.quantile(.25)))),
    ("high avg_neg (neg >= 75th pctl of P1s, >=3 articles)",
     lambda r: ("negQ4", (r.p1_articles >= 3) & (r.p1_neg >= r[r.p1_articles >= 3].p1_neg.quantile(.75)))),
    ("news storm (articles >= 75th pctl AND tone below median)",
     lambda r: ("storm", (r.p1_articles >= r.p1_articles.quantile(.75)) & (r.p1_tone <= r.p1_tone.median()))),
]

for pos in ["RB", "WR"]:
    REPORT.append(f"\n## H{'1' if pos=='RB' else '2'} — {pos}1 bad news x {pos}2 wiki buzz")
    print(f"\n================ {pos}1 news x {pos}2 buzz ================")
    for label, cut in NEWS_CUTS:
        try:
            lbl, r, cells, obs, p, per = interaction_test(pos, cut)
        except Exception as e:
            print(f"  {label}: FAILED {e}"); continue
        print(f"\n--- {label}  (pairs n={len(r)}) ---")
        print(cells.to_string())
        print(f"interaction delta on hit: {obs:+.3f}  bootstrap p={p:.3f}")
        print("bad1 & buzz2 cell by season:\n", per.to_string() if len(per) else "  (empty)")
        REPORT += [f"\n### {label} — pairs n={len(r)}", "```", cells.to_string(),
                   f"interaction delta (hit): {obs:+.3f}, bootstrap p={p:.3f}", "```"]

# ---------------- H3: team-level buzz ----------------
print("\n================ H3: team-level August buzz ================")
REPORT.append("\n## H3 — team-level August buzz")
sk = df[df.position.isin(["QB", "RB", "WR", "TE"])].copy()
team = (sk.groupby(["team", "season"])
          .agg(tm_spike=("spike", "median"), tm_views=("aug_views", "sum"),
               tm_over=("overperf", "sum"), n_players=("player_id", "size"),
               tm_beat=("beat", "mean")).reset_index().dropna(subset=["tm_spike"]))
team["spike_q"] = pd.qcut(team.groupby("season").tm_spike.rank(pct=True), 5, labels=False)
tt = team.groupby("spike_q").agg(n=("tm_over", "size"), over=("tm_over", "mean"),
                                 beat=("tm_beat", "mean")).round(2)
print("team overperformance (sum act-ffa pts) by team-buzz quintile:\n", tt.to_string())
c = team[["tm_spike", "tm_over"]].corr().iloc[0, 1]
print(f"corr(team spike, team overperf) = {c:+.3f}")
REPORT += ["```", "team overperf by Aug team-buzz quintile:", tt.to_string(),
           f"corr(team median spike, team sum overperf) = {c:+.3f}", "```"]

# cheap players on buzzy teams
ch = sk[(sk.cheap == 1)].merge(team[["team", "season", "tm_spike"]], on=["team", "season"])
ch["tmq"] = pd.qcut(ch.groupby("season").tm_spike.rank(pct=True), 4, labels=False)
cht = ch.groupby("tmq").agg(n=("hit", "size"), hit=("hit", "mean")).round(3)
print("\ncheap-player hit rate by TEAM buzz quartile:\n", cht.to_string())
own = ch[ch.spike_pctl >= 0.75].groupby("tmq").agg(n=("hit", "size"), hit=("hit", "mean")).round(3)
print("cheap + OWN buzz>=75pctl, by TEAM buzz quartile:\n", own.to_string())
REPORT += ["```", "cheap-player hit rate by team-buzz quartile:", cht.to_string(),
           "cheap + own buzz >=75pctl by team-buzz quartile:", own.to_string(), "```"]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT) + "\n")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
