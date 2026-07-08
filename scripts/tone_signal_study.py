"""
Does GDELT news TONE carry actionable fantasy signal? Two questions:

  1. AUGUST TONE -> DRAFT: the buzz studies (LATE_BREAKOUTS §7-8) showed August news
     VOLUME finds camp-riser darts but is dead for priced players because hype and havoc
     look identical in raw attention. Tone is the proposed disambiguator: high volume +
     positive tone = camp riser; high volume + negative tone = injury/holdout/suspension.
  2. TONE -> WEEK-TO-WEEK: does last month's tone (level, or shift vs the player's own
     baseline) predict next month's per-game scoring relative to the player's season mean?

Data: nflv_gdelt_bq (monthly articles + avg tone per player, GKG Feb-2015+), the
late-breakout frame (player-seasons 2016-25 with price/VORP/hit), nflv_weekly game logs.
League scoring: PPR + 6-pt pass TD - 1 INT. Report: outputs/reports/tone_signal.md
"""
import os, re, sqlite3, collections
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
gd = pd.read_sql("SELECT name, ym, articles, avg_tone FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)

# per player-year: August volume/tone + Jan-May baseline (volume median, tone mean)
aug = gd[gd.m == 8].rename(columns={"articles": "aug_n", "avg_tone": "aug_tone"})[["name", "y", "aug_n", "aug_tone"]]
base = (gd[gd.m.between(1, 5)].groupby(["name", "y"])
        .agg(base_n=("articles", "median"), base_tone=("avg_tone", "mean")).reset_index())
sent = aug.merge(base, on=["name", "y"], how="left")
sent["vol_spike"] = np.log((sent.aug_n + 10) / (sent.base_n.fillna(0) + 10))
sent["tone_shift"] = sent.aug_tone - sent.base_tone

# ---------------- Part 1: August tone -> draft outcomes ----------------
df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[df.season >= 2016].copy()
df["nm"] = df.name.map(norm)
df = df.merge(sent.rename(columns={"name": "nm", "y": "season"}), on=["nm", "season"], how="left")

print("=" * 76)
print("PART 1 — AUGUST TONE vs DRAFT OUTCOMES (2016-2025)")
print("=" * 76)
cov = df.aug_n.notna().mean()
print(f"sentiment coverage: {cov:.0%} of {len(df)} player-seasons\n")

# (a) CHEAP POOL: does tone separate good buzz from bad buzz?
cp = df[(df.cheap == 1) & df.position.isin(["QB", "RB", "WR", "TE"]) & df.aug_n.notna()].copy()
cp["hiVol"] = cp.groupby("season").vol_spike.rank(pct=True) >= 0.8
print(f"(a) DART POOL (cheap, n={len(cp)}, base hit rate {cp.hit.mean():.1%})")
for lab, sub in (("high Aug volume-spike (top 20%)", cp[cp.hiVol]),
                 ("  ... and tone POSITIVE (>+1)", cp[cp.hiVol & (cp.aug_tone > 1)]),
                 ("  ... and tone neutral (-1..+1)", cp[cp.hiVol & cp.aug_tone.between(-1, 1)]),
                 ("  ... and tone NEGATIVE (<-1)", cp[cp.hiVol & (cp.aug_tone < -1)]),
                 ("low/normal volume", cp[~cp.hiVol])):
    if len(sub): print(f"   {lab:<38} n={len(sub):>4}  hit {sub.hit.mean():.1%}  lift {sub.hit.mean()/max(cp.hit.mean(),1e-9):.1f}x")
r, p = stats.spearmanr(cp.aug_tone, cp.hit)
print(f"   pooled Spearman tone vs hit: r={r:+.3f} (p={p:.3f})")

# (b) PRICED POOL: tone × volume vs beating price/projection
pr = df[(df.in_ffa == 1) & (df.adp_f <= 120) & df.aug_n.notna()].copy()
pr["bucket"] = pd.cut(pr.adp_f, [0, 36, 72, 120], labels=["1-36", "37-72", "73-120"])
pr["vorp_res"] = pr.vorp - pr.groupby(["season", "bucket", "position"], observed=True).vorp.transform("median")
pr["proj_res"] = pr.act_pts - pr.ffa_points
pr["hiVol"] = pr.groupby(["season", "bucket"], observed=True).vol_spike.rank(pct=True) >= 0.8
print(f"\n(b) PRICED POOL (ADP<=120, n={len(pr)})")
print("   volume-spike × tone cells (mean VORP vs price peers | mean vs projection | n):")
for vl, vsub in (("HIGH vol-spike", pr[pr.hiVol]), ("normal vol", pr[~pr.hiVol])):
    for tl, tsub in (("tone>+1", vsub[vsub.aug_tone > 1]), ("-1..+1", vsub[vsub.aug_tone.between(-1, 1)]),
                     ("tone<-1", vsub[vsub.aug_tone < -1])):
        if len(tsub) >= 8:
            print(f"   {vl:<15} {tl:<8} vorpΔ {tsub.vorp_res.mean():+6.1f} | projΔ {tsub.proj_res.mean():+6.1f} | n={len(tsub)}")
for col in ("aug_tone", "tone_shift"):
    ok = pr.dropna(subset=[col])
    r1, p1 = stats.spearmanr(ok[col], ok.vorp_res); r2, p2 = stats.spearmanr(ok[col], ok.proj_res)
    print(f"   Spearman {col:<10} vs vorpΔ r={r1:+.3f} (p={p1:.3f}) | vs projΔ r={r2:+.3f} (p={p2:.3f})")

# ---------------- Part 2: tone -> next-month per-game performance ----------------
print("\n" + "=" * 76)
print("PART 2 — LAST MONTH'S TONE vs NEXT MONTH'S PER-GAME SCORING (2015-2025)")
print("=" * 76)
wk = pd.read_sql("""SELECT player_display_name nm, season, week, position,
                           fantasy_points_ppr + 2*COALESCE(passing_tds,0) + COALESCE(passing_interceptions,0) AS pts
                    FROM nflv_weekly WHERE season_type='REG' AND season>=2015
                      AND position IN ('QB','RB','WR','TE')""", con)
wk["nm"] = wk.nm.map(norm)
wk["month"] = np.select([wk.week <= 4, wk.week <= 8, wk.week <= 13, wk.week <= 17], [9, 10, 11, 12], 1)
pm = wk.groupby(["nm", "season", "month", "position"]).pts.agg(["mean", "count"]).reset_index()
pm = pm[pm["count"] >= 2]
seas = wk.groupby(["nm", "season"]).pts.mean().rename("season_mean").reset_index()
pm = pm.merge(seas, on=["nm", "season"])
pm["perf"] = pm["mean"] - pm.season_mean                     # month PPG vs own season mean
# prior-month sentiment (Aug for Sep, Sep for Oct, ...)
gs = gd.rename(columns={"name": "nm"})[["nm", "y", "m", "articles", "avg_tone"]]
gs["pm_month"] = gs.m + 1                                    # sentiment month M predicts month M+1
pm = pm.merge(gs.rename(columns={"y": "season", "pm_month": "month", "articles": "prev_n", "avg_tone": "prev_tone"})
                [["nm", "season", "month", "prev_n", "prev_tone"]], on=["nm", "season", "month"], how="left")
# player's trailing tone baseline (Jan-May same year) for a within-player tone SHIFT
pm = pm.merge(base.rename(columns={"name": "nm", "y": "season"}), on=["nm", "season"], how="left")
pm["tone_shift"] = pm.prev_tone - pm.base_tone
ok = pm.dropna(subset=["prev_tone", "perf"])
print(f"player-months with prior-month tone + >=2 games: {len(ok)}")
for lab, sub in (("all", ok), ("high-volume months (prev_n>=100)", ok[ok.prev_n >= 100]),
                 ("QB", ok[ok.position == "QB"]), ("RB/WR", ok[ok.position.isin(["RB", "WR"])])):
    r1, p1 = stats.spearmanr(sub.prev_tone, sub.perf)
    s2 = sub.dropna(subset=["tone_shift"])
    r2, p2 = stats.spearmanr(s2.tone_shift, s2.perf)
    print(f"  {lab:<34} n={len(sub):>6}  tone-level r={r1:+.3f} (p={p1:.3f})  tone-shift r={r2:+.3f} (p={p2:.3f})")
print("\nextreme-cell check (high volume, prev_n>=100):")
hv = ok[ok.prev_n >= 100]
for lab, sub in (("prev tone < -2 (bad news month)", hv[hv.prev_tone < -2]),
                 ("prev tone -2..+2", hv[hv.prev_tone.between(-2, 2)]),
                 ("prev tone > +2 (good news month)", hv[hv.prev_tone > 2])):
    if len(sub): print(f"  {lab:<34} n={len(sub):>5}  next-month PPG vs own mean: {sub.perf.mean():+.2f}")

open(os.path.join(ROOT, "outputs", "reports", "tone_signal.md"), "w", encoding="utf-8").write(
    "# GDELT tone: August draft signal + week-to-week predictiveness\n\n"
    "Generated by `scripts/tone_signal_study.py` from `nflv_gdelt_bq` (BigQuery GKG,\n"
    "Feb 2015+), the late-breakout frame, and `nflv_weekly` game logs.\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/tone_signal.md")
