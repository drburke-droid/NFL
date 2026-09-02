"""
Does the draft market OVERBID hyped players, and do overbids underperform?

The hypothesis (bullish fantasy takes -> inflated ADP -> underperformance) is tested
without needing hype text: the overbid itself is measurable as the gap between a
player's market price (ADP) and his projection-implied rank. If hype systematically
misprices players, the most-overbid names should return less than their draft-slot
peers. If the market is instead aggregating real information projections miss, they
should match or beat their slot.

  overbid = log(proj positional rank) - log(market positional rank)
            (>0 = drafted earlier than projections justify)

Outcomes (from the late-breakout frame, 2016-2025):
  vorpDelta = VORP minus median VORP of same season x ADP-bucket x position (slot peers)
  projDelta = actual points minus FFA projection (secondary - mechanically entangled)

Hype overlays:
  - Wiki August pageview spike vs Jan-May baseline (nflv_wiki_buzz, by player_id)
  - Fantasy-filtered GDELT mentions/tone, Jun-Aug draft window (nflv_gdelt_fantasy,
    known-thin coverage - top names only)

Report: outputs/reports/adp_hype.md
"""
import os, re, sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")
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
df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[(df.in_ffa == 1) & (df.adp_f <= 200) & df.position.isin(["QB", "RB", "WR", "TE"])
        & df.ffa_points.notna() & df.vorp.notna()].copy()
df["nm"] = df.name.map(norm)

# market vs projection positional ranks
df["mkt_rank"] = df.groupby(["season", "position"]).adp_f.rank(method="first")
df["proj_rank"] = df.groupby(["season", "position"]).ffa_points.rank(ascending=False, method="first")
df["overbid"] = np.log(df.proj_rank) - np.log(df.mkt_rank)

# outcome vs draft-slot peers
df["bucket"] = pd.cut(df.adp_f, [0, 24, 48, 84, 132, 200],
                      labels=["1-24", "25-48", "49-84", "85-132", "133-200"])
df["vorpD"] = df.vorp - df.groupby(["season", "bucket", "position"], observed=True).vorp.transform("median")
df["projD"] = df.act_pts - df.ffa_points
df["ob_q"] = df.groupby("season").overbid.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))

print("=" * 76)
print("PART 1 — OVERBID (ADP vs projection rank) vs OUTCOME, 2016-2025")
print("=" * 76)
print(f"priced pool: {len(df)} player-seasons (in_ffa, ADP<=200, QB/RB/WR/TE)\n")
print("overbid quintiles (Q4 = most overbid vs projections):")
print(f"   {'quintile':<26}{'n':>5}{'vorpΔ':>9}{'projΔ':>9}{'hit%':>7}{'meanADP':>9}")
for q in sorted(df.ob_q.dropna().unique()):
    sub = df[df.ob_q == q]
    lab = {0: "Q0 most UNDERbid", 4: "Q4 most OVERbid"}.get(q, f"Q{int(q)}")
    print(f"   {lab:<26}{len(sub):>5}{sub.vorpD.mean():>+9.1f}{sub.projD.mean():>+9.1f}"
          f"{sub.hit.mean():>7.1%}{sub.adp_f.mean():>9.0f}")
r1, p1 = stats.spearmanr(df.overbid, df.vorpD)
r2, p2 = stats.spearmanr(df.overbid, df.projD)
print(f"\n   Spearman overbid vs vorpΔ r={r1:+.3f} (p={p1:.4f}) | vs projΔ r={r2:+.3f} (p={p2:.4f})")
for pos in ("QB", "RB", "WR", "TE"):
    sub = df[df.position == pos]
    r, p = stats.spearmanr(sub.overbid, sub.vorpD)
    print(f"   {pos:<3} n={len(sub):>4}  overbid vs vorpΔ r={r:+.3f} (p={p:.3f})")
sub = df[df.adp_f <= 84]
r, p = stats.spearmanr(sub.overbid, sub.vorpD)
print(f"   early rounds only (ADP<=84) n={len(sub)}  r={r:+.3f} (p={p:.3f})")

# ---------------- Part 2: wiki buzz overlay ----------------
print("\n" + "=" * 76)
print("PART 2 — WIKI AUGUST BUZZ x OVERBID")
print("=" * 76)
wb = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
wb["y"] = wb.ym.str[:4].astype(int); wb["m"] = wb.ym.str[4:].astype(int)
aug = wb[wb.m == 8].rename(columns={"views": "aug_v", "y": "season"})[["player_id", "season", "aug_v"]]
base = (wb[wb.m.between(1, 5)].groupby(["player_id", "y"]).views.median()
        .rename("base_v").reset_index().rename(columns={"y": "season"}))
df = df.merge(aug, on=["player_id", "season"], how="left").merge(base, on=["player_id", "season"], how="left")
df["buzz"] = np.log((df.aug_v + 100) / (df.base_v + 100))
bz = df.dropna(subset=["buzz"]).copy()
bz["hiBuzz"] = bz.groupby("season").buzz.rank(pct=True) >= 0.8
bz["hiOB"] = bz.ob_q == 4
print(f"buzz coverage: {len(bz)}/{len(df)} player-seasons\n")
print(f"   {'cell':<34}{'n':>5}{'vorpΔ':>9}{'hit%':>7}")
for lab, sub in (("overbid Q4 + HIGH Aug buzz", bz[bz.hiOB & bz.hiBuzz]),
                 ("overbid Q4 + normal buzz", bz[bz.hiOB & ~bz.hiBuzz]),
                 ("not-overbid + HIGH Aug buzz", bz[~bz.hiOB & bz.hiBuzz]),
                 ("not-overbid + normal buzz", bz[~bz.hiOB & ~bz.hiBuzz])):
    if len(sub):
        print(f"   {lab:<34}{len(sub):>5}{sub.vorpD.mean():>+9.1f}{sub.hit.mean():>7.1%}")
r, p = stats.spearmanr(bz[bz.hiOB].buzz, bz[bz.hiOB].vorpD)
print(f"\n   within overbid Q4: buzz vs vorpΔ r={r:+.3f} (p={p:.3f})")

# ---------------- Part 3: fantasy-filtered GDELT overlay ----------------
print("\n" + "=" * 76)
print("PART 3 — FANTASY-ARTICLE MENTIONS/TONE (GDELT, Jun-Aug) — thin coverage")
print("=" * 76)
try:
    gf = pd.read_sql("SELECT name nm, season, mentions, avg_tone FROM nflv_gdelt_fantasy", con)
except Exception:
    gf = pd.DataFrame(columns=["nm", "season", "mentions", "avg_tone"])
m = df.merge(gf, on=["nm", "season"], how="left")
m["mentions"] = m.mentions.fillna(0)
cov = (m.mentions > 0).mean()
print(f"any fantasy-article mention: {cov:.0%} of priced pool; "
      f">=5 mentions: {(m.mentions>=5).mean():.0%}\n")
r, p = stats.spearmanr(m.mentions, m.overbid)
print(f"   mentions vs overbid    r={r:+.3f} (p={p:.4f})   (is fantasy coverage where the overbids are?)")
r, p = stats.spearmanr(m.mentions, m.vorpD)
print(f"   mentions vs vorpΔ      r={r:+.3f} (p={p:.4f})")
well = m[m.mentions >= 5].dropna(subset=["avg_tone"])
if len(well) >= 30:
    r, p = stats.spearmanr(well.avg_tone, well.vorpD)
    print(f"   tone vs vorpΔ (n={len(well)}, >=5 mentions) r={r:+.3f} (p={p:.3f})")
    hi = well[well.avg_tone > well.avg_tone.median()]
    lo = well[well.avg_tone <= well.avg_tone.median()]
    print(f"   bullish half: vorpΔ {hi.vorpD.mean():+.1f} (n={len(hi)}) | "
          f"bearish half: vorpΔ {lo.vorpD.mean():+.1f} (n={len(lo)})")
    ob_well = well[well.ob_q == 4]
    if len(ob_well) >= 10:
        print(f"   overbid Q4 AND covered by fantasy press (n={len(ob_well)}): "
              f"vorpΔ {ob_well.vorpD.mean():+.1f}, hit {ob_well.hit.mean():.1%}")
else:
    print(f"   only {len(well)} priced players with >=5 mentions — too thin, skipping tone tests")

open(os.path.join(ROOT, "outputs", "reports", "adp_hype.md"), "w", encoding="utf-8").write(
    "# Overbid players (ADP >> projection) — do they underperform?\n\n"
    "Generated by `scripts/adp_hype_study.py` from the late-breakout frame (2016-2025),\n"
    "`nflv_wiki_buzz`, and `nflv_gdelt_fantasy` (fantasy-filtered GDELT, thin coverage).\n\n"
    "```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/adp_hype.md")
