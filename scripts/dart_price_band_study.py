"""
Is paying up (~$10) for dart throws +EV? Price-band study on UPSIDE-profile players.

"Dart profile" = the player you buy for what he MIGHT become, not what he is:
  age <= 26 AND not an established producer (prior season < 12 PPG, or no prior
  season at all — rookies included), skill positions, 2016-2025, priced by FFA
  (aav > 0 so a real market existed).

Bins by FFA AAV: $1-2, $3-5, $6-9, $10-14, $15-24.
Outcomes per bin:
  hit         late-breakout label (startable-season)
  vorp        season value over replacement (risk of the roster spot)
  vorp/$      value per auction dollar
  hit/$       startable-hit probability per dollar  (the spread-vs-concentrate math)
  next_aav    next-season market price (keeper-value proxy)
  ksurplus    next_aav - price paid (what the keeper right nets you)
Per-slot vs per-dollar: 7 bench spots are as scarce as dollars late — a $12 dart
costs 1 spot + $12; four $3 darts cost 4 spots + $12.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "reports", "dart_price_bands.md")

df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[df.season.between(2016, 2025) & df.position.isin(["RB", "WR", "TE"])].copy()
df["aav"] = df.ffa_aav.fillna(0)
prof = df[(df.age.fillna(24) <= 26) & (df.prior_ppg.fillna(0) < 12) & (df.aav >= 1) & (df.aav <= 24)].copy()
prof["ksurplus"] = prof.next_aav.fillna(0) - prof.aav

BINS = [(1, 2.99, "$1-2"), (3, 5.99, "$3-5"), (6, 9.99, "$6-9"), (10, 14.99, "$10-14"), (15, 24, "$15-24")]
rows = []
for lo, hi, lbl in BINS:
    b = prof[(prof.aav >= lo) & (prof.aav <= hi)]
    if not len(b): continue
    mid = b.aav.mean()
    rows.append({
        "band": lbl, "n": len(b), "avg$": round(mid, 1),
        "hit%": round(100 * b.hit.mean(), 1),
        "hit/$": round(100 * b.hit.mean() / mid, 2),
        "vorp": round(b.vorp.mean(), 0),
        "vorp/$": round(b.vorp.mean() / mid, 1),
        "next$": round(b.next_aav.fillna(0).mean(), 1),
        "ksurp": round(b.ksurplus.mean(), 1),
        "bigK%": round(100 * (b.ksurplus >= 10).mean(), 1),
    })
res = pd.DataFrame(rows)
print("UPSIDE-PROFILE (age<=26, prior<12 ppg incl. rookies), RB/WR/TE 2016-25:\n")
print(res.to_string(index=False))

# per-season consistency of the key comparison: hit% $10-14 vs 3x-the-$ of $1-2
per = []
for yr, g in prof.groupby("season"):
    a = g[(g.aav >= 10) & (g.aav < 15)]; c = g[g.aav < 3]
    if len(a) >= 3 and len(c) >= 10:
        per.append({"season": yr, "n10": len(a), "hit10": round(a.hit.mean(), 2),
                    "n1": len(c), "hit1": round(c.hit.mean(), 2)})
per = pd.DataFrame(per)
print("\nper-season $10-14 vs $1-2 hit rates:\n", per.to_string(index=False))

# the slot math: one $12 dart vs four $3 darts (same $)
h10 = prof[(prof.aav >= 10) & (prof.aav < 15)].hit.mean()
h3 = prof[(prof.aav >= 3) & (prof.aav < 6)].hit.mean()
h1 = prof[prof.aav < 3].hit.mean()
print(f"\nP(>=1 hit): one $12 dart = {h10:.0%} | four $3 darts (4 slots) = {1-(1-h3)**4:.0%} | "
      f"twelve $1 darts (12 slots!) = {1-(1-h1)**12:.0%}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("# Dart price bands — is paying up ~$10 for upside +EV?\n\n"
            "Upside profile: age<=26, prior season <12 PPG (rookies included), RB/WR/TE, 2016-25, FFA-priced.\n\n"
            "```\n" + res.to_string(index=False) + "\n\nper-season $10-14 vs $1-2:\n"
            + per.to_string(index=False)
            + f"\n\nP(>=1 hit): one $12 dart {h10:.0%} | four $3 darts {1-(1-h3)**4:.0%} | twelve $1 darts {1-(1-h1)**12:.0%}\n```\n")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
