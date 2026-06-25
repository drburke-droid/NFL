"""
Is the late-season role surge a valid signal INSIDE a specific subgroup, even
though it's noise on average?  (Anti-overfit discipline below.)

Mechanism under test: full-season prior_ppg mechanically UNDER-states a player
whose role only arrived in the second half. For a YOUNG ascending pass-catcher
who *ends the year entrenched* (high H2 snap share), the model should be
systematically too low -> positive residual next year.

Discipline against overfitting:
  - Subgroups are PRE-SPECIFIED from theory with round-number thresholds (no search).
  - Tested against the walk-forward projection RESIDUAL (next_ppg - model P50),
    i.e. signal BEYOND what the model already knows.
  - Each finding must REPLICATE in two disjoint eras (2013-2019 and 2020-2025);
    a one-era-only effect is called noise.
  - Report n and a bootstrap 95% CI; ignore small-n / CI-crosses-zero groups.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]
BASE = MS.FEATURES
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
RNG = np.random.default_rng(0)


def boot_ci(x, n=2000):
    if len(x) < 8: return (np.nan, np.nan)
    means = [RNG.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con); con.close()
    df = df.merge(ht, on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()

    # --- walk-forward BASE projection -> residual (signal must beat THIS) ---
    parts = []
    for pos in POS:
        d = df[df.position == pos]
        for T in range(2017, 2026):
            tr, te = d[d.season < T], d[d.season == T]
            if len(te) == 0 or len(tr) < 80: continue
            m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB)
            m.fit(tr[BASE].astype(float).fillna(-1), tr.next_ppg)
            t2 = te.copy(); t2["pred"] = m.predict(te[BASE].astype(float).fillna(-1))
            parts.append(t2)
    P = pd.concat(parts, ignore_index=True)
    P["resid"] = P.next_ppg - P.pred                      # +ve = model too LOW
    P = P[P.ht_h2_snap.notna()].copy()                    # need role data (2014+ target)
    posmed = P.groupby("position").prior_ppg.transform("median")

    # --- PRE-SPECIFIED subgroups (theory-driven, fixed thresholds) ---
    young = P.age <= 24
    exp2 = P.years_exp <= 2
    pc = P.position.isin(["WR", "TE"])
    rb = P.position == "RB"
    entrenched = P.ht_h2_snap >= 55                        # ended as a starter
    jump = P.ht_d_snap >= 12                               # role grew in-season
    lowprior = P.prior_ppg < posmed                        # full-season line understates
    G = {
        "ALL (baseline)": pd.Series(True, index=P.index),
        "Young pass-catcher (age<=24, WR/TE)": young & pc,
        "  + ended entrenched (H2 snap>=55%)": young & pc & entrenched,
        "  + entrenched & role jumped (Dsnap>=12)": young & pc & entrenched & jump,
        "YOUR CASE: young WR/TE, low line, won the job": young & pc & entrenched & jump & lowprior,
        "Young RB won backfield (age<=24, entrenched+jump)": young & rb & entrenched & jump,
        "Early-career any pos (exp<=2, entrenched+jump)": exp2 & entrenched & jump,
        "VETERAN late surge (age>=27, entrenched+jump)": (P.age >= 27) & entrenched & jump,
    }

    print(f"Walk-forward residual base: n={len(P)}, mean resid {P.resid.mean():+.2f} (should be ~0)\n")
    print(f"{'subgroup':52s} {'n':>4s} {'meanResid':>10s} {'95% CI':>16s} {'corr(H2snap)':>12s}")
    print("-" * 100)
    keep = {}
    for name, mask in G.items():
        s = P[mask]; n = len(s)
        if n < 12:
            print(f"{name:52s} {n:>4d}   (too small — ignore)"); continue
        mr = s.resid.mean(); lo, hi = boot_ci(s.resid.values)
        r = np.corrcoef(s.ht_h2_snap, s.resid)[0, 1] if n >= 12 else np.nan
        flag = "  <-- model too LOW" if lo > 0 else ("  (CI crosses 0)" if mr > 0 else "")
        print(f"{name:52s} {n:>4d} {mr:>+10.2f} [{lo:>+5.1f},{hi:>+5.1f}]{'':3s} {r:>+12.2f}{flag}")
        keep[name] = mask

    # --- ANTI-OVERFIT: replicate the surviving subgroup in two disjoint eras ---
    print("\n=== Temporal replication (must hold in BOTH eras to be believed) ===")
    print(f"{'subgroup':52s} {'2013-2019':>20s} {'2020-2025':>20s}")
    for name, mask in keep.items():
        if name == "ALL (baseline)": continue
        a = P[mask & (P.season <= 2019)]; b = P[mask & (P.season >= 2020)]
        def cell(s):
            if len(s) < 8: return f"n={len(s)} (thin)"
            lo, hi = boot_ci(s.resid.values); return f"n={len(s)} {s.resid.mean():+.1f} [{lo:+.1f},{hi:+.1f}]"
        print(f"{name:52s} {cell(a):>20s} {cell(b):>20s}")

    # --- name the players this rule would have flagged (face validity) ---
    print("\n=== Who the strongest rule flags historically (young WR/TE, won the job) + how they did ===")
    rule = keep.get("YOUR CASE: young WR/TE, low line, won the job")
    if rule is not None:
        s = P[rule].sort_values("resid", ascending=False)
        for _, r in s.head(14).iterrows():
            print(f"  {r.player_id[:12]:12s} {r.position} {int(r.season)}  "
                  f"H2snap {r.ht_h2_snap:.0f}%  proj {r.pred:.1f} -> actual {r.next_ppg:.1f}  (beat by {r.resid:+.1f})")


if __name__ == "__main__":
    main()
