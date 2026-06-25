"""
Player ARCHETYPES: do specific player types (pass-catching RB, undersized/slot WR,
fast deep-threat WR, ...) age differently, and does projecting by archetype improve
accuracy?

Archetypes are defined ONLY from pre-known traits (combine size/speed + prior-year
usage), so they're leakage-free. Two analyses:
  (1) DESCRIPTIVE aging: mean YoY change in PPG (next_ppg - prior_ppg) by age bucket,
      per archetype -> do burners decline earlier? do receiving backs last longer?
  (2) PREDICTIVE: BASE vs BASE+archetype-features (dummies + archetype*age interactions)
      walk-forward, per position. No per-archetype models (keeps full sample).
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
BASE = MS.FEATURES
BUST = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}; ELITE = {"QB": 21, "RB": 16, "WR": 15, "TE": 12}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def archetype(r):
    """Primary archetype from size/speed/usage (pre-known). One label per player-season."""
    p = r.position
    rec_sh = r.prior_receptions_pg / max(r.prior_carries_pg + r.prior_receptions_pg, 1e-6)
    if p == "RB":
        if r.prior_targets_pg >= 3.0 or rec_sh >= 0.28: return "RB_receiving"
        if r.prior_carries_pg >= 11: return "RB_workhorse"
        return "RB_rotational"
    if p == "WR":
        fast = pd.notna(r.forty) and r.forty <= 4.45
        small = r.height <= 71 and (pd.isna(r.weight) or r.weight <= 190)
        big = r.height >= 74 and (pd.notna(r.weight) and r.weight >= 210)
        deep = pd.notna(r.prior_air_yards_share) and r.prior_air_yards_share >= 0.32
        if fast and deep: return "WR_deep_threat"
        if small: return "WR_slot_small"
        if big: return "WR_big_possession"
        return "WR_balanced"
    if p == "TE":
        if r.prior_target_share >= 0.14 or r.prior_targets_pg >= 4.5: return "TE_receiving"
        return "TE_inline"
    if p == "QB":
        if r.prior_carries_pg >= 5.0: return "QB_mobile"
        return "QB_pocket"
    return "other"


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con); con.close()
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    df["bmi"] = 703 * df.weight / (df.height ** 2)
    df["arch"] = df.apply(archetype, axis=1)
    df["d_ppg"] = df.next_ppg - df.prior_ppg
    df["agebin"] = pd.cut(df.age, [0, 24, 26, 28, 30, 99], labels=["≤24", "25-26", "27-28", "29-30", "31+"])

    print("=== Archetype counts (player-seasons, 2012-2025) ===")
    print(df.arch.value_counts().to_string())

    print("\n=== DO THEY AGE DIFFERENTLY?  mean YoY ΔPPG (next - prior) by age bucket ===")
    print("    (established players only, prior_ppg >= 6; negative = declining)\n")
    est = df[df.prior_ppg >= 6]
    for pos in ["RB", "WR", "TE", "QB"]:
        sub = est[est.position == pos]
        tab = sub.pivot_table(index="arch", columns="agebin", values="d_ppg", aggfunc="mean", observed=True)
        cnt = sub.pivot_table(index="arch", columns="agebin", values="d_ppg", aggfunc="size", observed=True)
        print(f"-- {pos} --")
        for arch in tab.index:
            cells = []
            for ab in ["≤24", "25-26", "27-28", "29-30", "31+"]:
                if ab in tab.columns and pd.notna(tab.loc[arch, ab]):
                    n = int(cnt.loc[arch, ab]); v = tab.loc[arch, ab]
                    cells.append(f"{ab}:{v:+4.1f}(n{n})" if n >= 8 else f"{ab}: --  ")
                else: cells.append(f"{ab}: --  ")
            print(f"  {arch:18s} " + "  ".join(cells))
        print()

    # ---- (2) Predictive: does archetype add value beyond BASE? ----
    EXTRA = ["height", "bmi", "vertical", "broad_jump"]          # combine traits not in BASE
    archs = sorted(df.arch.unique())
    for a in archs:
        df[f"is_{a}"] = (df.arch == a).astype(int)
        df[f"{a}_x_age"] = df[f"is_{a}"] * df.age                # archetype-specific aging
    ARCHF = [f"is_{a}" for a in archs] + [f"{a}_x_age" for a in archs]

    def wf(feats, pos, target=None, line=None):
        d = df[df.position == pos].copy()
        if target: d["y"] = (d.next_ppg < line[pos]).astype(int) if target == "bust" else (d.next_ppg >= line[pos]).astype(int)
        rows = []
        for T in range(2017, 2026):
            tr, te = d[d.season < T], d[d.season == T]
            if len(te) == 0 or len(tr) < 80: continue
            if target:
                if tr.y.sum() < 12: continue
                m = lgb.LGBMClassifier(objective="binary", **GB); m.fit(tr[feats].astype(float).fillna(-1), tr.y)
                te = te.copy(); te["p"] = m.predict_proba(te[feats].astype(float).fillna(-1))[:, 1]
            else:
                m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB); m.fit(tr[feats].astype(float).fillna(-1), tr.next_ppg)
                te = te.copy(); te["pred"] = m.predict(te[feats].astype(float).fillna(-1))
            rows.append(te)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    print("=== PREDICTIVE: central PPG MAE, BASE vs +combine vs +archetype ===")
    print(f"{'pos':5s} {'BASE':>7s} {'+combine':>9s} {'+arch':>8s}")
    feature_sets = [("BASE", BASE), ("+combine", BASE + EXTRA), ("+arch", BASE + EXTRA + ARCHF)]
    for pos in ["RB", "WR", "TE", "QB"]:
        maes = []
        for lab, fs in feature_sets:
            r = wf(fs, pos)
            maes.append(mean_absolute_error(r.next_ppg, r.pred) if len(r) else np.nan)
        print(f"{pos:5s} {maes[0]:7.3f} {maes[1]:9.3f} {maes[2]:8.3f}")

    print("\n=== PREDICTIVE: bust/boom AUC, BASE vs +archetype ===")
    for tgt, line in [("bust", BUST), ("boom", ELITE)]:
        for pos in ["RB", "WR", "TE"]:
            b = wf(BASE, pos, tgt, line); h = wf(BASE + EXTRA + ARCHF, pos, tgt, line)
            if not len(b) or not len(h): continue
            yb = (b.next_ppg < line[pos]).astype(int) if tgt == "bust" else (b.next_ppg >= line[pos]).astype(int)
            yh = (h.next_ppg < line[pos]).astype(int) if tgt == "bust" else (h.next_ppg >= line[pos]).astype(int)
            print(f"  {tgt:5s} {pos}: BASE {roc_auc_score(yb,b.p):.3f} -> +arch {roc_auc_score(yh,h.p):.3f} ({roc_auc_score(yh,h.p)-roc_auc_score(yb,b.p):+.3f})")


if __name__ == "__main__":
    main()
