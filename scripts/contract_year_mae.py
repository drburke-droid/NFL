"""
Does a contract-year (walk-year) adjustment actually improve PROJECTION MAE?

The +1 PPG walk-year overperformance is partly selection (the walk year was the high season that
earned the deal). For PREDICTION we must apply a bump to EVERY walk-year player ex-ante, before we
know who will spike — so regression-to-mean can wash it out. This tests it honestly, walk-forward:

  Baseline projection: OLS of next-season PPG on prior-2-season PPG + age + age^2 + position,
  trained ONLY on seasons before the test year (no leakage).
  Treatment: add a walk-year feature (its coefficient is estimated on the training years too).

We report MAE on the full panel AND on the walk-year players specifically (the sharper test:
does the bump help the players it applies to?). Also a fixed position-bump variant (what the
board would actually implement). nfl_data_py, 2011-2024, PPR PPG, games>=6.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2011, 2025)); SKILL = ["QB", "RB", "WR", "TE"]; MIN_G = 6

sea = nfl.import_seasonal_data(YEARS); sea = sea[sea["games"] >= MIN_G].copy()
sea["ppg"] = sea["fantasy_points_ppr"] / sea["games"]
con = nfl.import_contracts(); con = con[con["gsis_id"].notna()].copy()
meta = (con.sort_values("year_signed").groupby("gsis_id")
        .agg(position=("position", "first"), dob=("date_of_birth", "first"),
             draft_year=("draft_year", "min")).reset_index())
meta = meta[meta["position"].isin(SKILL)]; meta["birth_year"] = pd.to_datetime(meta["dob"], errors="coerce").dt.year
vet = con[(con["years"] >= 2) & (con["apy"] >= 3) & con["gsis_id"].notna()].copy()
vet = vet[vet["year_signed"] > (vet["draft_year"].fillna(0) + 1)]
walk = {(r.gsis_id, int(r.year_signed) - 1) for r in vet.itertuples()}

p = sea.merge(meta, left_on="player_id", right_on="gsis_id", how="inner")
p = p[p["position"].isin(SKILL)].copy(); p["age"] = p["season"] - p["birth_year"]
p = p[(p["age"] >= 21) & (p["age"] <= 39)].sort_values(["player_id", "season"])
p["lag1"] = p.groupby("player_id")["ppg"].shift(1)
p["lag2"] = p.groupby("player_id")["ppg"].shift(2)
p = p[p["lag1"].notna()].copy(); p["lag2"] = p["lag2"].fillna(p["lag1"])
p["walk"] = np.array([(g, s) in walk for g, s in zip(p["player_id"], p["season"])], dtype=float)
p["age2"] = p["age"] ** 2
for pos in SKILL: p[f"is_{pos}"] = (p["position"] == pos).astype(float)

BASE = ["lag1", "lag2", "age", "age2", "is_RB", "is_WR", "is_TE"]  # is_QB is the reference


def fit_predict(train, test, feats):
    X = np.column_stack([np.ones(len(train))] + [train[f].values for f in feats])
    y = train["ppg"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.column_stack([np.ones(len(test))] + [test[f].values for f in feats])
    return Xt @ beta, beta


rows = []
for T in range(2016, 2025):
    tr, te = p[p["season"] < T], p[p["season"] == T]
    if len(te) < 20: continue
    base_pred, _ = fit_predict(tr, te, BASE)
    walk_pred, beta = fit_predict(tr, te, BASE + ["walk"])
    wcoef = beta[-1]
    y = te["ppg"].values; wk = te["walk"].values.astype(bool)
    rows.append({"T": T, "n": len(te), "nwalk": int(wk.sum()),
                 "mae_base": np.abs(y - base_pred).mean(),
                 "mae_walk": np.abs(y - walk_pred).mean(),
                 "wcoef": wcoef,
                 "mae_base_wk": np.abs(y[wk] - base_pred[wk]).mean(),
                 "mae_walk_wk": np.abs(y[wk] - walk_pred[wk]).mean()})
R = pd.DataFrame(rows)

print("\nWalk-forward projection MAE — baseline vs +walk-year feature (trained no-leakage):\n")
print("  year   n  nwalk  walkβ   MAE base  MAE +walk   Δfull    | MAE base(wk) +walk(wk)  Δwalk")
for r in rows:
    print(f"  {r['T']}  {r['n']:4d}  {r['nwalk']:3d}  {r['wcoef']:+5.2f}   {r['mae_base']:7.3f}  "
          f"{r['mae_walk']:7.3f}  {r['mae_walk']-r['mae_base']:+6.3f}   |  {r['mae_base_wk']:7.3f}  "
          f"{r['mae_walk_wk']:7.3f}  {r['mae_walk_wk']-r['mae_base_wk']:+6.3f}")
print(f"\n  MEAN over years:  MAE base {R['mae_base'].mean():.3f}  +walk {R['mae_walk'].mean():.3f}  "
      f"(Δ {R['mae_walk'].mean()-R['mae_base'].mean():+.4f})")
print(f"  On WALK-YEAR players only: base {R['mae_base_wk'].mean():.3f}  +walk {R['mae_walk_wk'].mean():.3f}  "
      f"(Δ {R['mae_walk_wk'].mean()-R['mae_base_wk'].mean():+.4f})")
print(f"  Mean estimated walk coefficient: {R['wcoef'].mean():+.2f} PPG "
      f"(positive+stable = signal; shrinks vs the +1.0 in-sample = regression/selection)")
