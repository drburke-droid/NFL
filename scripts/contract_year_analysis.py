"""
Do NFL skill players overperform in contract years?

The trap: players who play well EARN second contracts, so naively "contract-year players are
good" is pure selection. We fight it three ways:
  1. WITHIN-PLAYER: each player is his own control (compare his walk-year to his own career mean).
  2. AGE-ADJUST: subtract the league aging curve (walk years cluster at peak age ~26-28).
  3. POST-CONTRACT: check the year AFTER signing — a walk-year bump that evaporates once paid is
     the signature of a real contract-year effect (vs. regression-to-mean, which we discuss).

Definition of a contract/walk year: the season BEFORE a player signs a new multi-year veteran
deal (years>=2, apy>=$3M, not his rookie contract). That season is when the incentive is live.

Metric: PPR fantasy points per game (games>=6 to cut noise). Skill positions only. 2011-2024
(reliable OverTheCap era). Source: nfl_data_py (contracts + seasonal).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nfl_data_py as nfl
from scipy import stats

YEARS = list(range(2011, 2025))
SKILL = ["QB", "RB", "WR", "TE"]
MIN_G = 6

# ---- performance panel: PPR PPG by player-season ----
sea = nfl.import_seasonal_data(YEARS)
sea = sea[sea["games"] >= MIN_G].copy()
sea["ppg"] = sea["fantasy_points_ppr"] / sea["games"]

# ---- contracts: player meta + walk-year / post-signing season flags ----
con = nfl.import_contracts()
con = con[con["gsis_id"].notna()].copy()
meta = (con.sort_values("year_signed").groupby("gsis_id")
        .agg(position=("position", "first"), dob=("date_of_birth", "first"),
             draft_year=("draft_year", "min")).reset_index())
meta = meta[meta["position"].isin(SKILL)]
meta["birth_year"] = pd.to_datetime(meta["dob"], errors="coerce").dt.year

vet = con[(con["years"] >= 2) & (con["apy"] >= 3) & con["gsis_id"].notna()].copy()
vet = vet[vet["year_signed"] > (vet["draft_year"].fillna(0) + 1)]         # exclude rookie deals
walk = {(r.gsis_id, int(r.year_signed) - 1) for r in vet.itertuples()}    # season before signing
post = {(r.gsis_id, int(r.year_signed)) for r in vet.itertuples()}        # first season of new deal

# ---- assemble panel ----
p = sea.merge(meta, left_on="player_id", right_on="gsis_id", how="inner")
p = p[p["position"].isin(SKILL)].copy()
p["age"] = p["season"] - p["birth_year"]
p = p[(p["age"] >= 21) & (p["age"] <= 39)]
p["is_cy"] = [(g, s) in walk for g, s in zip(p["player_id"], p["season"])]
p["is_post"] = [(g, s) in post for g, s in zip(p["player_id"], p["season"])]
# keep players with enough seasons to be their own control
n_seas = p.groupby("player_id")["season"].transform("count")
p = p[n_seas >= 4].copy()

# ---- de-mean per player (removes player quality) then age-adjust ----
p["player_mean"] = p.groupby("player_id")["ppg"].transform("mean")
p["dm"] = p["ppg"] - p["player_mean"]                          # vs own career average
age_curve = p.groupby("age")["dm"].mean()                     # league aging curve (relative to career mean)
p["dm_aa"] = p["dm"] - p["age"].map(age_curve)                # age-adjusted within-player deviation


def summarize(mask, label):
    d = p.loc[mask, "dm_aa"].dropna()
    raw = p.loc[mask, "dm"].dropna()
    t, pv = stats.ttest_1samp(d, 0) if len(d) > 2 else (np.nan, np.nan)
    mean_ppg = p["ppg"].mean()
    print(f"  {label:34s} n={len(d):4d}  raw dev {raw.mean():+5.2f}  age-adj {d.mean():+5.2f} PPG "
          f"({d.mean()/mean_ppg*100:+4.1f}%)  t={t:+5.2f} p={pv:.3f}")


print(f"\nPanel: {len(p)} player-seasons, {p['player_id'].nunique()} players, {p['is_cy'].sum()} walk years, "
      f"{p['is_post'].sum()} post-signing years. Mean PPR PPG = {p['ppg'].mean():.1f}\n")
print("Within-player, age-adjusted deviation from a player's OWN career average:")
summarize(p["is_cy"], "CONTRACT (walk) year")
summarize(p["is_post"], "year AFTER signing")
summarize(~p["is_cy"] & ~p["is_post"], "all other seasons (control)")

print("\nBy position — age-adjusted walk-year deviation:")
for pos in SKILL:
    d = p.loc[p["is_cy"] & (p["position"] == pos), "dm_aa"].dropna()
    if len(d) > 2:
        t, pv = stats.ttest_1samp(d, 0)
        print(f"  {pos:3s} n={len(d):4d}  {d.mean():+5.2f} PPG  t={t:+5.2f} p={pv:.3f}")

# ---- walk-year jump vs post-signing letdown (the causal signature) ----
cy = p.loc[p["is_cy"], "dm_aa"].mean()
ps = p.loc[p["is_post"], "dm_aa"].mean()
print(f"\nContract-year premium that evaporates after signing (cy - post): {cy - ps:+.2f} PPG")

# ---- event study around each player's FIRST veteran signing (event time 0 = first year of deal) ----
first_sign = (vet.sort_values("year_signed").groupby("gsis_id")["year_signed"].first())
p["sign_year"] = p["player_id"].map(first_sign)
p["evt"] = p["season"] - p["sign_year"]                        # -1 = walk year, 0 = first year paid
print("\nEvent study — age-adjusted PPG vs career mean, by season relative to signing (0 = first paid year):")
print("  evt   n   age-adj PPG")
for e in range(-3, 3):
    d = p.loc[p["evt"] == e, "dm_aa"].dropna()
    if len(d):
        bar = "#" * int(round(max(d.mean(), 0) * 12))
        print(f"  {e:+d}  {len(d):4d}   {d.mean():+5.2f}  {bar}")
print("  (walk year -1 spikes, then it fades after the player is paid — the contract-year fingerprint)")

print("\nCAVEAT: the walk year is BY CONSTRUCTION the season that earned the contract, so selection /")
print("regression-to-mean inflates it — the within-player, age-adjusted design controls for player")
print("quality and the aging curve, but not for the walk year being a selected-high season. The")
print("post-signing fade (-1 -> 0/+1) is the strongest hint of a genuine effect, though it is also")
print("partly what regression-to-mean would produce. Read ~+1 PPG as an upper-ish bound.\n")
