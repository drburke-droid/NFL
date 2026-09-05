"""Franchise/regime fantasy-factory study, 1999-2025.
Startable = QB top-12, RB top-24, WR top-36, TE top-12 by season PPR (~84/season,
matching the repo's dart-hit convention).
Lens A: which franchises/eras put startable players on the field (Poisson + BH FDR;
        5-yr regime windows vs multinomial permutation null).
Lens B: which franchises' DRAFTED skill players yield startable seasons vs
        round-adjusted expectation (development skill)."""
import sqlite3, os
import numpy as np, pandas as pd
import scipy.stats as st

SCRATCH = os.path.dirname(os.path.abspath(__file__))
root = r"C:\Users\drbur\Documents\GitHub\NFL"
con = sqlite3.connect(os.path.join(root, "db", "nfl_odds.db"))

PARQ = os.path.join(SCRATCH, "player_stats_season.parquet")
if not os.path.exists(PARQ):
    hist = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/"
                           "player_stats/player_stats_season.parquet")
    hist.to_parquet(PARQ)
hist = pd.read_parquet(PARQ)
hist = hist[(hist.season_type == "REG")][
    ["season", "player_id", "player_display_name", "position", "recent_team", "fantasy_points_ppr"]]
cur = pd.read_sql("""SELECT season, player_id, player_display_name, position,
                     recent_team, fantasy_points_ppr FROM nflv_season WHERE season=2025""", con)
df = pd.concat([hist[hist.season <= 2024], cur], ignore_index=True)
df = df[df.position.isin(["QB", "RB", "WR", "TE"])].dropna(subset=["fantasy_points_ppr"])

FR = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "SL": "LA"}
df["franchise"] = df.recent_team.replace(FR)

TOPN = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
df["pos_rank"] = df.groupby(["season", "position"]).fantasy_points_ppr.rank(
    ascending=False, method="first")
df["startable"] = df.apply(lambda r: r.pos_rank <= TOPN[r.position], axis=1)
S = df[df.startable].copy()
seasons = sorted(df.season.unique())
print(f"seasons {seasons[0]}-{seasons[-1]} | startable/season = "
      f"{S.groupby('season').size().mean():.0f}")

# per-season team counts (fill 0 for teams with none)
teams_by_season = df.groupby("season").franchise.unique().to_dict()
cnt = S.groupby(["season", "franchise"]).size().rename("k").reset_index()
full = []
for y in seasons:
    tt = [t for t in teams_by_season[y] if isinstance(t, str) and len(t) <= 3]
    base = pd.DataFrame({"season": y, "franchise": tt})
    full.append(base)
panel = pd.concat(full).merge(cnt, on=["season", "franchise"], how="left").fillna({"k": 0})
panel = panel[panel.franchise != "FA"]
slots = S.groupby("season").size().rename("slots").reset_index()
nteams = panel.groupby("season").franchise.nunique().rename("nteams").reset_index()
panel = panel.merge(slots, on="season").merge(nteams, on="season")
panel["exp"] = panel.slots / panel.nteams

# ---------- LENS A1: franchise totals ----------
fr = panel.groupby("franchise").agg(k=("k", "sum"), exp=("exp", "sum"), yrs=("season", "size"))
fr = fr[fr.yrs >= 20]  # full-history franchises
fr["ratio"] = fr.k / fr.exp
fr["p"] = [st.poisson.sf(k - 1, e) if k > e else st.poisson.cdf(k, e)
           for k, e in zip(fr.k, fr.exp)]
fr["p"] = (fr.p * 2).clip(upper=1)  # two-sided
def bh_fdr(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q

fr["q"] = bh_fdr(fr.p)
fr = fr.sort_values("ratio", ascending=False)
print("\n[A1] FRANCHISE TOTALS 1999-2025 (exp ~", round(fr.exp.mean(), 1), "per franchise):")
print(fr.round(3).to_string())

# ---------- LENS A2: 5-year regime windows ----------
W = 5
pv = panel.pivot_table(index="season", columns="franchise", values="k", aggfunc="sum")
# HOU stays NaN pre-2002 (not yet a franchise); every other gap is a genuine 0
pv.loc[:, pv.columns != "HOU"] = pv.loc[:, pv.columns != "HOU"].fillna(0)
roll = pv.rolling(W).sum()  # windows crossing HOU's birth stay NaN; stack() drops them
rng = np.random.default_rng(7)
NSIM = 10000
sl = slots.set_index("season").slots
nt = nteams.set_index("season").nteams
sim_max = np.zeros(NSIM); sim_min = np.zeros(NSIM)
years = list(pv.index)
slot_arr = np.array([sl[y] for y in years]); nt_arr = np.array([nt[y] for y in years])
for s in range(NSIM):
    rows = []
    for i in range(len(years)):
        row = rng.multinomial(slot_arr[i], np.ones(nt_arr[i]) / nt_arr[i]).astype(float)
        if nt_arr[i] < 32:  # pre-2002: 32nd franchise not yet born
            row = np.append(row, [np.nan] * (32 - nt_arr[i]))
        rows.append(row)
    r = pd.DataFrame(np.vstack(rows)).rolling(W).sum().dropna(how="all").values
    sim_max[s] = np.nanmax(r); sim_min[s] = np.nanmin(r)
crit_hi = np.quantile(sim_max, 0.95); crit_lo = np.quantile(sim_min, 0.05)
print(f"\n[A2] 5-YR REGIME WINDOWS (null 95th pctl of best-anywhere run = {crit_hi:.0f}, "
      f"5th pctl of worst = {crit_lo:.0f}; league-avg 5-yr = {5 * (fr.exp.mean() / 27):.1f}x5"
      f" = {5 * fr.exp.mean() / 27:.1f})")
stack = roll.stack().rename("k5").reset_index()
best = stack.nlargest(15, "k5")
worst = stack.nsmallest(15, "k5")
print("hottest 5-yr runs (window END year):")
for _, r in best.iterrows():
    flag = "***" if r.k5 >= crit_hi else ""
    print(f"  {r.franchise} {int(r.season) - 4}-{int(r.season)}: {r.k5:.0f} startable {flag}")
print("coldest 5-yr runs:")
for _, r in worst.iterrows():
    flag = "***" if r.k5 <= crit_lo else ""
    print(f"  {r.franchise} {int(r.season) - 4}-{int(r.season)}: {r.k5:.0f} startable {flag}")

# ---------- LENS A3: persistence ----------
lag = pv.shift(1).stack().rename("k_prev").reset_index().merge(
    pv.stack().rename("k").reset_index(), on=["season", "franchise"]).dropna()
rho = st.spearmanr(lag.k_prev, lag.k)
print(f"\n[A3] persistence: spearman(count_t, count_t+1) = {rho.statistic:.3f} (p={rho.pvalue:.1e})")
tr5 = pv.rolling(5).mean().shift(1).stack().rename("trail5").reset_index().merge(
    pv.stack().rename("k").reset_index(), on=["season", "franchise"]).dropna()
lag2 = pv.shift(1).stack().rename("k_prev").reset_index()
tr5 = tr5.merge(lag2, on=["season", "franchise"]).dropna()
X2 = np.column_stack([np.ones(len(tr5)), tr5.k_prev.values, tr5.trail5.values])
yv2 = tr5.k.values
beta, *_ = np.linalg.lstsq(X2, yv2, rcond=None)
res = yv2 - X2 @ beta
sig2 = (res**2).sum() / (len(yv2) - 3)
se = np.sqrt(np.diag(sig2 * np.linalg.inv(X2.T @ X2)))
tstat = beta / se
pvals = 2 * st.t.sf(np.abs(tstat), len(yv2) - 3)
print("    k ~ k_prev + trail5yr:  coefs",
      dict(zip(["const", "k_prev", "trail5"], np.round(beta, 3))),
      "p(k_prev) =", f"{pvals[1]:.4f}", "p(trail5) =", f"{pvals[2]:.4f}")

# coach-regime attribution 2011+ (nflv_coaches)
co = pd.read_sql("SELECT team, season, hc FROM nflv_coaches", con)
co["franchise"] = co.team.replace(FR)
cp = panel.merge(co[["franchise", "season", "hc"]], on=["franchise", "season"], how="inner")
reg = cp.groupby(["franchise", "hc"]).agg(k=("k", "sum"), exp=("exp", "sum"),
                                          yrs=("season", "size")).reset_index()
reg = reg[reg.yrs >= 4]
reg["ratio"] = reg.k / reg.exp
reg["p"] = [min(1, 2 * (st.poisson.sf(k - 1, e) if k > e else st.poisson.cdf(k, e)))
            for k, e in zip(reg.k, reg.exp)]
reg["q"] = bh_fdr(reg.p)
print("\n[A4] HC REGIMES 2011-25 (>=4 yrs), top/bottom by ratio:")
rs = reg.sort_values("ratio", ascending=False)
print(pd.concat([rs.head(10), rs.tail(10)]).round(3).to_string(index=False))

# ---------- LENS B: drafting franchise, round-adjusted ----------
dr = pd.read_sql("""SELECT season draft_year, round, pick, team, gsis_id, position
                    FROM nflv_draft WHERE season BETWEEN 1999 AND 2020
                    AND position IN ('QB','RB','WR','TE')""", con)
PFR = {"SDG": "LAC", "NOR": "NO", "GNB": "GB", "NWE": "NE", "KAN": "KC", "SFO": "SF",
       "TAM": "TB", "LVR": "LV", "RAI": "LV", "OAK": "LV", "RAM": "LA", "STL": "LA",
       "CRD": "ARI", "CLT": "IND", "HTX": "HOU", "OTI": "TEN", "RAV": "BAL", "SEA": "SEA"}
dr["franchise"] = dr.team.replace({**FR, **PFR})
sp = S.groupby("player_id").size().rename("startable_seasons").reset_index()
dr = dr.merge(sp, left_on="gsis_id", right_on="player_id", how="left").fillna(
    {"startable_seasons": 0})
rmean = dr.groupby("round").startable_seasons.mean().rename("exp_ss")
dr = dr.merge(rmean, on="round")
fb = dr.groupby("franchise").agg(picks=("gsis_id", "size"), obs=("startable_seasons", "sum"),
                                 exp=("exp_ss", "sum"))
fb["ratio"] = fb.obs / fb.exp
# bootstrap: resample players within round
NB = 3000
null = {f: np.zeros(NB) for f in fb.index}
by_round = {r: g.startable_seasons.values for r, g in dr.groupby("round")}
fr_rounds = dr.groupby("franchise")["round"].apply(list)
for f, rounds in fr_rounds.items():
    draws = np.vstack([rng.choice(by_round[r], NB) for r in rounds]).sum(axis=0)
    null[f] = draws
fb["p"] = [min(1, 2 * min((null[f] >= fb.loc[f, "obs"]).mean(),
                          (null[f] <= fb.loc[f, "obs"]).mean())) for f in fb.index]
fb["q"] = bh_fdr(fb.p)
print("\n[B] DRAFT-AND-DEVELOP, classes 1999-2020 (startable seasons by drafting team, "
      "round-adjusted):")
print(fb.sort_values("ratio", ascending=False).round(3).to_string())

pv.to_csv(os.path.join(SCRATCH, "franchise_panel.csv"))
con.close()
