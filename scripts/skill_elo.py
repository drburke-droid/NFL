"""Player-skill 'Elo': opponent- and teammate-adjusted per-play skill.

Adjusted plus-minus via ridge regression, per position x season:
    epa_play(player-week) = intercept + player_effect + defense_effect(opp) + e
weighted by plays. Teammates in the same game share the defense term, so a
player is implicitly measured against his own team's context; each defense
term is identified from every offense that faced it (the cross-team signal).
For WR the defense effect is split by depth rank (WR1 vs WR2/3) - a shadow
corner suppresses WR1s specifically, and the model can see it.

Adjusted season effects are era-z-scored, then Kalman-smoothed over careers
(same machinery as skill_kalman.py) -> a causal, opponent-adjusted skill
rating: the closest thing to an NFL player Elo this data supports.

Validation at every step (all out-of-time):
  - ridge alpha chosen by corr(effect_t, NEXT season raw skill)
  - YoY stickiness: adjusted vs raw
  - one-step-ahead prediction vs the unadjusted Kalman pipeline
  - WR1-specific defense effects (the lockdown-corner test)

Writes table player_skill_elo.
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("sk", os.path.join(HERE, "skill_kalman.py"))
SK = importlib.util.module_from_spec(_s)
# skill_kalman runs main() on import guard; safe to exec (guarded by __main__)
_s.loader.exec_module(SK)

DB = os.path.join(os.path.dirname(HERE), "db", "nfl_odds.db")
POS_ROLE = {"QB": "pass", "RB": "rush", "WR": "rec", "TE": "rec"}
MIN_WK_PLAYS = {"QB": 10, "RB": 5, "WR": 3, "TE": 3}
SEASONS = range(2011, 2026)

con = sqlite3.connect(DB)
wk = pd.read_sql("""SELECT gsis_id AS player_id, season, week, role, plays, epa_play
                    FROM nflv_pbp_skill_wk WHERE epa_play IS NOT NULL""", con)
pos = pd.read_sql("SELECT DISTINCT player_id, position FROM player_skill_seasons", con)
opp = pd.read_sql("""SELECT player_id, season, week, team, opponent_team
                     FROM nflv_weekly""", con).drop_duplicates(["player_id", "season", "week"])
raw = pd.read_sql("""SELECT s.player_id, s.position, s.season, s.plays, s.skill_z,
                            t.skill_true_causal
                     FROM player_skill_seasons s
                     JOIN player_skill_true t USING(player_id, season)""", con)
names = pd.read_sql("SELECT DISTINCT player_id, player_display_name AS name FROM nflv_traj", con)

wk = wk.merge(pos, on="player_id").merge(opp, on=["player_id", "season", "week"], how="inner")
wk = wk[wk.apply(lambda r: POS_ROLE.get(r["position"]) == r["role"], axis=1)]
wk = wk[wk["plays"] >= wk["position"].map(MIN_WK_PLAYS)]

# WR depth rank within team-season (by total targets) for rank-split defense effects
tt = wk[wk["position"] == "WR"].groupby(["team", "season", "player_id"])["plays"].sum().reset_index()
tt["wrrank"] = tt.groupby(["team", "season"])["plays"].rank(ascending=False, method="first")
wk = wk.merge(tt[["team", "season", "player_id", "wrrank"]], on=["team", "season", "player_id"], how="left")


def fit_season(d, position, alpha):
    """Ridge APM for one position-season. Returns (player effects, defense effects)."""
    d = d.copy()
    if position == "WR":
        d["defkey"] = d["opponent_team"] + np.where(d["wrrank"] == 1, "@1", "@23")
    else:
        d["defkey"] = d["opponent_team"]
    Xp = pd.get_dummies(d["player_id"], sparse=False)
    Xd = pd.get_dummies(d["defkey"], sparse=False)
    X = np.hstack([Xp.to_numpy(float), Xd.to_numpy(float)])
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, d["epa_play"], sample_weight=d["plays"])
    npl = Xp.shape[1]
    pe = pd.Series(m.coef_[:npl], index=Xp.columns, name="eff")
    de = pd.Series(m.coef_[npl:], index=Xd.columns, name="def_eff")
    return pe, de


def run_all(alpha_by_pos):
    effs, defs = [], []
    for position in POS_ROLE:
        for season in SEASONS:
            d = wk[(wk["position"] == position) & (wk["season"] == season)]
            if len(d) < 100: continue
            pe, de = fit_season(d, position, alpha_by_pos[position])
            pl = d.groupby("player_id")["plays"].sum()
            e = pe.reset_index().rename(columns={"index": "player_id"})
            e["season"] = season; e["position"] = position
            e["plays"] = e["player_id"].map(pl)
            effs.append(e)
            dd = de.reset_index().rename(columns={"index": "defkey"})
            dd["season"] = season; dd["position"] = position
            defs.append(dd)
    return pd.concat(effs, ignore_index=True), pd.concat(defs, ignore_index=True)


# ---- choose alpha per position by out-of-time signal ----
print("=== Ridge alpha selection: corr(adjusted effect_t, raw skill z_{t+1}) ===")
nxt_raw = raw[["player_id", "season", "skill_z"]].copy()
nxt_raw["season"] -= 1
best_alpha = {}
for position in POS_ROLE:
    scores = {}
    for alpha in (4.0, 12.0, 40.0, 120.0):
        rows = []
        for season in SEASONS:
            d = wk[(wk["position"] == position) & (wk["season"] == season)]
            if len(d) < 100: continue
            pe, _ = fit_season(d, position, alpha)
            e = pe.reset_index().rename(columns={"index": "player_id"})
            e["season"] = season
            rows.append(e)
        e = pd.concat(rows).merge(nxt_raw, on=["player_id", "season"]).dropna()
        scores[alpha] = pearsonr(e["eff"], e["skill_z"])[0] if len(e) > 50 else np.nan
    best_alpha[position] = max(scores, key=lambda a: scores[a])
    print(f"  {position}: " + "  ".join(f"a={a}: r={scores[a]:.3f}" for a in scores)
          + f"  -> alpha {best_alpha[position]}")

eff, deff = run_all(best_alpha)

# era-normalize adjusted effects (plays-weighted z within position-season)
def wz(g):
    w = g["plays"]; m = np.average(g["eff"], weights=w)
    sd = np.sqrt(np.average((g["eff"] - m) ** 2, weights=w)) or 1.0
    return (g["eff"] - m) / sd
eff["elo_z"] = np.nan
for _, g in eff.groupby(["position", "season"]):
    eff.loc[g.index, "elo_z"] = wz(g)

# ---- YoY stickiness: adjusted vs raw ----
print("\n=== YoY stickiness (same qualifiers, min season plays) ===")
MINP = {"QB": 150, "RB": 80, "WR": 40, "TE": 40}
cmp_ = eff.merge(raw[["player_id", "season", "position", "skill_z", "plays"]]
                 .rename(columns={"plays": "plays_raw"}),
                 on=["player_id", "season", "position"])
cmp_ = cmp_[cmp_["plays_raw"] >= cmp_["position"].map(MINP)]
nx = cmp_[["player_id", "season", "elo_z", "skill_z"]].copy(); nx["season"] -= 1
pair = cmp_.merge(nx, on=["player_id", "season"], suffixes=("", "_n"))
for position in POS_ROLE:
    d = pair[pair["position"] == position]
    r_adj = pearsonr(d["elo_z"], d["elo_z_n"])[0]
    r_raw = pearsonr(d["skill_z"], d["skill_z_n"])[0]
    r_x = pearsonr(d["elo_z"], d["skill_z_n"])[0]   # adjusted -> next raw
    print(f"  {position}: adjusted {r_adj:.2f} vs raw {r_raw:.2f} | adj_t -> raw_t+1 {r_x:.2f} (n={len(d)})")

# ---- Kalman over adjusted series -> the Elo ----
print("\n=== Kalman over adjusted skill (one-step MAE vs repeat-last-adjusted) ===")
out = []
for position in POS_ROLE:
    d = eff[(eff["position"] == position)].rename(columns={"elo_z": "skill_composite"})
    d = d[d["plays"] >= MINP[position] * 0.6]
    medP = d["plays"].median()
    best = None
    for phi in (0.75, 0.85, 0.92, 0.97):
        for Q in (0.02, 0.05, 0.10, 0.20):
            for R0 in (0.4, 0.7, 1.0, 1.6, 2.5):
                mae, bmae = SK.one_step_score(d, phi, Q, R0, medP)
                if best is None or mae < best[0]: best = (mae, bmae, phi, Q, R0)
    mae, bmae, phi, Q, R0 = best
    print(f"  {position}: phi={phi} Q={Q} R0={R0} | {mae:.3f} vs {bmae:.3f} ({100*(bmae-mae)/bmae:+.0f}%)")
    for pid, g in d.groupby("player_id"):
        g = g.sort_values("season")
        y = g["skill_composite"].to_numpy(); yrs = g["season"].to_numpy()
        rv = R0 * medP / g["plays"].to_numpy()
        xf, xs, _ = SK.kalman_career(yrs, y, rv, phi, Q)
        for i in range(len(y)):
            out.append({"player_id": pid, "position": position, "season": int(yrs[i]),
                        "elo_obs": float(y[i]), "elo_true": float(xs[i]),
                        "elo_true_causal": float(xf[i]), "plays": int(g["plays"].iloc[i])})
res = pd.DataFrame(out)
res.to_sql("player_skill_elo", con, if_exists="replace", index=False)
print(f"\nSaved player_skill_elo: {len(res):,} rows")

# ---- head-to-head: Elo vs unadjusted pipeline predicting next RAW skill ----
print("\n=== Predicting next season's RAW skill z: Elo-causal vs unadjusted-causal ===")
h = res.merge(raw[["player_id", "season", "position", "skill_true_causal"]],
              on=["player_id", "season", "position"])
nx = raw[["player_id", "season", "position", "skill_z"]].copy(); nx["season"] -= 1
h = h.merge(nx, on=["player_id", "season", "position"]).dropna(subset=["skill_z"])
for position in POS_ROLE:
    d = h[h["position"] == position]
    r_elo = pearsonr(d["elo_true_causal"], d["skill_z"])[0]
    r_old = pearsonr(d["skill_true_causal"], d["skill_z"])[0]
    print(f"  {position}: Elo r={r_elo:.3f} vs unadjusted r={r_old:.3f} (n={len(d)})")

# ---- lockdown-corner test: WR1-specific defense effects ----
print("\n=== Lockdown test: defenses that suppress WR1s specifically (2016+) ===")
dw = deff[deff["position"] == "WR"].copy()
dw["team"] = dw["defkey"].str.split("@").str[0]
dw["rk"] = np.where(dw["defkey"].str.endswith("@1"), "wr1", "wr23")
piv = dw.pivot_table(index=["team", "season"], columns="rk", values="def_eff").dropna()
piv["wr1_extra_suppress"] = piv["wr1"] - piv["wr23"]
top = piv.nsmallest(8, "wr1_extra_suppress").reset_index()
for r in top.itertuples():
    print(f"  {r.season} {r.team}: WR1 effect {r.wr1:+.3f} vs WR2/3 {r.wr23:+.3f} EPA/target "
          f"(extra WR1 suppression {r.wr1_extra_suppress:+.3f})")

# ---- 2025 Elo leaders + CMC ----
print("\n=== 2025 Elo (causal, opponent+teammate adjusted) top 5 by position ===")
r25 = res[res["season"] == 2025].merge(names, on="player_id", how="left")
for position in POS_ROLE:
    top = r25[r25["position"] == position].nlargest(5, "elo_true_causal")
    print(f"  {position}: " + "; ".join(f"{t.name} {t.elo_true_causal:+.2f}" for t in top.itertuples()))
cmc = res.merge(names, on="player_id").query("name=='Christian McCaffrey'").sort_values("season")
print("\nMcCaffrey Elo by season (obs -> causal):")
print("  " + "  ".join(f"{r.season}:{r.elo_obs:+.2f}->{r.elo_true_causal:+.2f}" for r in cmc.itertuples()))
con.close()
