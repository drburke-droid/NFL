"""Skill = efficiency + earning volume: blend vol_z into the Elo before ranking.

Why: the Elo (player_skill_elo) is per-play efficiency, opponent-adjusted. It
carries ZERO credit for commanding targets — so 2025 Ja'Marr Chase (185
targets, league-average EPA/target with Browning/Flacco throwing) ranks WR25
behind 50-target boutique deep threats. Earning volume IS receiver skill
(getting open, coverage gravity), and within-player weekly usage barely drags
efficiency (r=-0.04), so volume isn't just diluted efficiency.

Method: Kalman-smooth vol_z (plays/game, era-z) with the same machinery as the
Elo, then blend causal estimates: alpha_skill = w*elo + (1-w)*vol. Choose w
per position OUT-OF-TIME by partial corr with next-season PPG controlling
prior PPG + age (the criterion skill_forward_value.py uses). Report where the
blend puts 2025 WRs. Writes table player_skill_alpha.
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("sk", os.path.join(HERE, "skill_kalman.py"))
SK = importlib.util.module_from_spec(_s)
_s.loader.exec_module(SK)
DB = os.path.join(os.path.dirname(HERE), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]
MINP = {"QB": 150, "RB": 80, "WR": 40, "TE": 40}

con = sqlite3.connect(DB)
elo = pd.read_sql("""SELECT player_id, position, season, elo_true, elo_true_causal, plays
                     FROM player_skill_elo""", con)
ps = pd.read_sql("""SELECT player_id, position, season, plays, vol_z
                    FROM player_skill_seasons WHERE vol_z IS NOT NULL""", con)
sd = pd.read_sql("SELECT player_id, season, position, age, prior_ppg, next_ppg FROM season_dataset", con)
names = pd.read_sql("SELECT DISTINCT player_id, player_display_name AS name FROM nflv_traj", con)

# ---- Kalman over vol_z careers (same grid search, one-step validated) ----
print("=== Kalman over volume z (one-step MAE vs repeat-last) ===")
vout = []
for pos in POS:
    d = ps[(ps["position"] == pos) & (ps["plays"] >= MINP[pos] * 0.6)] \
        .rename(columns={"vol_z": "skill_composite"})
    medP = d["plays"].median()
    best = None
    for phi in (0.75, 0.85, 0.92, 0.97):
        for Q in (0.02, 0.05, 0.10, 0.20):
            for R0 in (0.4, 0.7, 1.0, 1.6, 2.5):
                mae, bmae = SK.one_step_score(d, phi, Q, R0, medP)
                if best is None or mae < best[0]: best = (mae, bmae, phi, Q, R0)
    mae, bmae, phi, Q, R0 = best
    print(f"  {pos}: phi={phi} Q={Q} R0={R0} | {mae:.3f} vs {bmae:.3f} ({100*(bmae-mae)/bmae:+.0f}%)")
    for pid, g in d.groupby("player_id"):
        g = g.sort_values("season")
        y = g["skill_composite"].to_numpy(); yrs = g["season"].to_numpy()
        rv = R0 * medP / g["plays"].to_numpy()
        xf, xs, _ = SK.kalman_career(yrs, y, rv, phi, Q)
        for i in range(len(y)):
            vout.append({"player_id": pid, "position": pos, "season": int(yrs[i]),
                         "vol_true_causal": float(xf[i]), "vol_true": float(xs[i])})
vol = pd.DataFrame(vout)

# ---- join causal elo + causal vol + next-season outcomes ----
d = elo.merge(vol, on=["player_id", "position", "season"], how="inner")
out = sd.rename(columns={"season": "tgt"}); out["season"] = out["tgt"] - 1
d = d.merge(out[["player_id", "season", "age", "prior_ppg", "next_ppg"]],
            on=["player_id", "season"], how="left")
d["age"] = d["age"].fillna(d["age"].median())


def partial(dd, xcol):
    dd = dd.dropna(subset=[xcol, "next_ppg", "prior_ppg", "age"])
    if len(dd) < 60: return np.nan, len(dd)
    A = np.column_stack([np.ones(len(dd)), dd["prior_ppg"], dd["age"]])
    rx = dd[xcol] - A @ np.linalg.lstsq(A, dd[xcol], rcond=None)[0]
    ry = dd["next_ppg"] - A @ np.linalg.lstsq(A, dd["next_ppg"], rcond=None)[0]
    return pearsonr(rx, ry)[0], len(dd)


print("\n=== Blend weight w: partial corr(w*elo + (1-w)*vol, next PPG | prior PPG, age) ===")
WGRID = (1.0, 0.85, 0.7, 0.5, 0.3, 0.0)
best_w = {}
for pos in POS:
    dd = d[d["position"] == pos].copy()
    row = {}
    for w in WGRID:
        dd["blend"] = w * dd["elo_true_causal"] + (1 - w) * dd["vol_true_causal"]
        row[w], n = partial(dd, "blend")
    best_w[pos] = max(row, key=lambda k: row[k])
    print(f"  {pos}: " + "  ".join(f"w={w}: r={row[w]:+.3f}" for w in WGRID)
          + f"  -> w={best_w[pos]} (n={n})")

wcol = d["position"].map(best_w)
d["alpha_skill"] = wcol * d["elo_true_causal"] + (1 - wcol) * d["vol_true_causal"]
# smoothed (RTS) blend for display curves — same weights, chosen on the causal side
d["alpha_true"] = wcol * d["elo_true"] + (1 - wcol) * d["vol_true"]
d[["player_id", "position", "season", "elo_true_causal", "vol_true_causal",
   "alpha_skill", "alpha_true", "plays"]].to_sql("player_skill_alpha", con, if_exists="replace", index=False)
print(f"\nSaved player_skill_alpha: {len(d):,} rows")

# ---- 2025 WR list under the blend ----
print("\n=== 2025 WR alpha-skill top 20 (w_eff=%.2f) vs pure-Elo rank ===" % best_w["WR"])
r25 = d[(d["season"] == 2025) & (d["position"] == "WR")].merge(names, on="player_id", how="left")
r25["rk_elo"] = r25["elo_true_causal"].rank(ascending=False)
for i, r in enumerate(r25.nlargest(20, "alpha_skill").itertuples(), 1):
    print(f"  {i:2d}. {str(r.name):24s} {r.alpha_skill:+.2f}  "
          f"(eff {r.elo_true_causal:+.2f}, vol {r.vol_true_causal:+.2f}, Elo rank {r.rk_elo:.0f})")
ch = d.merge(names, on="player_id").query("name==\"Ja'Marr Chase\"").sort_values("season")
print("\nChase: season, eff-causal, vol-causal, alpha:")
for r in ch.itertuples():
    print(f"  {r.season}: eff {r.elo_true_causal:+.2f}  vol {r.vol_true_causal:+.2f}  alpha {r.alpha_skill:+.2f}")
con.close()
