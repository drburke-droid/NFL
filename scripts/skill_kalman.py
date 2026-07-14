"""Latent 'true skill' via Kalman filter + RTS smoother over career skill curves.

Model per player career (position-specific parameters):
    x_{t+1} = phi * x_t + w,  w ~ N(0, Q)          latent skill, mean-reverting to 0
    y_t     = x_t + e,        e ~ N(0, R0 * medP/plays_t)   observed season z

phi (persistence), Q (skill drift), R0 (measurement noise for a median-plays
season) are grid-searched per position to maximize one-step-ahead predictive
accuracy of the NEXT observed season — so the smoothing is validated against
held-out-in-time data, not chosen for looks. Missing seasons propagate the
state (variance grows). RTS smoother gives the display curve; the filtered
(causal) estimate is also kept — it is the honest "current skill" number.

Writes table player_skill_true and prints validation (raw vs smoothed one-step
prediction of next observed z).
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]


def kalman_career(years, y, rvar, phi, Q):
    """Filter + RTS smooth one career. years must be sorted; gaps allowed.
    Returns (filtered_mean, smoothed_mean) arrays aligned to observations."""
    n = len(y)
    xf = np.zeros(n); Pf = np.zeros(n)          # filtered
    xp = np.zeros(n); Pp = np.zeros(n)          # one-step predictions
    x, P = 0.0, 1.0                             # prior: position mean, unit var
    prev_year = None
    for i in range(n):
        gap = 1 if prev_year is None else years[i] - prev_year
        for _ in range(max(1, gap) if prev_year is not None else 1):
            x = phi * x
            P = phi * phi * P + Q
            if prev_year is None:
                x, P = 0.0, 1.0                  # first obs: stationary prior
                break
        xp[i], Pp[i] = x, P
        K = P / (P + rvar[i])
        x = x + K * (y[i] - x)
        P = (1 - K) * P
        xf[i], Pf[i] = x, P
        prev_year = years[i]
    # RTS smoother
    xs = xf.copy(); Ps = Pf.copy()
    for i in range(n - 2, -1, -1):
        gap = years[i + 1] - years[i]
        phig = phi ** gap
        Ppred = phig * phig * Pf[i] + Q * sum(phi ** (2 * k) for k in range(gap))
        C = Pf[i] * phig / Ppred
        xs[i] = xf[i] + C * (xs[i + 1] - phig * xf[i])
        Ps[i] = Pf[i] + C * C * (Ps[i + 1] - Ppred)
    return xf, xs, xp


def one_step_score(d, phi, Q, R0, medP):
    """Mean |error| of predicting each season's observed z from prior seasons."""
    errs, base = [], []
    for _, g in d.groupby("player_id"):
        g = g.sort_values("season")
        y = g["skill_composite"].to_numpy()
        yrs = g["season"].to_numpy()
        rv = R0 * medP / g["plays"].to_numpy()
        _, _, xp = kalman_career(yrs, y, rv, phi, Q)
        if len(y) >= 2:
            errs.extend(np.abs(xp[1:] - y[1:]))
            base.extend(np.abs(y[:-1] - y[1:]))   # naive: repeat last observed
    return np.mean(errs), np.mean(base)


def main():
    con = sqlite3.connect(DB)
    ps = pd.read_sql("""SELECT player_id, position, season, plays, skill_composite
                        FROM player_skill_seasons
                        WHERE skill_composite IS NOT NULL""", con)
    out = []
    print("=== Kalman 'true skill': grid-searched per position ===")
    print(f"{'pos':4s} {'phi':>5s} {'Q':>5s} {'R0':>5s} | one-step MAE model vs repeat-last | n careers")
    for pos in POS:
        d = ps[ps["position"] == pos]
        medP = d["plays"].median()
        best = None
        for phi in (0.75, 0.85, 0.92, 0.97):
            for Q in (0.02, 0.05, 0.10, 0.20):
                for R0 in (0.4, 0.7, 1.0, 1.6, 2.5):
                    mae, bmae = one_step_score(d, phi, Q, R0, medP)
                    if best is None or mae < best[0]:
                        best = (mae, bmae, phi, Q, R0)
        mae, bmae, phi, Q, R0 = best
        print(f"{pos:4s} {phi:5.2f} {Q:5.2f} {R0:5.2f} | {mae:.3f} vs {bmae:.3f} "
              f"({100*(bmae-mae)/bmae:+.0f}%) | {d['player_id'].nunique()}")
        for pid, g in d.groupby("player_id"):
            g = g.sort_values("season")
            y = g["skill_composite"].to_numpy()
            yrs = g["season"].to_numpy()
            rv = R0 * medP / g["plays"].to_numpy()
            xf, xs, _ = kalman_career(yrs, y, rv, phi, Q)
            for i in range(len(y)):
                out.append({"player_id": pid, "position": pos, "season": int(yrs[i]),
                            "skill_obs": float(y[i]), "skill_true": float(xs[i]),
                            "skill_true_causal": float(xf[i]), "plays": int(g["plays"].iloc[i])})
    res = pd.DataFrame(out)
    res.to_sql("player_skill_true", con, if_exists="replace", index=False)
    print(f"\nSaved player_skill_true: {len(res):,} rows.")

    # McCaffrey check
    cmc = pd.read_sql("""SELECT t.season, t.skill_obs, t.skill_true, t.skill_true_causal, t.plays
                         FROM player_skill_true t
                         JOIN (SELECT DISTINCT player_id, player_display_name FROM nflv_traj) n
                           USING(player_id)
                         WHERE n.player_display_name='Christian McCaffrey'
                         ORDER BY season""", con)
    print("\nMcCaffrey observed vs true (smoothed | causal):")
    for r in cmc.itertuples():
        print(f"  {r.season}: obs {r.skill_obs:+.2f} -> true {r.skill_true:+.2f} "
              f"(causal {r.skill_true_causal:+.2f}) on {r.plays} plays")
    con.close()


if __name__ == "__main__":
    main()
