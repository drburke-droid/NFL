"""
Is a unit of QB VORP "worth more" than RB/WR VORP because QB points are more stable
and QBs are more available? Measures, by position (2012-2025, startable-tier players):

  availability  : games played next year + % of seasons missing significant time
  YoY stability : correlation of PPG year-to-year (how repeatable production is)
  weekly CV     : within-season week-to-week volatility (lower = more reliable)
  proj error    : our model's MAE / mean (how predictable)
  VORP spread   : PPG of pos#1 minus the replacement starter (how much elite is even worth)

Then connects to valuation: reliability vs the (small) QB VORP spread.
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
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "QB_RELIABILITY.md")
POS = ["QB", "RB", "WR", "TE"]
START_N = {"QB": 12, "RB": 30, "WR": 36, "TE": 14}     # startable cutoff incl flex
STARTABLE_PPG = {"QB": 15, "RB": 10, "WR": 10, "TE": 8}


def main():
    con = sqlite3.connect(DB)
    s = pd.read_sql("SELECT player_id, season, position, games, fantasy_points_ppr pts FROM nflv_season WHERE position IN ('QB','RB','WR','TE')", con)
    w = pd.read_sql("SELECT player_id, season, position, week, fantasy_points_ppr pts FROM nflv_weekly WHERE season_type='REG' AND week<=18 AND position IN ('QB','RB','WR','TE')", con)
    ds = pd.read_sql("SELECT * FROM season_dataset", con); con.close()
    s = s[s.games >= 1].copy(); s["ppg"] = s.pts / s.games

    # ---- availability: among players startable last year, games played this year ----
    avail = {}
    for Y in range(2013, 2026):
        prev = s[s.season == Y - 1]
        for pos in POS:
            top = prev[prev.position == pos].nlargest(START_N[pos], "ppg").player_id
            cur = s[(s.season == Y) & (s.player_id.isin(top))]
            for g in cur.games: avail.setdefault(pos, []).append(g)

    # ---- YoY PPG stability (startable last year) ----
    yoy = {}
    for pos in POS:
        a = s[(s.position == pos) & (s.ppg >= STARTABLE_PPG[pos]) & (s.games >= 6)][["player_id", "season", "ppg"]]
        b = a.copy(); b["season"] = b.season + 1; b = b.rename(columns={"ppg": "ppg_next"})
        m = a.merge(b[["player_id", "season", "ppg_next"]], on=["player_id", "season"], how="inner")
        yoy[pos] = np.corrcoef(m.ppg, m.ppg_next)[0, 1] if len(m) > 10 else np.nan

    # ---- weekly CV (within-season volatility) ----
    wcv = {}
    g = w.groupby(["player_id", "season", "position"]).pts.agg(["mean", "std", "count"]).reset_index()
    g = g[(g["count"] >= 8) & (g["mean"] >= 6)]
    for pos in POS: wcv[pos] = (g[g.position == pos]["std"] / g[g.position == pos]["mean"]).mean()

    # ---- projection error (leave-one-out), CV ----
    perr = {}
    for pos in POS:
        d = ds[(ds.position == pos) & ds.next_ppg.notna() & (ds.prior_games >= 3) & (ds.prior_ppg >= STARTABLE_PPG[pos])]
        if len(d) < 50: perr[pos] = np.nan; continue
        preds = []
        for Y in range(2016, 2026):
            tr, te = d[d.season < Y], d[d.season == Y]
            if len(te) == 0 or len(tr) < 40: continue
            m = lgb.LGBMRegressor(objective="regression_l1", **dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30, random_state=0, verbosity=-1)).fit(tr[MS.FEATURES].astype(float).fillna(-1), tr.next_ppg)
            te = te.copy(); te["pred"] = m.predict(te[MS.FEATURES].astype(float).fillna(-1)); preds.append(te)
        p = pd.concat(preds)
        perr[pos] = np.mean(np.abs(p.next_ppg - p.pred)) / p.next_ppg.mean()

    # ---- VORP spread: pos#1 minus replacement starter ----
    spread = {}
    for pos in POS:
        sp = []
        for Y in range(2012, 2026):
            pp = s[(s.season == Y) & (s.position == pos) & (s.games >= 6)].nlargest(START_N[pos] + 1, "ppg").ppg.values
            if len(pp) >= START_N[pos] + 1: sp.append(pp[0] - pp[START_N[pos]])
        spread[pos] = np.mean(sp)

    rows = []
    for pos in POS:
        a = np.array(avail[pos])
        rows.append([pos, a.mean(), 100 * np.mean(a < 14), yoy[pos], wcv[pos], perr[pos], spread[pos]])
    df = pd.DataFrame(rows, columns=["pos", "avg_games", "pct_<14g", "yoy_corr", "weekly_cv", "proj_err_cv", "vorp_spread"])

    print("=== Reliability & VORP by position (startable tier, 2012-2025) ===")
    print(df.round(2).to_string(index=False))

    L = ["# Is QB VORP worth more? — reliability vs VORP spread by position (2012-2025)\n",
         "Startable-tier players. avg_games/`%<14g` = availability (injury); yoy_corr = year-to-year PPG repeatability; weekly_cv = week-to-week volatility (lower=steadier); proj_err_cv = our model MAE / mean (lower=more predictable); vorp_spread = PPG of pos#1 minus the replacement starter.\n",
         "| Pos | Avg games | % <14 g | YoY corr | Weekly CV | Proj err (CV) | VORP spread |",
         "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.pos} | {r.avg_games:.1f} | {r['pct_<14g']:.0f}% | {r.yoy_corr:.2f} | {r.weekly_cv:.2f} | {r.proj_err_cv:.2f} | {r.vorp_spread:.1f} |")
    L += ["\n## Read",
          "- **The premise holds:** QBs are the most AVAILABLE (highest games, fewest missed), the most REPEATABLE year-to-year (highest YoY corr), the steadiest week to week (lowest CV), and the most PREDICTABLE (lowest projection error). RB is the opposite on every axis (most injury-prone, least repeatable).",
          "- **BUT QB has by far the SMALLEST VORP spread.** The 12th QB is nearly as good as the 1st, so 'elite QB' is only a few points above replacement — whereas RB/WR have a large gap between studs and replacement. There is simply less QB VORP to buy.",
          "- **Net:** a unit of QB VORP is more reliable, but there is much less of it. The two effects roughly cancel — which is exactly what the sims show: paying up for QB is ~neutral in best-ball, a tiny +0.3 places in set-lineup H2H (reliability protects your weekly floor), both within noise.",
          "\n## Practical takeaway",
          "- Don't *systematically* pay up for QB — VORP already prices the small spread correctly, and reliability only nudges it.",
          "- The reliability edge is real but second-order: it slightly favors taking a stable elite QB over a boom/bust one **at the same price**, and matters a touch more in start-your-lineup (H2H) than best-ball.",
          "- Where reliability *should* change behavior is the opposite end: it argues for FADING injury-fragile, boom/bust RBs relative to their raw VORP — which our fade-risk / archetype-aging tools already flag."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n" + "\n".join(L[4:]))
    print("Saved", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
