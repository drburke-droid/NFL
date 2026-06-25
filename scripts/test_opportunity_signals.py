"""
Niche teammate-OPPORTUNITY signals: does the projection miss structural changes
in a player's situation — touches VACATED by departures, or STOLEN by arrivals?

The season model uses a player's OWN prior stats; it barely sees that the RB1
ahead of him left, or that a featured back just arrived. So a few pre-specified,
mechanism-driven situations may be systematically mis-projected.

All features are leakage-free: prior-year (Y-1) volume + who is on the Y roster
(known by draft day). Tested against the walk-forward projection RESIDUAL
(next_ppg - model P50); every claim must REPLICATE in 2013-2019 AND 2020-2025.
Pre-specified hypotheses only (round thresholds, no search).
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
POS = ["QB", "RB", "WR", "TE"]; BASE = MS.FEATURES
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
RNG = np.random.default_rng(0)
boot = lambda x: (np.nan, np.nan) if len(x) < 8 else tuple(np.percentile(
    [RNG.choice(x, len(x), True).mean() for _ in range(2000)], [2.5, 97.5]))


def opportunity_features(con):
    s = pd.read_sql("""SELECT player_id, season, recent_team team, position,
                       carries, targets FROM nflv_season WHERE position IN ('RB','WR','TE')""", con)
    s = s.groupby(["player_id", "season"], as_index=False).agg(
        team=("team", "last"), position=("position", "last"),
        carries=("carries", "sum"), targets=("targets", "sum"))
    vol = {(r.player_id, r.season): r for r in s.itertuples()}
    members = {}
    for r in s.itertuples(): members.setdefault((r.team, r.season), []).append(r.player_id)
    dr = pd.read_sql("SELECT season, team, position, pick FROM nflv_draft WHERE position IN ('RB','WR','TE')", con)
    rook = {}
    for r in dr.itertuples():
        if r.pick and r.pick <= 50: rook.setdefault((r.team, r.season), []).append(r.position)

    rows = []
    for (team, Y), ids in members.items():
        prev = set(members.get((team, Y-1), []))
        cur = set(ids)
        def vac(posset):  # touches that LEFT (on team Y-1, gone in Y)
            c = t = 0.0
            for pid in prev - cur:
                v = vol.get((pid, Y-1))
                if v and v.position in posset: c += v.carries or 0; t += v.targets or 0
            return c, t
        def inc(posset):  # touches that ARRIVED (new in Y, had volume in Y-1 elsewhere)
            c = t = 0.0
            for pid in cur - prev:
                v = vol.get((pid, Y-1))
                if v and v.position in posset: c += v.carries or 0; t += v.targets or 0
            return c, t
        vac_rb_c, _ = vac({"RB"}); _, vac_pc_t = vac({"WR", "TE"})
        inc_rb_c, _ = inc({"RB"}); _, inc_pc_t = inc({"WR", "TE"})
        rk = rook.get((team, Y), [])
        for pid in ids:  # returning players (on team in both Y-1 and Y)
            if pid not in prev: continue
            rows.append({"player_id": pid, "season": Y,
                         "vac_rb_carries": vac_rb_c, "inc_rb_carries": inc_rb_c,
                         "vac_pc_targets": vac_pc_t, "inc_pc_targets": inc_pc_t,
                         "rook_rb": rk.count("RB"), "rook_wr": rk.count("WR") + rk.count("TE")})
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    opp = opportunity_features(con); con.close()
    df = df.merge(opp, on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()

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
    P["resid"] = P.next_ppg - P.pred
    P = P[P.vac_rb_carries.notna()].copy()
    rb = P.position == "RB"; wrte = P.position.isin(["WR", "TE"]); wr = P.position == "WR"

    H = {
        "A. RB: RB1 departed (vac>=150), no big add": rb & (P.vac_rb_carries >= 150) & (P.inc_rb_carries < 100) & (P.rook_rb == 0),
        "B. WR/TE: targets vacated (>=100)":           wrte & (P.vac_pc_targets >= 100),
        "C. WR: new featured back arrived (incRB>=150)": wr & (P.inc_rb_carries >= 150),
        "D. RB: backfield got crowded (incRB>=150 or rookie RB)": rb & ((P.inc_rb_carries >= 150) | (P.rook_rb >= 1)),
        "E. WR/TE: new pass-catcher added (incTgt>=100)": wrte & (P.inc_pc_targets >= 100),
        "F. RB: rookie RB drafted top-50 ahead of him": rb & (P.rook_rb >= 1),
    }
    print(f"Residual base n={len(P)}, mean {P.resid.mean():+.2f} (≈0 expected)\n")
    print(f"{'hypothesis':54s} {'n':>4s} {'meanResid':>10s} {'95% CI':>15s}")
    print("-" * 90)
    surv = {}
    for name, mask in H.items():
        s = P[mask]; n = len(s)
        if n < 15: print(f"{name:54s} {n:>4d}   (too small)"); continue
        mr = s.resid.mean(); lo, hi = boot(s.resid.values)
        flag = "  <-- model too LOW" if lo > 0 else ("  <-- model too HIGH" if hi < 0 else "  (CI crosses 0)")
        print(f"{name:54s} {n:>4d} {mr:>+10.2f} [{lo:>+4.1f},{hi:>+4.1f}]{flag}")
        if lo > 0 or hi < 0: surv[name] = mask

    print("\n=== Continuous signal: corr(opportunity, residual) within position ===")
    for lab, mask, col in [("RB resid vs net RB carries (vac-inc)", rb, None),
                           ("WR/TE resid vs vacated targets", wrte, "vac_pc_targets"),
                           ("WR resid vs incoming RB carries", wr, "inc_rb_carries")]:
        s = P[mask].copy()
        x = (s.vac_rb_carries - s.inc_rb_carries) if col is None else s[col]
        print(f"  {lab:40s}: r = {np.corrcoef(x, s.resid)[0,1]:+.3f}  (n={len(s)})")

    print("\n=== Temporal replication (must hold in BOTH eras) ===")
    for name, mask in surv.items():
        a, b = P[mask & (P.season <= 2019)], P[mask & (P.season >= 2020)]
        def cell(s):
            if len(s) < 8: return f"n={len(s)} thin"
            lo, hi = boot(s.resid.values); return f"n={len(s)} {s.resid.mean():+.1f} [{lo:+.1f},{hi:+.1f}]"
        print(f"  {name:54s} | '13-19 {cell(a):28s} | '20-25 {cell(b)}")

    # face validity: name the strongest surviving rule's hits
    if surv:
        nm = pd.read_sql("SELECT DISTINCT player_id, player_display_name FROM nflv_season",
                         sqlite3.connect(DB)).set_index("player_id").player_display_name.to_dict()
        name0 = max(surv, key=lambda k: abs(P[surv[k]].resid.mean()))
        s = P[surv[name0]].assign(nm=lambda d: d.player_id.map(nm)).sort_values("resid", key=abs, ascending=False)
        print(f"\n=== Biggest hits for: {name0} ===")
        for _, r in s.head(12).iterrows():
            print(f"  {str(r.nm)[:22]:22s} {r.position} {int(r.season)}  proj {r.pred:.1f} -> {r.next_ppg:.1f}  ({r.resid:+.1f})")


if __name__ == "__main__":
    main()
