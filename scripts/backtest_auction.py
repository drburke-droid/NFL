"""
Backtest: what the risk-adjusted Draft Room would have suggested on AUCTION DAY for a
given season (default 2025) — using ONLY pre-season info (walk-forward) — and then how
each player ACTUALLY finished (above / to / below expectations).

Projections (leakage-free, train < YEAR): central P50, floor P15, bust prob. Risk-adjusted
certainty-equivalent + VONA + market compression = same engine as docs/index.html.
Verdict = actual season PPR total vs projected total (band +/-15%).
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
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
POS = ["QB", "RB", "WR", "TE"]; STARTABLE = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}
POS_INJ = {"QB": 0.26, "RB": 0.40, "WR": 0.33, "TE": 0.39}; K_RISK = 0.7
REPL_RANK = {"QB": 12, "RB": 29, "WR": 29, "TE": 13}        # marginal startable incl flex
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    nm = dict(con.execute("SELECT DISTINCT player_id, player_display_name FROM nflv_season").fetchall()); con.close()
    df = df[df.prior_games >= 3].copy()
    tr = df[(df.season < YEAR) & df.next_ppg.notna()]
    te = df[df.season == YEAR].copy()
    F = MS.FEATURES
    te["central"] = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB).fit(tr[F].astype(float).fillna(-1), tr.next_ppg).predict(te[F].astype(float).fillna(-1)).clip(0)
    te["floor"] = lgb.LGBMRegressor(objective="quantile", alpha=0.15, **GB).fit(tr[F].astype(float).fillna(-1), tr.next_ppg).predict(te[F].astype(float).fillna(-1)).clip(0)
    bz = {}
    for pos in POS:
        trp = tr[tr.position == pos]; tep = te[te.position == pos]
        if len(tep) == 0 or (trp.next_ppg < STARTABLE[pos]).sum() < 12: continue
        m = lgb.LGBMClassifier(objective="binary", **GB).fit(trp[F].astype(float).fillna(-1), (trp.next_ppg < STARTABLE[pos]).astype(int))
        for pid, b in zip(tep.player_id, m.predict_proba(tep[F].astype(float).fillna(-1))[:, 1]): bz[pid] = b
    te["bust"] = te.player_id.map(bz).fillna(0.30)
    te["proj_total"] = te.central * 16

    def risk(r):
        inj = max(0, ((POS_INJ.get(r.position, .33)) - .26) / .14)
        down = max(0, (r.central - r.floor) / r.central) if r.central > 0 else 0
        return min(.9, .45 * inj + .40 * r.bust + .30 * down)
    te["risk"] = te.apply(risk, axis=1)
    te["ra"] = te.proj_total * (1 - K_RISK * te.risk)

    # VONA auction value on risk-adjusted points
    repl = {}
    for pos in POS:
        s = te[te.position == pos].ra.sort_values(ascending=False).values
        repl[pos] = s[REPL_RANK[pos] - 1] if REPL_RANK[pos] - 1 < len(s) else (s[-1] if len(s) else 0)
    te["edge"] = (te.ra - te.position.map(repl)).clip(lower=0)
    disc = 12 * 200 - 12 * 16; per = disc / te.edge.sum()
    real = lambda v: v if v <= 25 else 25 + (v - 25) * 0.80
    te["bid"] = te.apply(lambda r: round(real(max(1, 1 + r.edge * per)) if r.edge > 0 else 1), axis=1)

    # actual vs expected
    te["actual"] = te.next_ppr_total
    te["ratio"] = te.actual / te.proj_total.clip(lower=1)
    def verdict(r):
        if pd.isna(r.actual): return "—"
        if r.ratio >= 1.15: return "ABOVE"
        if r.ratio >= 0.85: return "to exp"
        return "BELOW"
    te["verdict"] = te.apply(verdict, axis=1)

    top = te.sort_values("bid", ascending=False).head(20)
    print(f"=== {YEAR} auction-day top 20 (walk-forward, risk-adjusted) — then how they finished ===\n")
    print(f"  #  Bid  Pos  Player                Proj  Actual  PPG '{str(YEAR)[2:]}  Verdict")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        ppg = (r.actual / r.next_games) if (r.next_games and r.next_games > 0) else 0
        print(f"  {i:>2}  ${int(r.bid):>2}  {r.position:<3}  {str(nm.get(r.player_id,''))[:20]:<20} {int(r.proj_total):>4}  {('' if pd.isna(r.actual) else int(r.actual)):>5}   {ppg:>5.1f}   {r.verdict}")
    g = top.verdict.value_counts()
    print(f"\n  Of the top 20: {g.get('ABOVE',0)} ABOVE, {g.get('to exp',0)} to expectation, {g.get('BELOW',0)} BELOW")


if __name__ == "__main__":
    main()
