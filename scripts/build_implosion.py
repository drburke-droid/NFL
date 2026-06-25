"""
Implosion / Fade-risk score for 2026 — the mirror of value-leap.

Flags players who were drafted HIGH (startable studs last year) most likely to
collapse below startable this season — the ones who get "drafted much later" the
following year. Validated walk-forward (test_implosion.py): AUC 0.67,
precision@top10% 2.24x base (~1 in 3 of the top decile implode).

Calibrated probability (Platt). Synthesizes late-season fade (half-trend) +
incoming competition (opportunity) + one-year-wonder / volatility signals. Writes
nflv_implosion. Complements the calibrated bust% (which lacks the fade/competition
signals) by targeting the expensive-bust event specifically.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


MS = _load("model_season"); BVL = _load("build_value_leap")
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]
HIGH = {"QB": 16, "RB": 12, "WR": 12, "TE": 8}
BUST = {"QB": 14, "RB": 9, "WR": 9, "TE": 7}
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]
OPP = ["vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
FEATS = MS.FEATURES + HT + OPP
GB = dict(n_estimators=400, learning_rate=0.03, num_leaves=24, min_child_samples=25,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr FROM nflv_opportunity", con)
    df = df.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    df["collapse"] = ((df.next_ppg < df.position.map(BUST)) & ((df.prior_ppg - df.next_ppg) >= 4)).astype(int)
    train = df[df.prior_ppg >= df.position.map(HIGH)].copy()

    v26 = BVL.frame_2026(con); con.close()
    out = []
    for pos in POS:
        tr = train[train.position == pos]
        te = v26[(v26.position == pos) & (v26.prior_ppg >= HIGH[pos])].copy()
        if len(te) == 0 or tr.collapse.sum() < 10: continue
        cc = CalibratedClassifierCV(lgb.LGBMClassifier(objective="binary", **GB), method="sigmoid", cv=3)
        cc.fit(tr[FEATS].astype(float).fillna(-1), tr.collapse)
        te["fade_prob"] = np.clip(cc.predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1], 0.02, 0.90).round(3)
        out.append(te[["player_id", "player_display_name", "position", "team", "prior_ppg", "fade_prob"]])
    res = pd.concat(out, ignore_index=True).sort_values("fade_prob", ascending=False)

    con = sqlite3.connect(DB); res.to_sql("nflv_implosion", con, if_exists="replace", index=False); con.close()
    print(f"nflv_implosion: scored {len(res)} high-drafted 2026 players.")
    print("\nHighest fade risk (drafted high, implosion upside):")
    for _, r in res.head(20).iterrows():
        print(f"  {r.fade_prob:.0%}  {str(r.player_display_name)[:22]:22s} {r.position} {r.team}  (2025 {r.prior_ppg:.1f} PPG)")


if __name__ == "__main__":
    main()
