"""
Value-Leap score for 2026 — the season-level analog of the explosion model.

Flags CHEAP / undraftable-caliber players (prior PPG below a draftable line) most
likely to break out into a startable asset this year — i.e. the late-round darts
who get "drafted much higher" the following season. Validated walk-forward
(test_value_leap.py): AUC 0.70, precision@top10% 2.75x base.

Calibrated probability (Platt) so the surfaced % is realistic. Synthesizes the
niche signals we built: late-season surge (half-trend) + opportunity (vacated /
incoming touches) + base profile. Writes nflv_value_leap.
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


MS = _load("model_season"); BSD = _load("build_season_dataset")
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
# skill positions only — a cheap QB "leap" is usually just winning a starting job
# (QB is a streaming position), so those aren't actionable draft sleepers
POS = ["RB", "WR", "TE"]
LOW = {"RB": 7, "WR": 7, "TE": 5}
HI = {"RB": 11, "WR": 11, "TE": 8}
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]
OPP = ["vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
FEATS = MS.FEATURES + HT + OPP
GB = dict(n_estimators=400, learning_rate=0.03, num_leaves=24, min_child_samples=25,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def frame_2026(con):
    sf = BSD.build_season_frame(con)
    prior = sf[sf.season == 2025][BSD.FEAT_COLS].copy()
    prior.columns = ["player_id", "player_display_name", "position", "prior_season", "prior_team"] + ["prior_" + c for c in BSD.FEAT_COLS[5:]]
    prior["season"] = 2026
    prior2 = sf[sf.season == 2024][["player_id", "ppg", "games"]].rename(columns={"ppg": "prior2_ppg", "games": "prior2_games"})
    ctx = sf[sf.season == 2025][["player_id", "recent_team", "age", "years_exp", "height", "weight",
        "draft_round", "draft_pick", "forty", "vertical", "broad_jump", "cone", "shuttle"]].rename(columns={"recent_team": "team"})
    ctx["age"] += 1; ctx["years_exp"] += 1
    v = prior.merge(prior2, on="player_id", how="left").merge(ctx, on="player_id", how="left")
    v["team_change"] = 0
    ht = pd.read_sql("SELECT * FROM nflv_half_trend WHERE season=2026", con)
    opp = pd.read_sql("""SELECT player_id, season, vac_rb_carries, inc_rb_carries,
                         vac_pc_targets, inc_pc_targets, rook_rb, rook_wr
                         FROM nflv_opportunity WHERE season=2026""", con)
    v = v.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left")
    return v[v.prior_games.fillna(0) >= 3].copy()


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr FROM nflv_opportunity", con)
    df = df.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()
    df["leap"] = ((df.next_ppg >= df.position.map(HI)) & ((df.next_ppg - df.prior_ppg) >= 4)).astype(int)
    train = df[df.prior_ppg < df.position.map(LOW)].copy()

    v26 = frame_2026(con); con.close()
    out = []
    for pos in POS:
        tr = train[train.position == pos]
        te = v26[(v26.position == pos) & (v26.prior_ppg < LOW[pos])].copy()
        if len(te) == 0 or tr.leap.sum() < 10: continue
        cc = CalibratedClassifierCV(lgb.LGBMClassifier(objective="binary", **GB), method="sigmoid", cv=3)
        cc.fit(tr[FEATS].astype(float).fillna(-1), tr.leap)
        te["leap_prob"] = np.clip(cc.predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1], 0.01, 0.85).round(3)
        out.append(te[["player_id", "player_display_name", "position", "team", "prior_ppg", "leap_prob"]])
    res = pd.concat(out, ignore_index=True).sort_values("leap_prob", ascending=False)

    con = sqlite3.connect(DB); res.to_sql("nflv_value_leap", con, if_exists="replace", index=False); con.close()
    print(f"nflv_value_leap: scored {len(res)} cheap 2026 players (prior PPG below draftable line).")
    print("\nTop 20 value-leap candidates (cheap now, breakout upside):")
    for _, r in res.head(20).iterrows():
        print(f"  {r.leap_prob:.0%}  {str(r.player_display_name)[:22]:22s} {r.position} {r.team}  (2025 {r.prior_ppg:.1f} PPG)")


if __name__ == "__main__":
    main()
