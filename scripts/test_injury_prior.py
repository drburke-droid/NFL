"""
Injury/games-aware prior: does the projection under-rate players coming off an
injury-shortened season (e.g. Burrow), and can a games-weighted "healthy" prior
fix it WITHOUT hurting everyone else?

The model over-anchors on prior_ppg even when that season was a tiny sample. New
leakage-free features (all from prior seasons):
  prior2_games   games in year-2
  gw_prior       games-weighted avg of last 2 seasons' PPG (down-weights a short year)
  healthy_prior  max(prior_ppg, prior2_ppg)  -- a 'when healthy' proxy
  short_season   prior_games <= 11
  bounce         short recent year AFTER a healthy productive year
  games_trend    prior_games - prior2_games

Walk-forward BASE vs BASE+INJ: overall MAE/AUC, and specifically the injury-return
subgroup (where BASE is expected to under-project). Ships only if it helps the
subgroup without hurting the pool.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]; BASE = MS.FEATURES
INJ = ["prior2_games", "gw_prior", "healthy_prior", "short_season", "bounce", "games_trend"]
BUST = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}; ELITE = {"QB": 21, "RB": 16, "WR": 15, "TE": 12}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ns = pd.read_sql("SELECT player_id, season, games FROM nflv_season", con).drop_duplicates(["player_id", "season"])
    con.close()
    # prior2_games = games two seasons before the target
    g2 = ns.copy(); g2["season"] = g2.season + 2
    df = df.merge(g2.rename(columns={"games": "prior2_games"}), on=["player_id", "season"], how="left")
    df = df[df.next_ppg.notna() & (df.prior_games >= 3)].copy()

    pg, p2g = df.prior_games.fillna(0), df.prior2_games.fillna(0)
    pp, p2p = df.prior_ppg, df.prior2_ppg
    df["gw_prior"] = ((pp.fillna(0)*pg + p2p.fillna(0)*p2g) / (pg + p2g).replace(0, np.nan)).fillna(pp)
    df["healthy_prior"] = pd.concat([pp, p2p], axis=1).max(axis=1)
    df["short_season"] = (pg <= 11).astype(int)
    df["bounce"] = ((pg <= 11) & (p2g >= 14) & ((p2p - pp) >= 3)).astype(int)
    df["games_trend"] = pg - p2g

    # injury-return subgroup: short recent year after a healthy, productive one
    FLOOR = {"QB": 14, "RB": 9, "WR": 9, "TE": 6}
    df["inj_return"] = ((pg <= 11) & (p2g >= 14) &
                        (p2p >= df.position.map(FLOOR)) & (p2p > pp)).astype(int)
    print(f"Rows {len(df):,} | prior2_games coverage {df.prior2_games.notna().mean():.0%} | injury-return cases {int(df.inj_return.sum())}\n")

    def wf(feats, target=None, line=None):
        rows = []
        for pos in POS:
            d = df[df.position == pos].copy()
            if target: d["y"] = (d.next_ppg < line[pos]).astype(int) if target == "bust" else (d.next_ppg >= line[pos]).astype(int)
            for T in range(2017, 2026):
                tr, te = d[d.season < T], d[d.season == T]
                if len(te) == 0 or len(tr) < 80: continue
                if target:
                    if tr.y.sum() < 12: continue
                    m = lgb.LGBMClassifier(objective="binary", **GB); m.fit(tr[feats].astype(float).fillna(-1), tr.y)
                    te = te.copy(); te["p"] = m.predict_proba(te[feats].astype(float).fillna(-1))[:, 1]
                else:
                    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB); m.fit(tr[feats].astype(float).fillna(-1), tr.next_ppg)
                    te = te.copy(); te["pred"] = m.predict(te[feats].astype(float).fillna(-1))
                rows.append(te)
        return pd.concat(rows, ignore_index=True)

    b, h = wf(BASE), wf(BASE + INJ)
    print("=== Overall central MAE (must NOT get worse) ===")
    print(f"{'pos':5s} {'BASE':>7s} {'+INJ':>7s} {'Δ':>7s}")
    for pos in POS + ["ALL"]:
        bb, hh = (b, h) if pos == "ALL" else (b[b.position == pos], h[h.position == pos])
        mb, mh = mean_absolute_error(bb.next_ppg, bb.pred), mean_absolute_error(hh.next_ppg, hh.pred)
        print(f"{pos:5s} {mb:7.3f} {mh:7.3f} {mh-mb:+7.3f}")

    print("\n=== INJURY-RETURN subgroup (the players we're trying to fix) ===")
    bi, hi = b[b.inj_return == 1], h[h.inj_return == 1]
    print(f"  n = {len(bi)}")
    print(f"  BASE : mean proj {bi.pred.mean():.1f}  vs actual {bi.next_ppg.mean():.1f}  (bias {bi.next_ppg.mean()-bi.pred.mean():+.2f}, MAE {mean_absolute_error(bi.next_ppg,bi.pred):.2f})")
    print(f"  +INJ : mean proj {hi.pred.mean():.1f}  vs actual {hi.next_ppg.mean():.1f}  (bias {hi.next_ppg.mean()-hi.pred.mean():+.2f}, MAE {mean_absolute_error(hi.next_ppg,hi.pred):.2f})")

    print("\n=== Boom/Bust AUC (overall) ===")
    for tgt, line in [("bust", BUST), ("boom", ELITE)]:
        bb, hh = wf(BASE, tgt, line), wf(BASE + INJ, tgt, line)
        yb = (bb.next_ppg < bb.position.map(line)).astype(int) if tgt == "bust" else (bb.next_ppg >= bb.position.map(line)).astype(int)
        yh = (hh.next_ppg < hh.position.map(line)).astype(int) if tgt == "bust" else (hh.next_ppg >= hh.position.map(line)).astype(int)
        print(f"  {tgt:5s} BASE {roc_auc_score(yb,bb.p):.3f} -> +INJ {roc_auc_score(yh,hh.p):.3f} ({roc_auc_score(yh,hh.p)-roc_auc_score(yb,bb.p):+.3f})")

    print("\n=== Biggest under-projections BASE fixes on (injury-return hits) ===")
    m = bi.merge(hi[["player_id", "season", "pred"]], on=["player_id", "season"], suffixes=("_base", "_inj"))
    nm = pd.read_sql("SELECT DISTINCT player_id, player_display_name FROM nflv_season", sqlite3.connect(DB)).set_index("player_id").player_display_name.to_dict()
    m["gain"] = m.pred_inj - m.pred_base
    for _, r in m.sort_values("gain", ascending=False).head(10).iterrows():
        print(f"  {str(nm.get(r.player_id))[:20]:20s} {r.position} {int(r.season)}: BASE {r.pred_base:.1f} -> +INJ {r.pred_inj:.1f} (actual {r.next_ppg:.1f})")


if __name__ == "__main__":
    main()
