"""
Backtest the auction projection for every year (2014-2025), grade each player
above/to/below expectation, then ask: can ANY combination of draft-day features
predict the verdict? (If yes, our projection is leaving signal on the table; if no,
the residuals are noise = the projection is efficient.)

For each year: walk-forward central/floor/bust (train < Y) -> proj_total = central*16,
verdict = actual PPR total vs proj (+/-15%). Attach EVERY feature we have (base +
half-trend + opportunity + comps + injury-prior + risk). Then walk-forward predict the
verdict and report skill vs baseline + which features matter and their direction.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]; STARTABLE = {"QB": 14, "RB": 10, "WR": 9, "TE": 7}
RELEV = {"QB": 13, "RB": 8, "WR": 8, "TE": 6}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr FROM nflv_opportunity", con)
    cf = pd.read_sql("SELECT * FROM nflv_comp_features", con)
    ns = pd.read_sql("SELECT player_id, season, games FROM nflv_season", con).drop_duplicates(["player_id", "season"])
    con.close()
    g2 = ns.rename(columns={"games": "prior2_games"}); g2["season"] = g2.season + 2
    df = df.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left").merge(cf, on=["player_id", "season"], how="left").merge(g2, on=["player_id", "season"], how="left")
    df = MS.add_injury_features(df)
    df = df[df.prior_games >= 3].copy()
    F = MS.FEATURES

    parts = []
    for Y in range(2014, 2026):
        tr = df[(df.season < Y) & df.next_ppg.notna()]; te = df[df.season == Y].copy()
        if len(tr) < 200 or len(te) == 0: continue
        te["central"] = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **GB).fit(tr[F].astype(float).fillna(-1), tr.next_ppg).predict(te[F].astype(float).fillna(-1)).clip(0)
        te["floorq"] = lgb.LGBMRegressor(objective="quantile", alpha=0.15, **GB).fit(tr[F].astype(float).fillna(-1), tr.next_ppg).predict(te[F].astype(float).fillna(-1)).clip(0)
        bz = {}
        for pos in POS:
            trp, tep = tr[tr.position == pos], te[te.position == pos]
            if len(tep) and (trp.next_ppg < STARTABLE[pos]).sum() >= 12:
                m = lgb.LGBMClassifier(objective="binary", **GB).fit(trp[F].astype(float).fillna(-1), (trp.next_ppg < STARTABLE[pos]).astype(int))
                bz.update(dict(zip(tep.player_id, m.predict_proba(tep[F].astype(float).fillna(-1))[:, 1])))
        te["bustp"] = te.player_id.map(bz).fillna(0.30)
        parts.append(te)
    P = pd.concat(parts, ignore_index=True)
    P = P[(P.next_ppr_total.notna()) & (P.central >= P.position.map(RELEV))].copy()   # fantasy-relevant
    P["proj_total"] = P.central * 16
    P["ratio"] = P.next_ppr_total / P.proj_total.clip(lower=1)
    P["verdict"] = np.where(P.ratio >= 1.15, "ABOVE", np.where(P.ratio < 0.85, "BELOW", "to_exp"))
    P["y"] = P.verdict.map({"BELOW": 0, "to_exp": 1, "ABOVE": 2})
    print(f"Relevant player-seasons 2014-2025: {len(P)}")
    print("Verdict mix:", {k: f"{v/len(P):.0%}" for k, v in P.verdict.value_counts().items()})

    # candidate features (all draft-day, leakage-free) + the projection's own pieces
    CAND = [c for c in (F + MS.INJURY_FEATURES + ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope",
            "vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr",
            "comp_prime", "comp_bust", "comp_elite", "comp_dist", "central", "floorq", "bustp"]) if c in P.columns]
    CAND = list(dict.fromkeys(CAND))

    # ---- walk-forward: can we predict the verdict above baseline? ----
    preds = []
    for T in range(2018, 2026):
        tr, te = P[P.season < T], P[P.season == T]
        if len(te) == 0 or len(tr) < 300: continue
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, **GB).fit(tr[CAND].astype(float).fillna(-1), tr.y)
        pr = m.predict_proba(te[CAND].astype(float).fillna(-1))
        t2 = te.copy(); t2["pred"] = pr.argmax(1)
        for j in range(3): t2[f"p{j}"] = pr[:, j]
        preds.append(t2)
    R = pd.concat(preds, ignore_index=True)
    base = R.y.value_counts(normalize=True).max()
    print(f"\n=== Predicting the verdict (walk-forward 2018-2025, n={len(R)}) ===")
    print(f"  accuracy {accuracy_score(R.y, R.pred):.3f}  vs majority-class baseline {base:.3f}")
    for j, lab in enumerate(["BELOW", "to_exp", "ABOVE"]):
        try: print(f"  AUC {lab:6s} (one-vs-rest): {roc_auc_score((R.y==j).astype(int), R[f'p{j}']):.3f}")
        except Exception: pass

    # ---- does the RISK lever surface 'confident ABOVE', or just safe 'to_exp'? ----
    POS_INJ = {"QB": 0.26, "RB": 0.40, "WR": 0.33, "TE": 0.39}
    inj = P.position.map(POS_INJ).fillna(.33).apply(lambda x: max(0, (x - .26) / .14))
    down = ((P.central - P.floorq) / P.central.clip(lower=1)).clip(lower=0)
    P["risk"] = (0.45 * inj + 0.40 * P.bustp + 0.30 * down).clip(0, .9)
    print("\n=== Verdict mix by RISK level (does low risk = more ABOVE, or just fewer BELOW?) ===")
    P["rt"] = pd.qcut(P.risk.rank(method="first"), 3, labels=["low-risk", "mid", "high-risk"])
    for t in ["low-risk", "mid", "high-risk"]:
        s = P[P.rt == t]; vc = s.verdict.value_counts(normalize=True)
        print(f"  {t:10s} (n={len(s)}):  ABOVE {vc.get('ABOVE',0):.0%}  to_exp {vc.get('to_exp',0):.0%}  BELOW {vc.get('BELOW',0):.0%}  | mean ratio {s.ratio.mean():.2f}")
    # practical: top-30/yr by RAW proj vs by RISK-ADJUSTED value
    print("\n=== Top-30/yr draftable set: ranked by RAW projection vs RISK-ADJUSTED ===")
    for K, lab in [(0.0, "raw proj   "), (0.7, "risk-adj .7"), (1.2, "risk-adj 1.2")]:
        P["score"] = P.central * (1 - K * P.risk)
        sel = P.groupby("season", group_keys=False).apply(lambda g: g.nlargest(30, "score"))
        vc = sel.verdict.value_counts(normalize=True)
        print(f"  {lab}:  ABOVE {vc.get('ABOVE',0):.0%}  to_exp {vc.get('to_exp',0):.0%}  BELOW {vc.get('BELOW',0):.0%}  | mean ratio {sel.ratio.mean():.2f}")

    # ---- which features, and direction (mean ratio by tercile) ----
    full = lgb.LGBMClassifier(objective="multiclass", num_class=3, **GB).fit(P[CAND].astype(float).fillna(-1), P.y)
    imp = pd.Series(full.feature_importances_, index=CAND).sort_values(ascending=False)
    print("\n=== Top features (gain) + direction: mean actual/proj ratio by low/mid/high tercile ===")
    for f in imp.head(12).index:
        s = P[[f, "ratio"]].dropna()
        if s[f].nunique() < 3:
            print(f"  {f:18s} (binary/const)"); continue
        try:
            s["t"] = pd.qcut(s[f].rank(method="first"), 3, labels=["lo", "mid", "hi"])
            mr = s.groupby("t", observed=True).ratio.mean()
            print(f"  {f:18s} ratio  lo {mr['lo']:.2f} | mid {mr['mid']:.2f} | hi {mr['hi']:.2f}")
        except Exception: print(f"  {f:18s} (skip)")


if __name__ == "__main__":
    main()
