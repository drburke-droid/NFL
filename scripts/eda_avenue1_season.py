"""
Avenue 1 EDA: what historical/preseason signals explain FULL-SEASON fantasy.

Targets: next_ppg (per-game PPR) and next_pos_finish (positional rank).
Framing:
  1. How sticky is production year over year (prior_ppg -> next_ppg)?
  2. What correlates with next_ppg, overall + by position?
  3. RESIDUAL analysis: what explains next_ppg BEYOND a naive "repeat last year"
     baseline (the actionable edge)?
  4. Mechanisms: age curves, draft capital, role stickiness, TD regression.
  5. Temporally-validated model: incremental lift of full features over prior_ppg.

Outputs CSVs + PNG charts to outputs/eda/, prints a summary.
"""
import os, sqlite3, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "eda")
os.makedirs(OUT, exist_ok=True)
POS = ["QB", "RB", "WR", "TE"]

FEATURES = [
    "prior_ppg","prior2_ppg","prior_games","prior_ppr_std","prior_cv","prior_off_pct",
    "prior_target_share","prior_air_yards_share","prior_wopr","prior_total_tds",
    "prior_rec_td_rate","prior_rush_td_rate","prior_passing_epa","prior_rushing_epa",
    "prior_receiving_epa","prior_targets_pg","prior_receptions_pg","prior_receiving_yards_pg",
    "prior_carries_pg","prior_rushing_yards_pg","prior_attempts_pg","prior_passing_yards_pg",
    "age","years_exp","draft_round","draft_pick","weight","forty","team_change",
]
LABELS = {
    "prior_ppg":"Prior PPG","prior2_ppg":"PPG 2yrs ago","prior_games":"Prior games played",
    "prior_ppr_std":"Prior weekly std","prior_cv":"Prior coeff. of variation",
    "prior_off_pct":"Prior snap %","prior_target_share":"Prior target share",
    "prior_air_yards_share":"Prior air-yards share","prior_wopr":"Prior WOPR",
    "prior_total_tds":"Prior total TDs","prior_rec_td_rate":"Prior rec TD/yd",
    "prior_rush_td_rate":"Prior rush TD/yd","prior_passing_epa":"Prior pass EPA",
    "prior_rushing_epa":"Prior rush EPA","prior_receiving_epa":"Prior rec EPA",
    "prior_targets_pg":"Prior targets/g","prior_receptions_pg":"Prior rec/g",
    "prior_receiving_yards_pg":"Prior rec yds/g","prior_carries_pg":"Prior carries/g",
    "prior_rushing_yards_pg":"Prior rush yds/g","prior_attempts_pg":"Prior pass att/g",
    "prior_passing_yards_pg":"Prior pass yds/g","age":"Age","years_exp":"Years exp",
    "draft_round":"Draft round","draft_pick":"Draft pick","weight":"Weight","forty":"40 time",
    "team_change":"Changed team",
}


def corr_table(df, target):
    rows = []
    for f in FEATURES:
        sub = df[[f, target]].dropna()
        if len(sub) < 50:
            continue
        rho, p = spearmanr(sub[f], sub[target])
        rows.append({"feature": f, "label": LABELS.get(f, f), "n": len(sub),
                     "spearman": round(rho, 3), "p": p})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    con.close()
    # focus on fantasy-relevant prior seasons (played enough to matter)
    df = df[df["prior_games"] >= 4].copy()
    # naive baseline + residual
    df["resid"] = df["next_ppg"] - df["prior_ppg"]
    print(f"Analysis rows (prior_games>=4): {len(df):,}")

    # ---------- 1. Stickiness ----------
    stick = {}
    for p in POS:
        s = df[df["position"] == p][["prior_ppg","next_ppg"]].dropna()
        rho = spearmanr(s["prior_ppg"], s["next_ppg"])[0]
        r2 = r2_score(s["next_ppg"], s["prior_ppg"])  # how good is "repeat last year"
        stick[p] = {"n": len(s), "spearman": round(rho,3),
                    "r2_repeat": round(r2,3),
                    "mae_repeat": round(mean_absolute_error(s["next_ppg"], s["prior_ppg"]),2)}
    print("\n=== 1. Year-over-year stickiness of PPG ('repeat last year' baseline) ===")
    print(pd.DataFrame(stick).T)

    # ---------- 2. Correlations vs next_ppg (overall + by pos) ----------
    ov = corr_table(df, "next_ppg"); ov.to_csv(os.path.join(OUT,"a1_corr_next_ppg_overall.csv"), index=False)
    print("\n=== 2. Top correlates of next_ppg (overall) ===")
    print(ov.head(15).to_string(index=False))
    bypos = {}
    for p in POS:
        t = corr_table(df[df["position"]==p], "next_ppg"); t["position"]=p; bypos[p]=t
    pd.concat(bypos.values()).to_csv(os.path.join(OUT,"a1_corr_next_ppg_bypos.csv"), index=False)

    # ---------- 3. Residual analysis (beyond repeat-last-year) ----------
    rt = corr_table(df, "resid"); rt.to_csv(os.path.join(OUT,"a1_corr_residual.csv"), index=False)
    print("\n=== 3. What predicts OVER/UNDER-performing last year (residual) ===")
    print(rt.head(15).to_string(index=False))

    # ---------- 4a. Age curves by position ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, p in zip(axes.ravel(), POS):
        s = df[(df["position"]==p) & df["age"].between(21,38)]
        g = s.groupby("age")["next_ppg"].agg(["mean","count"])
        g = g[g["count"]>=8]
        ax.plot(g.index, g["mean"], "o-")
        ax.set_title(f"{p}: next-season PPG by age"); ax.set_xlabel("age"); ax.set_ylabel("PPG")
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"a1_age_curves.png"), dpi=110); plt.close(fig)

    # peak-age table
    age_rows=[]
    for p in POS:
        s = df[(df["position"]==p) & df["age"].between(21,36)]
        g = s.groupby("age")["next_ppg"].mean()
        if len(g): age_rows.append({"position":p,"peak_age":int(g.idxmax()),"peak_ppg":round(g.max(),1)})
    print("\n=== 4a. Approx peak age (next-season PPG) ===")
    print(pd.DataFrame(age_rows).to_string(index=False))

    # ---------- 4b. Draft capital (within experienced players) ----------
    df["draft_bucket"] = pd.cut(df["draft_pick"], [0,15,40,100,200,300],
                                labels=["R1 top15","R1-2","R3-4","R5-7","UDFA"])
    dc = df.groupby(["position","draft_bucket"])["next_ppg"].agg(["mean","count"]).reset_index()
    dc.to_csv(os.path.join(OUT,"a1_draft_capital.csv"), index=False)

    # ---------- 4c. TD regression demonstration ----------
    # Among WR/TE/RB: high prior TD rate -> negative residual next year?
    td = df[df["position"].isin(["WR","TE","RB"])].copy()
    td["td_rate_bucket"] = pd.qcut(td["prior_total_tds"], 4, labels=["low","med","high","elite"], duplicates="drop")
    tdg = td.groupby("td_rate_bucket")["resid"].agg(["mean","count"])
    print("\n=== 4c. Mean residual (next - prior PPG) by prior total TDs ===")
    print(tdg)

    # ---------- 5. Temporal model: prior_ppg only vs full feature set ----------
    model_df = df.dropna(subset=["next_ppg"]).copy()
    train = model_df[model_df["season"] <= 2022]
    test  = model_df[model_df["season"] >= 2023]
    pos_dum_tr = pd.get_dummies(train["position"]); pos_dum_te = pd.get_dummies(test["position"])
    pos_dum_te = pos_dum_te.reindex(columns=pos_dum_tr.columns, fill_value=0)

    Xtr_full = pd.concat([train[FEATURES].fillna(-1).reset_index(drop=True), pos_dum_tr.reset_index(drop=True)], axis=1)
    Xte_full = pd.concat([test[FEATURES].fillna(-1).reset_index(drop=True), pos_dum_te.reset_index(drop=True)], axis=1)
    ytr, yte = train["next_ppg"].values, test["next_ppg"].values

    res = {}
    # baselines
    res["repeat_last_year"] = round(mean_absolute_error(yte, test["prior_ppg"].fillna(test["prior_ppg"].mean())),3)
    gb = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    gb.fit(train[["prior_ppg"]].fillna(-1), ytr)
    res["gb_prioronly"] = round(mean_absolute_error(yte, gb.predict(test[["prior_ppg"]].fillna(-1))),3)
    gb2 = GradientBoostingRegressor(n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
    gb2.fit(Xtr_full, ytr)
    pred_full = gb2.predict(Xte_full)
    res["gb_full"] = round(mean_absolute_error(yte, pred_full),3)
    res["gb_full_r2"] = round(r2_score(yte, pred_full),3)
    print("\n=== 5. Temporal validation (train<=2022, test 2023-25) MAE on next_ppg ===")
    print(json.dumps(res, indent=2))

    imp = pd.DataFrame({"feature": Xtr_full.columns, "importance": gb2.feature_importances_}) \
            .sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(OUT,"a1_model_importance.csv"), index=False)
    print("\nTop 15 model features (full):")
    print(imp.head(15).to_string(index=False))

    # save key tables for the report
    pd.DataFrame(stick).T.to_csv(os.path.join(OUT,"a1_stickiness.csv"))
    dc.to_csv(os.path.join(OUT,"a1_draft_capital.csv"), index=False)
    json.dump(res, open(os.path.join(OUT,"a1_model_results.json"),"w"), indent=2)
    print("\nSaved CSVs + charts to", OUT)


if __name__ == "__main__":
    main()
