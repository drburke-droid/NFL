"""
Forward 2026 season projection.

Builds 2026 feature rows from each player's 2025 production (prior side) + static
context aged forward one year, trains the season model on the full YoY dataset,
and outputs ranked 2026 PPG / finish / VBD projections.

Writes table season_proj_2026 + CSV.
"""
import os, sqlite3
import numpy as np, pandas as pd
import lightgbm as lgb

from build_season_dataset import build_season_frame, FEAT_COLS
from model_season import FEATURES, project_games, add_finish_vbd, POS

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
PARAMS = dict(objective="regression_l1", n_estimators=500, learning_rate=0.03,
              num_leaves=31, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
              random_state=0, verbosity=-1)
TARGET_SEASON = 2026


def main():
    con = sqlite3.connect(DB)
    sf = build_season_frame(con)
    ds = pd.read_sql("SELECT * FROM season_dataset", con)

    base_yr = TARGET_SEASON - 1            # 2025 production drives the projection
    # ---- prior-side production from base year ----
    prior = sf[sf["season"] == base_yr][FEAT_COLS].copy()
    prior.columns = ["player_id","player_display_name","position","prior_season","prior_team"] + \
                    ["prior_"+c for c in FEAT_COLS[5:]]
    prior["season"] = TARGET_SEASON

    # ---- two-years-ago ppg (base_yr-1) ----
    prior2 = sf[sf["season"] == base_yr-1][["player_id","ppg"]].rename(columns={"ppg":"prior2_ppg"})

    # ---- static context aged forward from base year ----
    ctx = sf[sf["season"] == base_yr][["player_id","recent_team","age","years_exp","height",
            "weight","draft_round","draft_pick","forty","vertical","broad_jump","cone","shuttle"]].copy()
    ctx = ctx.rename(columns={"recent_team":"team"})
    ctx["age"] = ctx["age"] + 1
    ctx["years_exp"] = ctx["years_exp"] + 1

    df = prior.merge(prior2, on="player_id", how="left").merge(ctx, on="player_id", how="left")
    df["team_change"] = 0  # offseason moves unknown at projection time
    df["pos_id"] = df["position"].map({p:i for i,p in enumerate(POS)})
    df["proj_games"] = project_games(df.rename(columns={}))  # uses prior_games + position

    # ---- train on all available YoY data, predict 2026 ----
    tr = ds[ds["next_ppg"].notna()].copy()
    tr["pos_id"] = tr["position"].map({p:i for i,p in enumerate(POS)})
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(tr[FEATURES+["pos_id"]].astype(float).fillna(-1), tr["next_ppg"])
    df["pred_ppg"] = m.predict(df[FEATURES+["pos_id"]].astype(float).fillna(-1))

    # only project players with a real base-year role
    df = df[df["prior_games"].fillna(0) >= 3].copy()
    df = add_finish_vbd(df, "pred_ppg", "proj_games", "pred")

    keep = ["player_id","player_display_name","position","team","prior_ppg","pred_ppg",
            "proj_games","proj_total_pred","finish_pred","vbd_pred"]
    out = df[keep].sort_values("vbd_pred", ascending=False)
    out.to_sql("season_proj_2026", con, if_exists="replace", index=False)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "season_proj_2026.csv"), index=False)
    con.close()

    print(f"season_proj_2026: {len(out):,} players")
    print("\nTop 20 by projected VBD:")
    show = out.head(20)[["player_display_name","position","team","prior_ppg","pred_ppg","finish_pred","vbd_pred"]]
    print(show.round(2).to_string(index=False))
    print("\nTop 8 QBs:")
    print(out[out.position=="QB"].head(8)[["player_display_name","team","pred_ppg","finish_pred"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
