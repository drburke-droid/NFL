"""Do PBP-level skill signals improve next-season projection?

Walk-forward gauntlet (train < T, predict T, 2016-2025), per position:
  A  base       = FEATURES + games-proxy injury features (current internal set)
  B  +skill     = A + PBP skill level/trend/healthy-baseline + real injury reports
  C  +skill+ngs = B + Next Gen Stats priors (2016+)
  D  market     = A + FFA consensus (the production anchor)
  E  everything = D + skill + NGS  (does skill add beyond the market?)

Also: injury-return subgroup MAE (players with 2+ weeks OUT in prior season),
new-feature importances, and 2026 projections from the best config with
trajectory-archetype labels attached -> table skill_projections_2026.

Report -> outputs/reports/player_yoy_skill_model.md
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("ms", os.path.join(HERE, "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "db", "nfl_odds.db")
REPORT = os.path.join(ROOT, "outputs", "reports", "player_yoy_skill_model.md")
POS = ["QB", "RB", "WR", "TE"]
GB = dict(objective="regression_l1", n_estimators=300, learning_rate=0.04, num_leaves=20,
          min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

SKILLF = ["p_skill_comp", "p_skill_trend2", "p_inseason_trend", "p_vol_z", "p_d_vol",
          "p_ppg_z", "p_healthy_gap", "p_clean_games", "p_inj_out", "p_inj_listed",
          "p2_inj_out", "p2_skill_comp", "skill_2yr_avg"]
NGSF = ["ngs_time_to_throw", "ngs_aggressiveness", "ngs_cpae",
        "ngs_separation", "ngs_cushion", "ngs_share_air", "ngs_yac_above_exp",
        "ngs_ryoe_att", "ngs_efficiency", "ngs_stacked_box"]
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def skill_feature_block(con):
    """Prior-season (and T-2) skill/injury features keyed to (player_id, target season)."""
    ps = pd.read_sql("SELECT * FROM player_skill_seasons", con)
    ta = pd.read_sql("SELECT player_id, season, d_vol, ppg_z FROM player_traj_arch", con)
    ps = ps.merge(ta, on=["player_id", "season"], how="left")
    ps["healthy_gap"] = ps["healthy_epa_play"] - ps["epa_play"]

    p1 = ps[["player_id", "season", "skill_composite", "trend2", "inseason_trend",
             "vol_z", "d_vol", "ppg_z", "healthy_gap", "clean_games",
             "inj_weeks_out", "inj_weeks_listed"]].copy()
    p1.columns = ["player_id", "season", "p_skill_comp", "p_skill_trend2",
                  "p_inseason_trend", "p_vol_z", "p_d_vol", "p_ppg_z",
                  "p_healthy_gap", "p_clean_games", "p_inj_out", "p_inj_listed"]
    p1["season"] += 1
    p2 = ps[["player_id", "season", "skill_composite", "inj_weeks_out"]].copy()
    p2.columns = ["player_id", "season", "p2_skill_comp", "p2_inj_out"]
    p2["season"] += 2
    blk = p1.merge(p2, on=["player_id", "season"], how="outer")
    blk["skill_2yr_avg"] = blk[["p_skill_comp", "p2_skill_comp"]].mean(axis=1)

    # NGS season aggregates (week=0), prior season -> target season key
    qb = pd.read_sql("""SELECT player_gsis_id player_id, season,
                        avg_time_to_throw ngs_time_to_throw, aggressiveness ngs_aggressiveness,
                        completion_percentage_above_expectation ngs_cpae
                        FROM nflv_ngs_passing WHERE week=0""", con)
    rc = pd.read_sql("""SELECT player_gsis_id player_id, season,
                        avg_separation ngs_separation, avg_cushion ngs_cushion,
                        percent_share_of_intended_air_yards ngs_share_air,
                        avg_yac_above_expectation ngs_yac_above_exp
                        FROM nflv_ngs_receiving WHERE week=0""", con)
    ru = pd.read_sql("""SELECT player_gsis_id player_id, season,
                        rush_yards_over_expected_per_att ngs_ryoe_att,
                        efficiency ngs_efficiency,
                        percent_attempts_gte_eight_defenders ngs_stacked_box
                        FROM nflv_ngs_rushing WHERE week=0""", con)
    ngs = qb.merge(rc, on=["player_id", "season"], how="outer") \
            .merge(ru, on=["player_id", "season"], how="outer")
    ngs = ngs.drop_duplicates(["player_id", "season"])
    ngs["season"] += 1
    return blk.merge(ngs, on=["player_id", "season"], how="outer")


def run_config(df, feats, label, results):
    for pos in POS:
        d = df[df["position"] == pos]
        preds = []
        for T in range(2016, 2026):
            tr, te = d[d["season"] < T], d[d["season"] == T]
            if len(te) == 0 or len(tr) < 60:
                continue
            m = lgb.LGBMRegressor(**GB)
            m.fit(tr[feats].astype(float).fillna(-1), tr["next_ppg"])
            t2 = te.copy()
            t2["pred"] = m.predict(te[feats].astype(float).fillna(-1))
            preds.append(t2)
        P = pd.concat(preds, ignore_index=True)
        results.setdefault(label, {})[pos] = P
    return results


def main():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM season_dataset", con)
    ns = pd.read_sql("SELECT player_id, season, games FROM nflv_season", con) \
           .drop_duplicates(["player_id", "season"])
    df = MS.attach_ffa(df, con)
    g2 = ns.rename(columns={"games": "prior2_games"}); g2["season"] += 2
    df = df.merge(g2, on=["player_id", "season"], how="left")
    df = MS.add_injury_features(df)
    blk = skill_feature_block(con)
    df = df.merge(blk, on=["player_id", "season"], how="left")
    df = df[df["next_ppg"].notna() & (df["prior_games"] >= 3)].copy()

    A = MS.FEATURES + MS.INJURY_FEATURES
    CFG = {
        "A base": A,
        "B +skill": A + SKILLF,
        "C +skill+ngs": A + SKILLF + NGSF,
        "D market(FFA)": A + MS.FFA_FEATURES,
        "E market+skill+ngs": A + MS.FFA_FEATURES + SKILLF + NGSF,
    }

    say("# Year-to-year skill model: does play-by-play skill signal beat the market?")
    say("")
    say(f"Rows: {len(df):,} player-seasons (2012-2025 priors, targets 2016-2025 walk-forward).")
    say(f"Skill-feature coverage: {df['p_skill_comp'].notna().mean():.0%} of rows; "
        f"NGS coverage {df['ngs_separation'].notna().mean():.0%} (2017+ targets only).")
    say("\n## Central projection MAE (PPG), walk-forward 2016-2025\n")

    results = {}
    for label, feats in CFG.items():
        run_config(df, feats, label, results)

    hdr = "| Pos | " + " | ".join(CFG) + " | best |"
    say(hdr); say("|" + "---|" * (len(CFG) + 2))
    for pos in POS:
        maes = {lab: mean_absolute_error(results[lab][pos]["next_ppg"],
                                         results[lab][pos]["pred"]) for lab in CFG}
        best = min(maes, key=maes.get)
        say(f"| {pos} | " + " | ".join(f"{maes[l]:.3f}" for l in CFG) + f" | {best} |")

    say("\nSpearman rho (E vs D):")
    for pos in POS:
        rE = spearmanr(results["E market+skill+ngs"][pos]["pred"],
                       results["E market+skill+ngs"][pos]["next_ppg"])[0]
        rD = spearmanr(results["D market(FFA)"][pos]["pred"],
                       results["D market(FFA)"][pos]["next_ppg"])[0]
        say(f"  {pos}: E {rE:.3f} vs D {rD:.3f}")

    # injury-return subgroup
    say("\n## Injury-return subgroup (2+ weeks OUT on injury report in prior season)\n")
    say("| Pos | n | A base MAE | B +skill MAE | Δ |")
    say("|---|---|---|---|---|")
    for pos in POS:
        a = results["A base"][pos]; b = results["B +skill"][pos]
        m = a["p_inj_out"].fillna(0) >= 2
        if m.sum() < 15:
            continue
        ma = mean_absolute_error(a[m]["next_ppg"], a[m]["pred"])
        mb = mean_absolute_error(b[m]["next_ppg"], b[m]["pred"])
        say(f"| {pos} | {int(m.sum())} | {ma:.3f} | {mb:.3f} | {mb-ma:+.3f} |")

    # importances of new features in config E, full fit
    full_feats = CFG["E market+skill+ngs"]
    mf = lgb.LGBMRegressor(**GB).fit(df[full_feats].astype(float).fillna(-1), df["next_ppg"])
    imp = pd.DataFrame({"feature": full_feats, "imp": mf.feature_importances_})
    newf = imp[imp["feature"].isin(SKILLF + NGSF)].sort_values("imp", ascending=False)
    say("\n## New-feature importance (config E, full fit)\n")
    say("| Feature | importance | rank among all |")
    say("|---|---|---|")
    imp = imp.sort_values("imp", ascending=False).reset_index(drop=True)
    rank = {f: i + 1 for i, f in enumerate(imp["feature"])}
    for r in newf.head(10).itertuples():
        say(f"| {r.feature} | {r.imp} | {rank[r.feature]}/{len(full_feats)} |")

    # ---------- 2026 projections with best config ----------
    say("\n## 2026 projections (trained on all data, best all-feature config)")
    bsd_spec = importlib.util.spec_from_file_location("bsd", os.path.join(HERE, "build_season_dataset.py"))
    bsd = importlib.util.module_from_spec(bsd_spec); bsd_spec.loader.exec_module(bsd)
    sf = bsd.build_season_frame(con)
    prior = sf[sf.season == 2025][bsd.FEAT_COLS].copy()
    prior.columns = (["player_id", "player_display_name", "position", "prior_season", "prior_team"]
                     + ["prior_" + c for c in bsd.FEAT_COLS[5:]])
    prior["season"] = 2026
    prior2 = sf[sf.season == 2024][["player_id", "ppg", "games"]] \
        .rename(columns={"ppg": "prior2_ppg", "games": "prior2_games"})
    ctx = sf[sf.season == 2025][["player_id", "recent_team", "age", "years_exp", "height",
                                 "weight", "draft_round", "draft_pick", "forty", "vertical",
                                 "broad_jump", "cone", "shuttle"]].rename(columns={"recent_team": "team"})
    ctx["age"] += 1; ctx["years_exp"] += 1
    v26 = MS.attach_ffa(prior.merge(prior2, on="player_id", how="left")
                             .merge(ctx, on="player_id", how="left"), con)
    try:
        r26 = pd.read_sql("SELECT player_id, team AS team_2026 FROM nflv_rosters_2026", con) \
                .drop_duplicates("player_id")
        v26 = v26.merge(r26, on="player_id", how="left")
        v26["team_change"] = (v26.team_2026.notna() & v26.team.notna()
                              & (v26.team_2026 != v26.team)).astype(int)
    except Exception:
        v26["team_change"] = 0
    v26 = v26[v26["prior_games"].fillna(0) >= 3].copy()
    v26 = MS.add_injury_features(v26)
    v26 = v26.merge(blk, on=["player_id", "season"], how="left")

    use_ffa = v26["ffa_points"].notna().sum() >= 20
    FEATS = A + SKILLF + NGSF + (MS.FFA_FEATURES if use_ffa else [])
    say(f"\nProduction features: base+inj+skill+ngs{'+FFA' if use_ffa else ' (no 2026 FFA yet)'}")

    arch = pd.read_sql("""SELECT player_id, archetype FROM player_traj_arch
                          WHERE season=2025""", con)
    out = []
    for pos in POS:
        tr = df[df["position"] == pos]
        te = v26[v26["position"] == pos].copy()
        if len(te) == 0:
            continue
        for a, nm in [(0.15, "floor"), (0.5, "central"), (0.85, "ceiling")]:
            m = lgb.LGBMRegressor(**{**GB, "objective": "quantile", "alpha": a})
            m.fit(tr[FEATS].astype(float).fillna(-1), tr["next_ppg"])
            te[nm] = m.predict(te[FEATS].astype(float).fillna(-1)).clip(min=0)
        out.append(te)
    R = pd.concat(out, ignore_index=True).merge(arch, on="player_id", how="left")
    keep = ["player_id", "player_display_name", "position", "team", "archetype",
            "central", "floor", "ceiling", "p_skill_comp", "p_skill_trend2",
            "p_inj_out", "p_healthy_gap"]
    R[keep].round(2).to_sql("skill_projections_2026", con, if_exists="replace", index=False)
    say(f"\nSaved skill_projections_2026: {len(R)} players.")
    for pos in POS:
        top = R[R["position"] == pos].nlargest(10, "central")
        say(f"\n**{pos} top 10 (central PPG | floor-ceiling | archetype):**")
        for r in top.itertuples():
            say(f"- {r.player_display_name}: {r.central:.1f} ({r.floor:.1f}-{r.ceiling:.1f}) — "
                f"{r.archetype if isinstance(r.archetype, str) else 'n/a'}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nreport -> {REPORT}")
    con.close()


if __name__ == "__main__":
    main()
