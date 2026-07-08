"""
Late-breakout study — who goes UNDRAFTED / near-free at the draft and finishes
the season VORP-POSITIVE (and therefore gets re-priced at the next draft)?

Anchors on DRAFT-DAY PRICE (FFA ADP + AAV, 2016-2025) rather than prior PPG
(the value-leap model's pool), so rookies and never-produced players are in
scope. Outcome is league-scoring VORP > 0 for the season just drafted.

Pipeline
  1. frame:      rostered QB/RB/WR/TE per season Y; draft-time features only
  2. outcome:    league points (PPR + 2*passTD + 1*INT) vs replacement rank
  3. proxy:      validate FFA AAV/ADP vs actual league clearing bids (2023-25)
  4. base rates: P(VORP+ | price bucket x pos x experience)
  5. lifts:      univariate hit-rate lifts inside the cheap pool
  6. model:      walk-forward LightGBM P(hit | cheap), precision@K vs baselines
  7. keeper:     empirical escalation rule + option value of a $1-2 dart
  8. 2026:       score the current cheap pool, write MD report + CSV

Writes outputs/models/LATE_BREAKOUTS.md, outputs/models/late_breakout_2026.csv.
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
OUT = os.path.join(ROOT, "outputs", "models")

POS = ["QB", "RB", "WR", "TE"]
# 12-team league: QB1 RB2 WR2 TE1 FLEX1 -> drafted-starter pool per the board
REPL_RANK = {"QB": 13, "RB": 27, "WR": 35, "TE": 13}
SEASONS = list(range(2016, 2026))          # years with FFA ADP/AAV
CHEAP_ADP = 120                             # later than round 10 of a 12-teamer
CHEAP_AAV = 2.0                             # $1-2 fliers


def league_pts(df):
    # nflverse PPR uses 4 pt pass TD / -2 INT; league is 6 / -1
    return (df["fantasy_points_ppr"].fillna(0)
            + 2.0 * df["passing_tds"].fillna(0)
            + 1.0 * df["passing_interceptions"].fillna(0))


def load_frames(con):
    seas = pd.read_sql("SELECT * FROM nflv_season WHERE season >= 2014", con)
    seas["lg_pts"] = league_pts(seas)
    seas["ppg"] = seas["lg_pts"] / seas["games"].replace(0, np.nan)

    ros = pd.read_sql("""SELECT season, team, position, status, full_name, gsis_id,
                                pfr_id, years_exp, entry_year, draft_number, birth_date
                         FROM nflv_rosters WHERE position IN ('QB','RB','WR','TE')""", con)
    ros = ros[ros.status.isin(["ACT", "RES", "DEV", "INA"])].dropna(subset=["gsis_id"])
    ros["age"] = ros.season - pd.to_datetime(ros.birth_date, errors="coerce").dt.year
    ros = ros.sort_values("status").drop_duplicates(["gsis_id", "season"])  # ACT first

    ffa = pd.read_sql("SELECT * FROM nflv_ffa_proj WHERE season >= 2015", con)
    ffa = ffa.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season"])

    draft = pd.read_sql("SELECT season AS draft_year, round AS draft_round, pick AS draft_pick, gsis_id AS player_id FROM nflv_draft", con)
    draft = draft.dropna(subset=["player_id"]).drop_duplicates("player_id")

    wk = pd.read_sql("""SELECT player_id, season, fantasy_points_ppr FROM nflv_weekly
                        WHERE season >= 2014 AND position IN ('QB','RB','WR','TE')""", con)
    cv = (wk.groupby(["player_id", "season"]).fantasy_points_ppr
            .agg(["mean", "std"]).reset_index())
    cv["cv"] = cv["std"] / cv["mean"].replace(0, np.nan)
    cv = cv[["player_id", "season", "cv"]]

    snaps = pd.read_sql("""SELECT season, pfr_player_id, AVG(offense_pct) AS snap_pct
                           FROM nflv_snaps WHERE game_type='REG'
                           GROUP BY season, pfr_player_id""", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT * FROM nflv_opportunity", con)
    return seas, ros, ffa, draft, cv, snaps, ht, opp


def replacement_table(seas):
    """Per season+position: total league points of the replacement-rank player."""
    rows = []
    for (y, p), g in seas[seas.position.isin(POS)].groupby(["season", "position"]):
        pts = g.lg_pts.sort_values(ascending=False).values
        r = REPL_RANK[p]
        rows.append({"season": y, "position": p,
                     "repl_pts": pts[r - 1] if len(pts) >= r else 0.0})
    return pd.DataFrame(rows)


def build_dataset(con):
    seas, ros, ffa, draft, cv, snaps, ht, opp = load_frames(con)
    repl = replacement_table(seas)

    frames = []
    for Y in SEASONS + [2026]:
        u = ros[ros.season == Y].copy()
        u = u.rename(columns={"gsis_id": "player_id", "full_name": "name"})

        f = ffa[ffa.season == Y][["player_id", "ffa_points", "ffa_ceiling", "ffa_sd",
                                  "ffa_uncertainty", "ffa_adp", "ffa_aav", "ffa_rank"]]
        u = u.merge(f, on="player_id", how="left")
        u["in_ffa"] = u.ffa_points.notna().astype(int)

        pr = seas[seas.season == Y - 1][["player_id", "recent_team", "games", "ppg", "lg_pts",
                                         "target_share", "wopr", "receiving_epa", "rushing_epa",
                                         "carries", "targets", "passing_tds", "rushing_tds",
                                         "receiving_tds"]].copy()
        pr["prior_tds"] = pr[["passing_tds", "rushing_tds", "receiving_tds"]].sum(axis=1)
        pr["carries_pg"] = pr.carries / pr.games.replace(0, np.nan)
        pr["targets_pg"] = pr.targets / pr.games.replace(0, np.nan)
        pr = pr[["player_id", "recent_team", "games", "ppg", "lg_pts", "target_share", "wopr",
                 "receiving_epa", "rushing_epa", "carries_pg", "targets_pg", "prior_tds"]]
        pr.columns = ["player_id"] + ["prior_" + c for c in pr.columns[1:]]
        u = u.merge(pr, on="player_id", how="left")

        p2 = seas[seas.season == Y - 2][["player_id", "ppg"]].rename(columns={"ppg": "prior2_ppg"})
        u = u.merge(p2, on="player_id", how="left")

        u = u.merge(cv[cv.season == Y - 1][["player_id", "cv"]].rename(columns={"cv": "prior_cv"}),
                    on="player_id", how="left")
        u = u.merge(snaps[snaps.season == Y - 1][["pfr_player_id", "snap_pct"]]
                    .rename(columns={"pfr_player_id": "pfr_id", "snap_pct": "prior_snap_pct"}),
                    on="pfr_id", how="left")
        u = u.merge(ht[ht.season == Y - 1].drop(columns=["season"]), on="player_id", how="left")
        u = u.merge(opp[opp.season == Y].drop(columns=["season", "team", "position"]),
                    on="player_id", how="left")

        u = u.merge(draft, on="player_id", how="left")
        u["team_change"] = ((u.prior_recent_team.notna()) & (u.team != u.prior_recent_team)).astype(int)
        u["is_rookie"] = (u.entry_year == Y).astype(int)
        # UDFA -> pseudo round 8 / pick 270
        u["draft_round"] = u.draft_round.fillna(8)
        u["draft_pick"] = u.draft_pick.fillna(270)

        # keep plausibly draftable players: in FFA, or played last year, or a rookie
        u = u[(u.in_ffa == 1) | (u.prior_games.fillna(0) > 0) | (u.is_rookie == 1)]

        # outcome (year Y actuals; 2026 unknown)
        act = seas[seas.season == Y][["player_id", "lg_pts", "games", "ppg"]]
        act.columns = ["player_id", "act_pts", "act_games", "act_ppg"]
        u = u.merge(act, on="player_id", how="left")
        u = u.merge(repl[repl.season == Y][["position", "repl_pts"]], on="position", how="left")
        u["vorp"] = u.act_pts.fillna(0) - u.repl_pts
        u["hit"] = (u.vorp > 0).astype(int)

        # next-draft re-pricing
        nx = ffa[ffa.season == Y + 1][["player_id", "ffa_adp", "ffa_aav"]]
        nx.columns = ["player_id", "next_adp", "next_aav"]
        u = u.merge(nx, on="player_id", how="left")

        u["season"] = Y
        frames.append(u)

    df = pd.concat(frames, ignore_index=True)
    df["adp_f"] = df.ffa_adp.fillna(300)
    df["aav_f"] = df.ffa_aav.fillna(0)
    df["cheap"] = ((df.adp_f > CHEAP_ADP) & (df.aav_f <= CHEAP_AAV)).astype(int)
    df["ffa_upside"] = df.ffa_ceiling - df.ffa_points
    return df


# ---------------------------------------------------------------- reporting
def price_proxy_check(df):
    """FFA AAV vs actual league clearing bids, by name-season match."""
    ed = pd.read_csv(os.path.join(ROOT, "outputs", "espn_drafts.csv"))
    ed = ed[(~ed.keeper) & ed.pos.isin(POS)]
    ed["key"] = ed.player.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    d = df.copy()
    d["key"] = d.name.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    m = ed.merge(d[["key", "season", "ffa_aav", "adp_f", "hit", "vorp"]],
                 on=["key", "season"], how="inner")
    r_aav = m[["bid", "ffa_aav"]].dropna().corr(method="spearman").iloc[0, 1]
    r_adp = m[["bid", "adp_f"]].corr(method="spearman").iloc[0, 1]
    cheap_bids = m[(m.adp_f > CHEAP_ADP) & (m.ffa_aav.fillna(0) <= CHEAP_AAV)].bid
    return m, r_aav, r_adp, cheap_bids


def keeper_escalation():
    ed = pd.read_csv(os.path.join(ROOT, "outputs", "espn_drafts.csv"))
    ed["key"] = ed.player.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    rows = []
    for y in [2024, 2025]:
        k = ed[(ed.season == y) & (ed.keeper)]
        p = ed[ed.season == y - 1][["key", "bid"]].rename(columns={"bid": "prior_bid"})
        rows.append(k.merge(p, on="key", how="left"))
    k = pd.concat(rows)
    k["delta"] = k.bid - k.prior_bid
    return k


def base_rates(df):
    d = df[df.season <= 2025]
    adp_bins = [0, 36, 72, 120, 168, 240, 400]
    labels = ["1-36", "37-72", "73-120", "121-168", "169+", "not in FFA"]
    d = d.copy()
    d["adp_bucket"] = pd.cut(d.adp_f, adp_bins, labels=labels)
    tab = d.groupby("adp_bucket").agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                      mean_vorp=("vorp", "mean"))
    pos_tab = d.pivot_table(index="adp_bucket", columns="position", values="hit", aggfunc="mean")
    ch = d[d.cheap == 1]
    exp_tab = ch.groupby(ch.years_exp.clip(0, 8)).agg(n=("hit", "size"), hit_rate=("hit", "mean"))
    return tab, pos_tab, exp_tab, ch


def repricing(df):
    """Does a cheap hit actually get re-priced next draft?"""
    ch = df[(df.cheap == 1) & (df.season <= 2024)].copy()
    g = ch.groupby("hit").agg(n=("next_aav", "size"),
                              in_ffa_next=("next_adp", lambda s: s.notna().mean()),
                              med_next_aav=("next_aav", "median"),
                              p75_next_aav=("next_aav", lambda s: s.quantile(.75)),
                              med_next_adp=("next_adp", "median"))
    return g


def univariate_lifts(ch):
    base = ch.hit.mean()
    checks = {
        "rookie R1-2 draft capital": (ch.is_rookie == 1) & (ch.draft_round <= 2),
        "rookie R3-4": (ch.is_rookie == 1) & (ch.draft_round.between(3, 4)),
        "rookie day 3 / UDFA": (ch.is_rookie == 1) & (ch.draft_round >= 5),
        "year 2-3, drafted R1-2": (ch.years_exp.between(1, 2)) & (ch.draft_round <= 2) & (ch.is_rookie == 0),
        "year 2-3, drafted R3+": (ch.years_exp.between(1, 2)) & (ch.draft_round >= 3) & (ch.is_rookie == 0),
        "vet 4+ yrs": ch.years_exp >= 3,
        "age <= 24": ch.age <= 24,
        "prior snap% >= 40": ch.prior_snap_pct >= 0.40,
        "prior tgt share >= 12%": ch.prior_target_share >= 0.12,
        "late-season surge (ht_d_ppg >= +2)": ch.ht_d_ppg >= 2,
        "H2 snap climb (ht_d_snap >= +10%)": ch.ht_d_snap >= 0.10,
        "vacated backfield touches": ch.vac_rb_carries.fillna(0) >= 100,
        "vacated targets >= 80": ch.vac_pc_targets.fillna(0) >= 80,
        "team change": ch.team_change == 1,
        "in FFA file at all (adp 121+)": ch.in_ffa == 1,
        "prior PPG 5-9 (fringe role)": ch.prior_ppg.between(5, 9),
        "prior PPG < 3 or none": ch.prior_ppg.fillna(0) < 3,
    }
    rows = []
    for k, mask in checks.items():
        sub = ch[mask.fillna(False)]
        if len(sub) >= 25:
            rows.append({"signal": k, "n": len(sub), "hit_rate": sub.hit.mean(),
                         "lift": sub.hit.mean() / base})
    return pd.DataFrame(rows).sort_values("lift", ascending=False), base


# ---------------------------------------------------------------- model
FEATS = ["adp_f", "aav_f", "in_ffa", "ffa_points", "ffa_upside", "ffa_uncertainty",
         "age", "years_exp", "is_rookie", "draft_round", "draft_pick",
         "prior_games", "prior_ppg", "prior2_ppg", "prior_cv", "prior_snap_pct",
         "prior_target_share", "prior_wopr", "prior_receiving_epa", "prior_rushing_epa",
         "prior_carries_pg", "prior_targets_pg", "prior_tds", "team_change",
         "ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_d_tch", "ht_slope",
         "vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets",
         "pos_QB", "pos_RB", "pos_TE", "pos_WR"]


def walk_forward(df):
    import lightgbm as lgb
    ch = df[df.cheap == 1].copy()
    ch = pd.concat([ch, pd.get_dummies(ch.position, prefix="pos")], axis=1)
    for c in FEATS:
        if c not in ch.columns: ch[c] = np.nan
    params = dict(n_estimators=350, learning_rate=0.03, num_leaves=31,
                  min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                  random_state=0, verbosity=-1)
    preds = []
    for Y in range(2019, 2026):
        tr = ch[(ch.season < Y) & (ch.season <= 2025)]
        te = ch[ch.season == Y]
        if not len(te): continue
        m = lgb.LGBMClassifier(**params).fit(tr[FEATS], tr.hit)
        p = te.copy(); p["prob"] = m.predict_proba(te[FEATS])[:, 1]
        preds.append(p)
    bt = pd.concat(preds)

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(bt.hit, bt.prob)
    ap = average_precision_score(bt.hit, bt.prob)

    def prec_at_k(frame, score, k=10, ascending=False):
        hits = []
        for y, g in frame.groupby("season"):
            top = g.sort_values(score, ascending=ascending).head(k)
            hits.append(top.hit.mean())
        return np.mean(hits)

    res = {
        "auc": auc, "ap": ap, "base": bt.hit.mean(), "n": len(bt),
        "p10_model": prec_at_k(bt, "prob"),
        "p10_ffa": prec_at_k(bt.assign(s=-bt.adp_f.rank() + bt.in_ffa * 1000), "s"),
        "p10_prior": prec_at_k(bt, "prior_ppg"),
        "p10_ffa_pts": prec_at_k(bt, "ffa_points"),
    }
    # calibration bins
    bt["bin"] = pd.qcut(bt.prob, 10, duplicates="drop")
    cal = bt.groupby("bin").agg(pred=("prob", "mean"), obs=("hit", "mean"), n=("hit", "size"))
    # feature importance from final model
    fm = lgb.LGBMClassifier(**params).fit(ch[ch.season <= 2025][FEATS], ch[ch.season <= 2025].hit)
    imp = pd.Series(fm.feature_importances_, index=FEATS).sort_values(ascending=False)
    # 2026 scores
    sc26 = ch[ch.season == 2026].copy()
    if len(sc26):
        sc26["prob"] = fm.predict_proba(sc26[FEATS])[:, 1]
    return bt, res, cal, imp, sc26


def main():
    con = sqlite3.connect(DB)
    df = build_dataset(con)
    print(f"dataset: {len(df)} player-seasons, {df[df.season<=2025].hit.sum()} hits, "
          f"cheap pool {df[(df.cheap==1)&(df.season<=2025)].shape[0]} "
          f"({df[(df.cheap==1)&(df.season<=2025)].hit.mean():.3f} hit rate)")

    # sanity: famous late breakouts should be cheap hits
    for nm, yr in [("Puka Nacua", 2023), ("James Robinson", 2020), ("Amon-Ra St. Brown", 2021),
                   ("Kyren Williams", 2023), ("Jordan Addison", 2023)]:
        r = df[(df.name == nm) & (df.season == yr)]
        if len(r):
            r = r.iloc[0]
            print(f"  {nm} {yr}: adp={r.adp_f:.0f} aav={r.aav_f:.1f} cheap={r.cheap} "
                  f"vorp={r.vorp:.0f} hit={r.hit} next_aav={r.next_aav}")

    m, r_aav, r_adp, cheap_bids = price_proxy_check(df)
    print(f"\nprice proxy: n={len(m)} matched bids; spearman(bid, ffa_aav)={r_aav:.3f}, "
          f"(bid, adp)={r_adp:.3f}")
    print(f"cheap-pool actual bids: n={len(cheap_bids)}, median=${cheap_bids.median():.0f}, "
          f"p90=${cheap_bids.quantile(.9):.0f}")

    k = keeper_escalation()
    print(f"\nkeeper escalation (n={k.delta.notna().sum()}): "
          f"median +${k.delta.median():.0f}, mean +${k.delta.mean():.1f}")
    print(k[["season", "player", "bid", "prior_bid", "delta"]].dropna().head(20).to_string(index=False))

    tab, pos_tab, exp_tab, ch = base_rates(df)
    print("\nP(VORP+ | ADP bucket):\n", tab.round(3).to_string())
    print("\nby position:\n", pos_tab.round(3).to_string())
    print("\ncheap pool by years_exp:\n", exp_tab.round(3).to_string())

    print("\nre-pricing next draft (cheap pool):\n", repricing(df).round(2).to_string())

    lifts, base = univariate_lifts(ch)
    print(f"\ncheap-pool base rate {base:.3f}; lifts:\n", lifts.round(3).to_string(index=False))

    bt, res, cal, imp, sc26 = walk_forward(df)
    print("\nwalk-forward 2019-2025:", {k: round(v, 3) for k, v in res.items()})
    print("\ncalibration:\n", cal.round(3).to_string())
    print("\ntop features:\n", imp.head(15).to_string())

    if len(sc26):
        cols = ["name", "position", "team", "age", "years_exp", "draft_round",
                "prior_ppg", "prior_snap_pct", "adp_f", "aav_f", "prob"]
        top = sc26.sort_values("prob", ascending=False).head(30)[cols]
        print("\n2026 top cheap-pool breakout candidates:\n", top.round(3).to_string(index=False))
        sc26.sort_values("prob", ascending=False)[cols + ["ffa_points", "vac_pc_targets",
            "vac_rb_carries", "ht_d_ppg", "is_rookie"]].to_csv(
            os.path.join(OUT, "late_breakout_2026.csv"), index=False)

    # stash intermediate artifacts for the report step
    df.to_pickle(os.path.join(OUT, "late_breakout_frame.pkl"))
    bt.to_pickle(os.path.join(OUT, "late_breakout_bt.pkl"))
    print("\nsaved frame + backtest pickles, 2026 CSV")


if __name__ == "__main__":
    main()
