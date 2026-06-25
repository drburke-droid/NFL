"""
Build the local draft-tool dataset (data.js) for the upcoming season under the
user's exact 12-team league + custom scoring. Covers QB/RB/WR/TE (model
projections) plus K and DST (recent-form projections).

Scoring note: the league's RB/WR/TE scoring is identical to nflverse PPR, so
projected PPR == custom points there. QB totals are adjusted for 6-pt pass TD /
-1 INT. K and DST are scored exactly under the league's kicking/defense rules
(see fetch_kdst.py) and projected from recent seasons.

VORP: 12 teams; starters QB1/RB2/WR2/TE1/FLEX1(RB/WR/TE)/K1/DST1. Replacement =
best non-starter at each position (skill positions also absorb the 12 flex spots).
"""
import os, json, sqlite3
import numpy as np, pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "draft_tool")
os.makedirs(OUTDIR, exist_ok=True)
TEAMS = 12
BASELINE = {"QB": TEAMS, "RB": TEAMS*2, "WR": TEAMS*2, "TE": TEAMS, "K": TEAMS, "DST": TEAMS}
FLEX_POS = ["RB", "WR", "TE"]
FLEX_SPOTS = TEAMS
TIER_GAP = {"QB": 20, "RB": 18, "WR": 18, "TE": 14, "K": 8, "DST": 8}
TARGET = 2026


def assign_archetype(r):
    """Player archetype from pre-known traits (combine size/speed + 2025 usage).
    Returns (key, human label). Same thresholds as the validated study
    (scripts/test_archetypes.py). Generic archetypes return an empty label."""
    nz = lambda k: (0.0 if pd.isna(r.get(k)) else float(r.get(k)))
    p = r["position"]; forty = r.get("forty"); ht = r.get("height"); wt = r.get("weight")
    cpg, rpg, tpg, tsh = nz("carries_pg"), nz("receptions_pg"), nz("targets_pg"), nz("target_share")
    aysh = r.get("air_yards_share"); rec_sh = rpg / max(cpg + rpg, 1e-6)
    if p == "RB":
        if tpg >= 3.0 or rec_sh >= 0.28: return "RB_receiving", "Pass-catch RB"
        if cpg >= 11: return "RB_workhorse", "Workhorse RB"
        return "RB_rotational", ""
    if p == "WR":
        fast = pd.notna(forty) and forty <= 4.45
        small = pd.notna(ht) and ht <= 71 and (pd.isna(wt) or wt <= 190)
        big = pd.notna(ht) and ht >= 74 and pd.notna(wt) and wt >= 210
        deep = pd.notna(aysh) and aysh >= 0.32
        if fast and deep: return "WR_deep_threat", "Deep threat"
        if small: return "WR_slot_small", "Slot/undersized"
        if big: return "WR_big_possession", "Big possession"
        return "WR_balanced", ""
    if p == "TE":
        if tsh >= 0.14 or tpg >= 4.5: return "TE_receiving", "Receiving TE"
        return "TE_inline", ""
    if p == "QB":
        if cpg >= 5.0: return "QB_mobile", "Mobile QB"
        return "QB_pocket", ""
    return "", ""


def archetype_age_risk(key, age):
    """Age-fragile archetypes (validated decline cliffs, ARCHETYPE_AGING.md)."""
    if age is None or pd.isna(age): return 0
    return int((key == "QB_mobile" and age >= 29) or (key == "WR_deep_threat" and age >= 28)
               or (key == "RB_receiving" and age >= 29))


def build_skill(con):
    # board_2026 = veterans (w/ breakout_prob, certainty inputs) + 2026 rookies (w/ hit_prob)
    proj = pd.read_sql("SELECT * FROM board_2026", con)
    s25 = pd.read_sql("""SELECT player_id, games, passing_tds, passing_interceptions,
                         fantasy_points_ppr FROM nflv_season WHERE season=2025""", con).drop_duplicates("player_id")
    df = proj.merge(s25, on="player_id", how="left")
    # 2025 first-half vs second-half role/production trend (scouting layer, NOT a
    # projection input — the change itself has no walk-forward value, see test_half_trend.py)
    hs = pd.read_sql("""SELECT player_id, ppg_H1 trend_h1, ppg_H2 trend_h2, d_ppg trend_dppg,
                        d_snap trend_dsnap, snap_H2 trend_snap2, d_tch trend_dtch, d_tgtsh trend_dtgtsh, g_H2 trend_g2
                        FROM half_split_2025""", con).drop_duplicates("player_id")
    df = df.merge(hs, on="player_id", how="left")
    exp = pd.read_sql("SELECT player_id, years_exp+1 exp26 FROM season_dataset WHERE season=2025",
                      con).drop_duplicates("player_id")
    df = df.merge(exp, on="player_id", how="left")
    # "Won the job (young)" flag — the one mechanism-validated case where a late role
    # surge is a real signal (early-career player who ENDED 2025 entrenched after an
    # in-season jump; the full-season line understates them). See test_risers_conditional.py:
    # this cell beats projection ~+0.7 PPG and replicates across 2013-19 AND 2020-25;
    # the same surge in veterans does the opposite. Kept as a display tag, not a proj change.
    df["won_job"] = ((df["exp26"] <= 2) & (df["age"] <= 25) & (df["trend_snap2"] >= 55) &
                     (df["trend_dsnap"] >= 12) & (df["trend_g2"] >= 3) &
                     (df["trend_h2"] > df["trend_h1"]) & (df["trend_h2"] >= 8) &
                     (df["is_rookie"] != 1)).fillna(False).astype(int)
    # "Vacated lead role" flag — validated ~+1.7 PPG signal (test_opportunity_signals.py):
    # a returning RB whose backfield lost its lead back (>=150 carries) with no real
    # replacement. Computed from CURRENT 2026 rosters in nflv_opportunity.
    opp = pd.read_sql("SELECT player_id, vacated_role, vac_rb_carries FROM nflv_opportunity WHERE season=2026",
                      con).drop_duplicates("player_id")
    df = df.merge(opp, on="player_id", how="left")
    df["vacated_role"] = df["vacated_role"].fillna(0).astype(int)
    df["vac_rb_carries"] = df["vac_rb_carries"].fillna(0).round(0)
    # archetype label + age-decline caution (display-only context; not a projection input,
    # see outputs/models/ARCHETYPE_AGING.md). Inputs: 2025 usage + combine size/speed.
    us = pd.read_sql("""SELECT player_id, games, carries, targets, receptions,
                        target_share, air_yards_share FROM nflv_season WHERE season=2025""",
                     con).drop_duplicates("player_id")
    for c in ["carries", "targets", "receptions"]:
        us[c + "_pg"] = us[c] / us["games"].clip(lower=1)
    cb = pd.read_sql("SELECT player_id, forty, height, weight FROM season_dataset", con)
    cb = cb.groupby("player_id").agg(forty=("forty", "median"), height=("height", "median"),
                                     weight=("weight", "median")).reset_index()
    df = df.merge(us[["player_id", "carries_pg", "targets_pg", "receptions_pg", "target_share", "air_yards_share"]],
                  on="player_id", how="left").merge(cb, on="player_id", how="left")
    arch = df.apply(assign_archetype, axis=1)
    df["arch_key"] = [a[0] for a in arch]; df["archetype"] = [a[1] for a in arch]
    df["age_risk"] = [archetype_age_risk(k, a) for k, a in zip(df["arch_key"], df["age"])]
    # value-leap: cheap players w/ breakout upside (season analog of explosion; validated)
    vl = pd.read_sql("SELECT player_id, leap_prob FROM nflv_value_leap", con).drop_duplicates("player_id")
    df = df.merge(vl, on="player_id", how="left")
    im = pd.read_sql("SELECT player_id, fade_prob FROM nflv_implosion", con).drop_duplicates("player_id")
    df = df.merge(im, on="player_id", how="left")
    # the vacancy is team-level; flag only the top returning RB with a real role
    # (not every backup on the team) so the tag points at the actual beneficiary
    df.loc[(df.vacated_role == 1) & (df.pred_ppg < 6), "vacated_role"] = 0
    vr = df[df.vacated_role == 1]
    if len(vr):
        win = vr.loc[vr.groupby("team")["pred_ppg"].idxmax()].index
        df.loc[(df.vacated_role == 1) & (~df.index.isin(win)), "vacated_role"] = 0

    df["proj_total_nfl"] = df["pred_ppg"] * df["proj_games"]
    qb = df["position"] == "QB"
    rate_td = (df["passing_tds"]/df["games"].clip(lower=1)); rate_int = (df["passing_interceptions"]/df["games"].clip(lower=1))
    rate_td = rate_td.fillna(rate_td[qb].median()); rate_int = rate_int.fillna(rate_int[qb].median())
    df["proj_pts"] = df["proj_total_nfl"]
    df.loc[qb,"proj_pts"] = df.loc[qb,"proj_total_nfl"] + 2*(rate_td[qb]*df.loc[qb,"proj_games"]) + 1*(rate_int[qb]*df.loc[qb,"proj_games"])
    df["actual_2025"] = df["fantasy_points_ppr"]
    df.loc[qb,"actual_2025"] = df.loc[qb,"fantasy_points_ppr"] + 2*df.loc[qb,"passing_tds"].fillna(0) + 1*df.loc[qb,"passing_interceptions"].fillna(0)
    df = df[df["proj_pts"].notna()].copy()
    df["proj_ppg"] = df["pred_ppg"]
    df = df.rename(columns={"player_display_name":"name"})

    df["pos_rank"] = df.groupby("position")["proj_pts"].rank(ascending=False, method="min").astype(int)
    # flex VORP
    flex_pool = df[(df.position.isin(FLEX_POS)) & (df.apply(lambda r: r["pos_rank"] > BASELINE[r["position"]], axis=1))]
    flex_starters = flex_pool.sort_values("proj_pts", ascending=False).head(FLEX_SPOTS)
    flex_count = flex_starters["position"].value_counts().to_dict()
    starters = {p: BASELINE[p] + (flex_count.get(p,0) if p in FLEX_POS else 0) for p in ["QB","RB","WR","TE"]}
    repl = {}
    for p in ["QB","RB","WR","TE"]:
        pp = df[df.position==p].sort_values("proj_pts", ascending=False)["proj_pts"].values
        repl[p] = float(pp[starters[p]]) if starters[p] < len(pp) else float(pp[-1])
    df["repl_pts"] = df["position"].map(repl)
    df["vorp"] = df["proj_pts"] - df["repl_pts"]
    df["is_flex_starter"] = df["player_id"].isin(flex_starters["player_id"]).astype(int)
    df["conf"] = np.where(df["is_rookie"]==1, "rookie", "model")
    return df, repl, starters


def build_kdst(con):
    """K/DST from the flattened, validated projections (nflv_kdst_proj). These
    positions are near-random year-over-year, so values are heavily shrunk and
    flagged low-confidence; VORP collapses to near-replacement (stream them)."""
    proj = pd.read_sql("SELECT * FROM nflv_kdst_proj", con)
    k25 = pd.read_sql("SELECT name, custom_pts a25 FROM nflv_kicking WHERE season=2025", con)
    d25 = pd.read_sql("SELECT team, custom_pts a25 FROM nflv_team_def WHERE season=2025", con)
    out = {}
    for pos, games, a25 in [("K", 16, k25), ("DST", 17, d25)]:
        s = proj[proj.position == pos].copy().sort_values("proj_pts", ascending=False)
        s["proj_games"] = games; s["proj_ppg"] = s["proj_pts"]/games
        s["pos_rank"] = s["proj_pts"].rank(ascending=False, method="min").astype(int)
        pp = s["proj_pts"].values
        repl = float(pp[TEAMS]) if TEAMS < len(pp) else float(pp[-1])
        s["repl_pts"] = repl; s["vorp"] = s["proj_pts"] - repl
        s["is_flex_starter"] = 0; s["age"] = np.nan; s["prior_ppg"] = np.nan; s["conf"] = "low"
        s["breakout_prob"] = np.nan; s["hit_prob"] = np.nan; s["is_rookie"] = 0
        s["bust"] = 0.6; s["boom"] = 0.05; s["floor"] = np.nan; s["ceiling"] = np.nan   # streamed: high bust, low boom
        for c in ["trend_h1","trend_h2","trend_dppg","trend_dsnap","trend_dtch","trend_dtgtsh","trend_g2"]:
            s[c] = np.nan
        s["won_job"] = 0; s["vacated_role"] = 0; s["vac_rb_carries"] = 0
        s["archetype"] = ""; s["age_risk"] = 0; s["leap_prob"] = np.nan; s["fade_prob"] = np.nan
        key = "name" if pos == "K" else "team"
        s = s.merge(a25, on=key, how="left"); s["actual_2025"] = s["a25"]
        out[pos] = (s, repl)
    return out


def main():
    con = sqlite3.connect(DB)
    skill, repl, starters = build_skill(con)

    kdst = build_kdst(con)
    kdf, krepl = kdst["K"]; ddf, drepl = kdst["DST"]
    con.close()

    repl.update({"K": round(krepl,1), "DST": round(drepl,1)})
    starters.update({"K": TEAMS, "DST": TEAMS})

    cols=["name","position","team","age","pos_rank","proj_pts","proj_games","proj_ppg",
          "vorp","repl_pts","actual_2025","prior_ppg","is_flex_starter","conf",
          "breakout_prob","hit_prob","is_rookie","bust","boom","floor","ceiling",
          "trend_h1","trend_h2","trend_dppg","trend_dsnap","trend_dtch","trend_dtgtsh","trend_g2","won_job",
          "vacated_role","vac_rb_carries","archetype","age_risk","leap_prob","fade_prob"]
    allp = pd.concat([skill[cols], kdf[cols], ddf[cols]], ignore_index=True)
    allp["delta_ly"] = allp["proj_pts"] - allp["actual_2025"]
    allp = allp.sort_values("vorp", ascending=False).reset_index(drop=True)
    allp["overall_rank"] = np.arange(1, len(allp)+1)

    # tiers per position by VORP gaps
    allp["tier"]=0
    for p,gap in TIER_GAP.items():
        sub = allp[allp.position==p].sort_values("vorp", ascending=False)
        tier=1; prev=None
        for i in sub.index:
            v=allp.at[i,"vorp"]
            if prev is not None and (prev-v)>gap: tier+=1
            allp.at[i,"tier"]=tier; prev=v

    for c in ["proj_pts","proj_games","proj_ppg","vorp","repl_pts","actual_2025","delta_ly","prior_ppg","age"]:
        allp[c]=allp[c].round(1)
    for c in ["breakout_prob","hit_prob","bust","boom","leap_prob","fade_prob"]:
        allp[c]=allp[c].round(3)
    for c in ["floor","ceiling","trend_h1","trend_h2","trend_dppg","trend_dtch"]:
        allp[c]=allp[c].round(1)
    for c in ["trend_dsnap","trend_dtgtsh","trend_g2"]:
        allp[c]=allp[c].round(0)
    out_cols=["overall_rank","name","position","team","age","pos_rank","tier","proj_pts",
              "proj_games","proj_ppg","vorp","repl_pts","actual_2025","delta_ly","prior_ppg",
              "is_flex_starter","conf","breakout_prob","hit_prob","is_rookie","bust","boom","floor","ceiling",
              "trend_h1","trend_h2","trend_dppg","trend_dsnap","trend_dtch","trend_dtgtsh","trend_g2","won_job",
              "vacated_role","vac_rb_carries","archetype","age_risk","leap_prob","fade_prob"]
    import math
    records = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
               for r in allp[out_cols].to_dict(orient="records")]

    meta = {
        "target_season": TARGET, "teams": TEAMS,
        "starters": {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX (RB/WR/TE)":1,"K":1,"DST":1},
        "replacement_points": {k: round(v,1) for k,v in repl.items()},
        "starters_by_pos_incl_flex": starters,
        "scoring": {"Pass Yds":0.04,"Pass TD":6,"INT":-1,"Rush/Rec Yds":0.1,"Rush/Rec TD":6,
                    "Reception":1,"Fumble":-2,"FG<50":3,"FG40-49":4,"FG50+":5,"PAT":1,
                    "Sack":0.5,"DEF INT/FR":1,"DEF TD":6,"Safety/Block":2},
        "notes": ["RB/WR/TE custom scoring == nflverse PPR (exact); QB adjusted for 6-pt pass TD & -1 INT.",
                  "Veterans: prior-year-anchored 2026 model + Brk% = P(>=+4 PPG jump vs 2025). Rookies (2026 class) projected from draft capital + landing spot + combine; Hit% = P(startable rookie year). No 2026 ECR/ADP yet — model guesses.",
                  "K & DST are near-random year-over-year (Spearman ~0.17) — projections heavily shrunk, flagged 'stream'; don't draft for them.",
                  "Tags: ROOK = 2026 rookie; BREAK = veteran breakout prob >= 50%; SLEEP = cheap (<= $8) with breakout prob >= 45%."],
        "player_count": len(records), "rookies": int((allp.is_rookie==1).sum()),
    }
    payload = "const META = "+json.dumps(meta)+";\n" + "const PLAYERS = "+json.dumps(records)+";\n"
    DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    for d in (OUTDIR, DOCS):                       # local copy + GitHub Pages copy
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "data.js"), "w", encoding="utf-8") as f:
            f.write(payload)

    print(f"Wrote {len(records)} players ({(allp.position=='K').sum()} K, {(allp.position=='DST').sum()} DST)")
    print("Replacement points:", meta["replacement_points"])
    for p in ["QB","RB","WR","TE","K","DST"]:
        top=allp[allp.position==p].nlargest(1,"proj_pts").iloc[0]
        print(f"  {p}1: {top['name']:24s} proj {top['proj_pts']:.0f}  VORP {top['vorp']:.0f}")


if __name__ == "__main__":
    main()
