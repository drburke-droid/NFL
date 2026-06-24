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


def build_skill(con):
    # board_2026 = veterans (w/ breakout_prob, certainty inputs) + 2026 rookies (w/ hit_prob)
    proj = pd.read_sql("SELECT * FROM board_2026", con)
    s25 = pd.read_sql("""SELECT player_id, games, passing_tds, passing_interceptions,
                         fantasy_points_ppr FROM nflv_season WHERE season=2025""", con).drop_duplicates("player_id")
    df = proj.merge(s25, on="player_id", how="left")

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
          "breakout_prob","hit_prob","is_rookie"]
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
    for c in ["breakout_prob","hit_prob"]:
        allp[c]=allp[c].round(3)
    out_cols=["overall_rank","name","position","team","age","pos_rank","tier","proj_pts",
              "proj_games","proj_ppg","vorp","repl_pts","actual_2025","delta_ly","prior_ppg",
              "is_flex_starter","conf","breakout_prob","hit_prob","is_rookie"]
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
