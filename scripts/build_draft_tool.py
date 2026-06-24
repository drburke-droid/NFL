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
RECENT_W = {2025: 0.6, 2024: 0.4, 2023: 0.2}  # recent-form blend for K/DST


def project_recent(by_season):
    avail = {s: by_season[s] for s in RECENT_W if s in by_season and pd.notna(by_season[s])}
    if not avail: return np.nan
    wsum = sum(RECENT_W[s] for s in avail)
    return sum(RECENT_W[s]*by_season[s] for s in avail)/wsum


def build_skill(con):
    proj = pd.read_sql("SELECT * FROM season_proj_2026", con)
    s25 = pd.read_sql("""SELECT player_id, games, passing_tds, passing_interceptions,
                         fantasy_points_ppr FROM nflv_season WHERE season=2025""", con).drop_duplicates("player_id")
    ros = pd.read_sql("SELECT gsis_id player_id, birth_date FROM nflv_rosters WHERE season=2025", con)
    ros["age"] = TARGET - pd.to_datetime(ros["birth_date"], errors="coerce").dt.year
    ros = ros[["player_id","age"]].drop_duplicates("player_id")
    df = proj.merge(s25, on="player_id", how="left").merge(ros, on="player_id", how="left")

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
    return df, repl, starters


def build_simple(hist, key, position, name_fn, proj_games, con):
    """Build K or DST projections (12 starters, no flex)."""
    rows=[]
    for kid, g in hist.groupby(key):
        by = dict(zip(g["season"], g["custom_pts"]))
        proj = project_recent(by)
        if pd.isna(proj): continue
        latest = g.sort_values("season").iloc[-1]
        rows.append({"name": name_fn(latest), "position": position,
                     "team": latest.get("team", latest[key] if key=="team" else None),
                     "proj_pts": proj, "actual_2025": by.get(2025, np.nan),
                     "proj_games": proj_games})
    d = pd.DataFrame(rows)
    # keep likely starters: top ~ (TEAMS+ a few) plus anyone with real 2025 usage
    d = d.sort_values("proj_pts", ascending=False)
    d["pos_rank"] = d["proj_pts"].rank(ascending=False, method="min").astype(int)
    pp = d["proj_pts"].values
    repl = float(pp[TEAMS]) if TEAMS < len(pp) else float(pp[-1])
    d["repl_pts"] = repl
    d["vorp"] = d["proj_pts"] - repl
    d["proj_ppg"] = d["proj_pts"]/d["proj_games"]
    d["is_flex_starter"] = 0; d["age"] = np.nan; d["prior_ppg"] = np.nan
    return d, repl


def main():
    con = sqlite3.connect(DB)
    skill, repl, starters = build_skill(con)

    # Kickers: keep starters (>=6 games in 2025 or top-40 by projection)
    kick = pd.read_sql("SELECT * FROM nflv_kicking", con)
    k25 = kick[kick.season==2025].set_index("player_id")["games"].to_dict()
    kdf, krepl = build_simple(kick, "player_id", "K", lambda r: r["name"], 16, con)
    kdf = kdf[kdf.apply(lambda r: True, axis=1)].head(40)  # cap depth
    # DST
    tdef = pd.read_sql("SELECT * FROM nflv_team_def", con)
    ddf, drepl = build_simple(tdef, "team", "DST", lambda r: str(r["team"])+" DST", 17, con)
    ddf["team"] = ddf["name"].str.replace(" DST","",regex=False)
    con.close()

    repl.update({"K": round(krepl,1), "DST": round(drepl,1)})
    starters.update({"K": TEAMS, "DST": TEAMS})

    cols=["name","position","team","age","pos_rank","proj_pts","proj_games","proj_ppg",
          "vorp","repl_pts","actual_2025","prior_ppg","is_flex_starter"]
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
    out_cols=["overall_rank","name","position","team","age","pos_rank","tier","proj_pts",
              "proj_games","proj_ppg","vorp","repl_pts","actual_2025","delta_ly","prior_ppg","is_flex_starter"]
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
                  "QB/RB/WR/TE: prior-year-anchored 2026 model. K & DST: recent-form blend (0.6x'25 + 0.4x'24).",
                  "K/DST scored exactly under league kicking/defense rules; FFA 2026 consensus not yet available."],
        "player_count": len(records),
    }
    with open(os.path.join(OUTDIR,"data.js"),"w",encoding="utf-8") as f:
        f.write("const META = "+json.dumps(meta)+";\n")
        f.write("const PLAYERS = "+json.dumps(records)+";\n")

    print(f"Wrote {len(records)} players ({(allp.position=='K').sum()} K, {(allp.position=='DST').sum()} DST)")
    print("Replacement points:", meta["replacement_points"])
    for p in ["QB","RB","WR","TE","K","DST"]:
        top=allp[allp.position==p].nlargest(1,"proj_pts").iloc[0]
        print(f"  {p}1: {top['name']:24s} proj {top['proj_pts']:.0f}  VORP {top['vorp']:.0f}")


if __name__ == "__main__":
    main()
