"""Year-to-year skill trajectory study.

Questions answered, all from PBP-level advanced stats (nflv_pbp_skill[_wk]),
NGS, and real injury reports (nflv_injuries):

1. STICKINESS - which advanced metrics are skill (stable year-to-year and
   split-half) vs role/context vs noise, per position?
2. AGING - how do per-play SKILL metrics age vs per-game VOLUME, by position?
3. INJURY DRAG - how much do players lose when playing listed (Q/D) or in the
   weeks right after returning from an absence? -> healthy-baseline correction.
4. TREND - does a player's multi-year skill slope (and in-season H2-H1 delta)
   predict NEXT season beyond mean reversion?

Writes outputs/reports/skill_trajectories.md and table player_skill_seasons
(per player-season: era-normalized skill z, healthy-adjusted skill, trend).
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
REPORT = os.path.join(ROOT, "outputs", "reports", "skill_trajectories.md")

MIN_PLAYS = {"pass": 150, "rush": 80, "rec": 40}
ROLE_METRICS = {
    "pass": ["epa_play", "cpoe", "success", "adot", "sack_rate", "comp_pct", "td_rate", "int_rate"],
    "rush": ["epa_play", "success", "ypc", "td_rate"],
    "rec":  ["epa_play", "success", "yacoe", "adot", "catch_rate", "ypt"],
}
POS_ROLE = {"QB": "pass", "RB": "rush", "WR": "rec", "TE": "rec"}
L = []  # report lines


def say(s=""):
    print(s, flush=True)
    L.append(s)


def zscore_by(df, col, by, wcol="plays"):
    """Play-weighted z-score within group (era/position normalization)."""
    def z(g):
        w = g[wcol]
        m = np.average(g[col], weights=w)
        sd = np.sqrt(np.average((g[col] - m) ** 2, weights=w))
        return (g[col] - m) / (sd if sd > 0 else 1.0)
    return df.groupby(by, group_keys=False).apply(z)


def main():
    con = sqlite3.connect(DB)
    sk = pd.read_sql("SELECT * FROM nflv_pbp_skill", con)
    wk = pd.read_sql("SELECT * FROM nflv_pbp_skill_wk", con)
    inj = pd.read_sql("SELECT * FROM nflv_injuries", con)
    ros = pd.read_sql("""SELECT gsis_id AS player_id, position, season, years_exp
                         FROM nflv_rosters WHERE gsis_id IS NOT NULL""", con)
    sd = pd.read_sql("""SELECT player_id, season, position, age, next_ppg, prior_ppg
                        FROM season_dataset""", con)

    # canonical position per player (mode across rosters)
    pos = (ros.groupby("player_id")["position"]
              .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
              .rename("position").reset_index())
    age = ros.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "years_exp"]] \
             .drop_duplicates(["player_id", "season"])

    sk = sk.rename(columns={"gsis_id": "player_id"}).merge(pos, on="player_id", how="left")
    sk = sk.merge(age, on=["player_id", "season"], how="left")
    sk = sk[sk["position"].isin(POS_ROLE)]
    # keep the row matching the player's primary role, plus RB receiving as secondary
    sk["primary"] = sk.apply(lambda r: POS_ROLE.get(r["position"]) == r["role"], axis=1)

    say("# Skill trajectories: what is skill, what is noise, what injuries cost")
    say("")
    say(f"Data: nflv_pbp_skill {sk['season'].min()}-{sk['season'].max()}, "
        f"{sk['player_id'].nunique():,} players, real injury reports "
        f"{inj['season'].min()}-{inj['season'].max()}.")

    # ================= 1. STICKINESS =================
    say("\n## 1. Which advanced metrics are actually skill?")
    say("\nYoY r = correlation season T vs T+1 (same player, qualified both years).")
    say("Split-half r = first-half vs second-half of the SAME season.")
    say("High both = skill. High split-half only = role/context. Low both = noise.\n")
    say("| Pos | Metric | YoY r | Split-half r | n (YoY) | Verdict |")
    say("|---|---|---|---|---|---|")
    stick = {}
    for pos_name, role in POS_ROLE.items():
        d = sk[(sk["position"] == pos_name) & (sk["role"] == role)
               & (sk["plays"] >= MIN_PLAYS[role])].copy()
        nxt = d[["player_id", "season"] + ROLE_METRICS[role]].copy()
        nxt["season"] -= 1
        pair = d.merge(nxt, on=["player_id", "season"], suffixes=("", "_next"))
        for m in ROLE_METRICS[role]:
            yy = pair[[m, f"{m}_next"]].dropna()
            r_yoy, n = (pearsonr(yy[m], yy[f"{m}_next"])[0], len(yy)) if len(yy) > 30 else (np.nan, len(yy))
            hh = d[[f"{m}_h1", f"{m}_h2"]].dropna() if f"{m}_h1" in d else pd.DataFrame()
            r_sh = pearsonr(hh[f"{m}_h1"], hh[f"{m}_h2"])[0] if len(hh) > 30 else np.nan
            verdict = ("**skill**" if (r_yoy > .45 and r_sh > .35) else
                       "role/context" if r_sh > .35 else
                       "part-skill" if r_yoy > .3 else "noise")
            stick[(pos_name, m)] = r_yoy
            say(f"| {pos_name} | {m} | {r_yoy:.2f} | {r_sh if not np.isnan(r_sh) else float('nan'):.2f} | {n} | {verdict} |")

    # ================= era-normalized skill z =================
    frames = []
    for pos_name, role in POS_ROLE.items():
        d = sk[(sk["position"] == pos_name) & (sk["role"] == role)
               & (sk["plays"] >= MIN_PLAYS[role] * 0.5)].copy()
        d["skill_z"] = zscore_by(d, "epa_play", ["position", "season"])
        # secondary skill metric per position
        sec = {"QB": "cpoe", "RB": "success", "WR": "yacoe", "TE": "yacoe"}[pos_name]
        d[sec] = d[sec].fillna(d[sec].median())
        d["skill2_z"] = zscore_by(d, sec, ["position", "season"])
        d["skill_composite"] = 0.65 * d["skill_z"] + 0.35 * d["skill2_z"]
        frames.append(d)
    Z = pd.concat(frames, ignore_index=True)

    # ================= 2. AGING: skill vs volume =================
    say("\n## 2. Skill ages differently than volume")
    say("\nMean YoY change in era-normalized PER-PLAY skill (composite z) vs change in")
    say("PER-GAME volume (plays/game, z), by age bucket. Negative = decline.\n")
    say("| Pos | Age | n | Δ skill z | Δ volume z | Reading |")
    say("|---|---|---|---|---|---|")
    Z["ppg_plays"] = Z["plays"] / Z["games"].clip(lower=1)
    Z["vol_z"] = zscore_by(Z, "ppg_plays", ["position", "season"])
    Z = Z.merge(sd[["player_id", "season", "age"]], on=["player_id", "season"], how="left")
    nxt = Z[["player_id", "season", "skill_composite", "vol_z"]].copy()
    nxt["season"] -= 1
    pairZ = Z.merge(nxt, on=["player_id", "season"], suffixes=("", "_next")).dropna(
        subset=["skill_composite_next", "age"])
    pairZ["d_skill"] = pairZ["skill_composite_next"] - pairZ["skill_composite"]
    pairZ["d_vol"] = pairZ["vol_z_next"] - pairZ["vol_z"]
    buckets = [(0, 23, "≤23"), (24, 25, "24-25"), (26, 27, "26-27"),
               (28, 29, "28-29"), (30, 32, "30-32"), (33, 45, "33+")]
    for pos_name in POS_ROLE:
        for lo, hi, lab in buckets:
            d = pairZ[(pairZ["position"] == pos_name) & pairZ["age"].between(lo, hi)]
            if len(d) < 25:
                continue
            ds, dv = d["d_skill"].mean(), d["d_vol"].mean()
            read = ("skill fades first" if ds < dv - .05 else
                    "role fades first" if dv < ds - .05 else "in step")
            say(f"| {pos_name} | {lab} | {len(d)} | {ds:+.2f} | {dv:+.2f} | {read} |")

    # ================= 3. INJURY DRAG =================
    say("\n## 3. What injuries cost (real injury reports, not games-played proxy)")
    # season_type is only populated from 2025 on; use week cutoffs for REG
    maxwk = np.where(inj["season"] >= 2021, 18, 17)
    inj = inj[inj["week"] <= maxwk]
    inj["listed"] = inj["report_status"].isin(["Questionable", "Doubtful"])
    inj["out"] = inj["report_status"].eq("Out")
    iw = (inj.groupby(["gsis_id", "season", "week"])
             .agg(listed=("listed", "max"), out=("out", "max")).reset_index()
             .rename(columns={"gsis_id": "player_id"}))
    w = wk.rename(columns={"gsis_id": "player_id"}).merge(pos, on="player_id", how="left")
    w = w[w["position"].isin(POS_ROLE)]
    w["primary"] = w.apply(lambda r: POS_ROLE.get(r["position"]) == r["role"], axis=1)
    w = w[w["primary"]]
    w = w.merge(iw, on=["player_id", "season", "week"], how="left")
    w[["listed", "out"]] = w[["listed", "out"]].fillna(False)

    # return-from-absence flag: first 2 games back after missing 2+ weeks
    w = w.sort_values(["player_id", "season", "week"])
    w["wk_gap"] = w.groupby(["player_id", "season"])["week"].diff()
    w["just_back"] = w["wk_gap"] >= 3
    w["ret_window"] = (w["just_back"] |
                       w.groupby(["player_id", "season"])["just_back"].shift(1).fillna(False))

    say("\nPer-play EPA in games played while LISTED (Q/D) or in the 2 games after")
    say("returning from a 2+ week absence, vs each player's own clean-week baseline")
    say("(same season, play-weighted, min 4 clean games):\n")
    say("| Pos | State | n player-wks | EPA/play vs own baseline | ")
    say("|---|---|---|---|")
    inj_drag = {}
    for pos_name in POS_ROLE:
        d = w[w["position"] == pos_name].copy()
        clean = d[~d["listed"] & ~d["ret_window"]]
        base = (clean.groupby(["player_id", "season"])
                     .apply(lambda g: np.average(g["epa_play"], weights=g["plays"])
                            if len(g) >= 4 else np.nan)
                     .rename("base_epa").reset_index())
        d = d.merge(base, on=["player_id", "season"]).dropna(subset=["base_epa"])
        for state, mask in [("listed Q/D", d["listed"] & ~d["ret_window"]),
                            ("return window", d["ret_window"])]:
            s = d[mask]
            if len(s) < 30:
                continue
            delta = np.average(s["epa_play"] - s["base_epa"], weights=s["plays"])
            inj_drag[(pos_name, state)] = delta
            say(f"| {pos_name} | {state} | {len(s)} | {delta:+.03f} |")

    # season-level injury aggregates for the model
    isea = (iw.groupby(["player_id", "season"])
              .agg(inj_weeks_listed=("listed", "sum"), inj_weeks_out=("out", "sum"))
              .reset_index())
    body = (inj.assign(body=inj["report_primary_injury"].str.title())
               .groupby(["gsis_id", "season"])["body"]
               .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
               .rename("inj_primary_body").reset_index()
               .rename(columns={"gsis_id": "player_id"}))

    # healthy-adjusted season skill: recompute season EPA using only clean weeks
    healthy = (w[~w["listed"] & ~w["ret_window"]]
               .groupby(["player_id", "season"])
               .apply(lambda g: pd.Series({
                   "healthy_epa_play": np.average(g["epa_play"], weights=g["plays"]),
                   "healthy_plays": g["plays"].sum(), "clean_games": len(g)}))
               .reset_index())

    # ================= 4. TREND =================
    say("\n## 4. Does skill trend predict the next season?")
    P = Z[Z["primary"] if "primary" in Z else slice(None)].copy()
    P = P[P["plays"] >= P["role"].map(MIN_PLAYS) * 0.6]
    P = P.sort_values(["player_id", "season"])
    g = P.groupby("player_id")
    P["skill_lag1"] = g["skill_composite"].shift(1)
    P["skill_lag2"] = g["skill_composite"].shift(2)
    P["season_lag1"] = g["season"].shift(1)
    contig = (P["season"] - P["season_lag1"]) == 1
    P["trend2"] = np.where(contig, P["skill_composite"] - P["skill_lag1"], np.nan)
    P["inseason_trend"] = np.where(
        P["epa_play_h2"].notna(), P["epa_play_h2"] - P["epa_play_h1"], np.nan)

    nxt = P[["player_id", "season", "skill_composite"]].copy()
    nxt["season"] -= 1
    P = P.merge(nxt, on=["player_id", "season"], how="left", suffixes=("", "_next"))
    P = P.merge(sd[["player_id", "season", "next_ppg", "prior_ppg"]],
                on=["player_id", "season"], how="left")

    say("\nPartial correlations with NEXT-season outcome, controlling current level")
    say("(residualize on current skill_composite + age):\n")
    say("| Pos | Predictor | vs next skill z | vs next PPG | n |")
    say("|---|---|---|---|---|")

    def partial(d, xcol, ycol):
        d = d.dropna(subset=[xcol, ycol, "skill_composite"])
        if len(d) < 40:
            return np.nan, len(d)
        A = np.column_stack([np.ones(len(d)), d["skill_composite"],
                             d["age"].fillna(d["age"].median() if d["age"].notna().any() else 27)])
        rx = d[xcol] - A @ np.linalg.lstsq(A, d[xcol], rcond=None)[0]
        ry = d[ycol] - A @ np.linalg.lstsq(A, d[ycol], rcond=None)[0]
        return pearsonr(rx, ry)[0], len(d)

    P = P.merge(sd[["player_id", "season", "age"]].rename(columns={"age": "age_sd"}),
                on=["player_id", "season"], how="left")
    P["age"] = P["age"] if "age" in P else P["age_sd"]
    P["age"] = P["age"].fillna(P["age_sd"])
    for pos_name in POS_ROLE:
        d = P[P["position"] == pos_name]
        for pred in ["trend2", "inseason_trend"]:
            r1, n1 = partial(d, pred, "skill_composite_next")
            r2, _ = partial(d, pred, "next_ppg")
            say(f"| {pos_name} | {pred} | {r1 if not np.isnan(r1) else float('nan'):.3f} "
                f"| {r2 if not np.isnan(r2) else float('nan'):.3f} | {n1} |")

    # mean reversion magnitude for context
    say("\nMean reversion (all positions): players 1z above era mean keep on average")
    hi = P[P["skill_composite"] > 1].dropna(subset=["skill_composite_next"])
    lo = P[P["skill_composite"] < -1].dropna(subset=["skill_composite_next"])
    if len(hi) and len(lo):
        say(f"{hi['skill_composite_next'].mean():.2f}z next year (n={len(hi)}); players 1z "
            f"below keep {lo['skill_composite_next'].mean():.2f}z (n={len(lo)}).")

    # ================= persist per-player-season table =================
    out = P[["player_id", "position", "season", "role", "games", "plays",
             "epa_play", "skill_z", "skill2_z", "skill_composite",
             "trend2", "inseason_trend", "vol_z"]].copy()
    out = out.merge(healthy, on=["player_id", "season"], how="left")
    out = out.merge(isea, on=["player_id", "season"], how="left")
    out = out.merge(body, on=["player_id", "season"], how="left")
    out[["inj_weeks_listed", "inj_weeks_out"]] = out[["inj_weeks_listed", "inj_weeks_out"]].fillna(0)
    out.to_sql("player_skill_seasons", con, if_exists="replace", index=False)
    say(f"\nSaved table player_skill_seasons: {len(out):,} rows.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nreport -> {REPORT}")
    con.close()


if __name__ == "__main__":
    main()
