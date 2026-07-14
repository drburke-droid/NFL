"""Trajectory archetypes: cluster player-season CAREER STATES and measure
next-season outcome distributions per archetype.

Unlike cluster_archetypes.py (what kind of player is this?), this asks
"where is this player in what kind of arc?" — combining age, era-normalized
skill level, multi-year skill trend, volume trend, and injury load from real
injury reports. KMeans per position; archetype names derived from centroids.

Outcome per archetype: next-season PPG delta, P(breakout +20%), P(bust -30%),
next-season availability. All outcomes are OUT of the clustering features.

Writes outputs/reports/trajectory_archetypes.md and table player_traj_arch.
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
REPORT = os.path.join(ROOT, "outputs", "reports", "trajectory_archetypes.md")
POS = ["QB", "RB", "WR", "TE"]
K = {"QB": 5, "RB": 6, "WR": 6, "TE": 5}
STATE_FEATS = ["age", "skill_composite", "trend2", "vol_z", "d_vol",
               "ppg_z", "games", "inj_weeks_out", "inj_weeks_listed"]
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def name_cluster(c, pos_med_age):
    """Rule-based name from centroid (original units)."""
    age = "Young" if c["age"] <= pos_med_age - 1.5 else ("Vet" if c["age"] >= pos_med_age + 1.5 else "Prime")
    lvl = "Elite" if c["ppg_z"] > 0.8 else ("Solid" if c["ppg_z"] > 0 else "Fringe")
    if c["inj_weeks_out"] > 2.2:
        tag = "Injury-Hit"
    elif c["trend2"] > 0.25 and c["d_vol"] > 0.1:
        tag = "Ascending"
    elif c["trend2"] > 0.25:
        tag = "Skill-Spiking"
    elif c["trend2"] < -0.25 and c["d_vol"] < -0.1:
        tag = "Fading"
    elif c["trend2"] < -0.25:
        tag = "Efficiency-Eroding"
    elif c["d_vol"] > 0.15:
        tag = "Role-Growing"
    elif c["d_vol"] < -0.15:
        tag = "Role-Shrinking"
    else:
        tag = "Steady"
    return f"{age} {lvl} {tag}"


def main():
    con = sqlite3.connect(DB)
    ps = pd.read_sql("SELECT * FROM player_skill_seasons", con)
    sd = pd.read_sql("""SELECT player_id, season, position, age, prior_ppg,
                               next_ppg, next_games FROM season_dataset""", con)
    traj = pd.read_sql("SELECT player_id, season, ppg, games FROM nflv_traj", con)

    # current-season PPG, era-normalized within position-season
    cur = traj.rename(columns={"ppg": "cur_ppg", "games": "cur_games"})
    df = ps.merge(cur, on=["player_id", "season"], how="left")
    df = df.merge(sd[["player_id", "season", "age", "next_ppg", "next_games"]],
                  on=["player_id", "season"], how="left", suffixes=("", "_sd"))
    # season_dataset rows are keyed by TARGET season; age there is for season T.
    # ps season = current season; next outcome lives in sd row season+1.
    nxt = sd.rename(columns={"season": "tgt_season"})
    nxt["season"] = nxt["tgt_season"] - 1
    df = ps.merge(cur, on=["player_id", "season"], how="left")
    df = df.merge(nxt[["player_id", "season", "age", "next_ppg", "next_games"]],
                  on=["player_id", "season"], how="left")
    df["age"] = df["age"] - 1  # sd age is for target season; current age = -1

    df["ppg_z"] = df.groupby(["position", "season"])["cur_ppg"].transform(
        lambda s: (s - s.mean()) / (s.std() or 1))
    # volume trend: change in vol_z vs prior season
    df = df.sort_values(["player_id", "season"])
    g = df.groupby("player_id")
    df["d_vol"] = np.where((df["season"] - g["season"].shift(1)) == 1,
                           df["vol_z"] - g["vol_z"].shift(1), np.nan)

    say("# Trajectory archetypes: career states and what happens next")
    say("")
    say("Clustered on: age, era-z skill (per-play), 2-yr skill trend, era-z volume,")
    say("volume trend, era-z PPG, games, injury weeks (out / listed). Outcomes below")
    say("are next-season results — NOT used in clustering.\n")

    all_lab = []
    for pos in POS:
        d = df[df["position"] == pos].copy()
        X = d[STATE_FEATS].copy()
        med = X.median()
        X = X.fillna(med)
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=K[pos], n_init=20, random_state=0).fit(Xs)
        d["cluster"] = km.labels_
        cent = pd.DataFrame(X, columns=STATE_FEATS).groupby(km.labels_).mean()
        names = {i: name_cluster(cent.loc[i], d["age"].median()) for i in cent.index}
        d["archetype"] = d["cluster"].map(names)
        all_lab.append(d)

        say(f"\n## {pos}")
        say("| Archetype | n | age | skill z | trend | vol z | PPG z | inj out | "
            "→ ΔPPG | →P(+20%) | →P(-30%) | →games |")
        say("|---|---|---|---|---|---|---|---|---|---|---|---|")
        out = d.dropna(subset=["next_ppg"])
        out = out[out["cur_ppg"].notna() & (out["cur_ppg"] > 0)]
        rows = []
        for i in cent.index:
            s = out[out["cluster"] == i]
            if len(s) < 15:
                continue
            dppg = (s["next_ppg"] - s["cur_ppg"]).mean()
            brk = (s["next_ppg"] >= 1.2 * s["cur_ppg"]).mean()
            bust = (s["next_ppg"] <= 0.7 * s["cur_ppg"]).mean()
            rows.append((names[i], len(d[d["cluster"] == i]), cent.loc[i], dppg, brk,
                         bust, s["next_games"].mean()))
        for nm, n, c, dppg, brk, bust, ng in sorted(rows, key=lambda r: -r[3]):
            say(f"| {nm} | {n} | {c['age']:.1f} | {c['skill_composite']:+.2f} | "
                f"{c['trend2']:+.2f} | {c['vol_z']:+.2f} | {c['ppg_z']:+.2f} | "
                f"{c['inj_weeks_out']:.1f} | {dppg:+.2f} | {brk:.0%} | {bust:.0%} | {ng:.1f} |")

    lab = pd.concat(all_lab, ignore_index=True)
    keep = ["player_id", "position", "season", "cluster", "archetype",
            "skill_composite", "trend2", "vol_z", "d_vol", "ppg_z",
            "inj_weeks_out", "inj_weeks_listed"]
    lab[keep].to_sql("player_traj_arch", con, if_exists="replace", index=False)
    say(f"\nSaved table player_traj_arch: {len(lab):,} rows.")

    # 2025 examples for face validity
    say("\n## 2025 season states (spot check)")
    names25 = pd.read_sql("SELECT DISTINCT player_id, player_display_name FROM nflv_traj", con)
    ex = lab[lab["season"] == 2025].merge(names25, on="player_id", how="left")
    for pos in POS:
        top = ex[(ex["position"] == pos)].nlargest(8, "cur_ppg")
        say(f"\n**{pos}**: " + "; ".join(
            f"{r.player_display_name} — {r.archetype}" for r in top.itertuples()))

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nreport -> {REPORT}")
    con.close()


if __name__ == "__main__":
    main()
