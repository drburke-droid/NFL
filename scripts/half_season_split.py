"""
First-half vs second-half 2025 split — find players whose role/production grew
late in the season (a leading signal for next-year value).

Splits the 2025 regular season at Week 9 (H1 = wk 1-9, H2 = wk 10-18) and, per
QB/RB/WR/TE, compares per-game fantasy production (this league's scoring) plus
ROLE metrics that reveal a genuine usage change rather than noise:
  - touches/g (carries + targets)        - target_share
  - snap %  (from nflv_snaps)             - wopr (WR/TE opportunity)

Writes half_split_2025 to the DB and a markdown report.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "HALF_SEASON_SPLIT_2025.md")
CUT = 9  # H1 = weeks 1..9, H2 = weeks 10..18


def fpts(d):
    return (d.passing_yards.fillna(0)*0.04 + d.passing_tds.fillna(0)*6 - d.passing_interceptions.fillna(0)
            + d.rushing_yards.fillna(0)*0.1 + d.rushing_tds.fillna(0)*6 - d.rushing_fumbles.fillna(0)*2
            + d.receiving_yards.fillna(0)*0.1 + d.receiving_tds.fillna(0)*6 + d.receptions.fillna(0))


def main():
    con = sqlite3.connect(DB)
    w = pd.read_sql("""SELECT * FROM nflv_weekly
                       WHERE season=2025 AND season_type='REG' AND week<=18
                       AND position IN ('QB','RB','WR','TE')""", con)
    # snap % per player-week (match on name+team since snaps lack gsis_id)
    sn = pd.read_sql("""SELECT season,week,player,team,offense_pct FROM nflv_snaps
                        WHERE season=2025 AND game_type='REG' AND week<=18""", con)
    con.close()
    nm = lambda s: s.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    w["_k"] = nm(w.player_display_name) + "|" + w.team
    sn["_k"] = nm(sn.player) + "|" + sn.team
    snap = sn.groupby(["_k", "week"])["offense_pct"].max().reset_index()
    w = w.merge(snap, on=["_k", "week"], how="left")

    w["fp"] = fpts(w)
    w["touches"] = w.carries.fillna(0) + w.targets.fillna(0)
    w["half"] = np.where(w.week <= CUT, "H1", "H2")

    g = w.groupby(["player_id", "player_display_name", "position", "team", "half"]).agg(
        g=("week", "nunique"), ppg=("fp", "mean"), tch=("touches", "mean"),
        tgtsh=("target_share", "mean"), wopr=("wopr", "mean"), snap=("offense_pct", "mean")
    ).reset_index()

    piv = g.pivot_table(index=["player_id", "player_display_name", "position", "team"],
                        columns="half", values=["g", "ppg", "tch", "tgtsh", "wopr", "snap"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index().rename(columns={"player_display_name": "player"})
    for c in ["g_H1", "g_H2"]:
        piv[c] = piv[c].fillna(0)
    for c in ["snap_H1", "snap_H2"]:               # 0-1 fraction -> percentage points
        piv[c] = (piv[c] * 100).round(0)
    piv["d_ppg"] = (piv.ppg_H2.fillna(0) - piv.ppg_H1.fillna(0)).round(1)
    piv["d_snap"] = (piv.snap_H2 - piv.snap_H1).round(0)
    piv["d_tch"] = (piv.tch_H2 - piv.tch_H1).round(1)
    piv["d_tgtsh"] = ((piv.tgtsh_H2 - piv.tgtsh_H1) * 100).round(1)
    for c in ["ppg_H1", "ppg_H2", "tch_H1", "tch_H2"]:
        piv[c] = piv[c].round(1)

    con = sqlite3.connect(DB)
    piv.to_sql("half_split_2025", con, if_exists="replace", index=False)
    con.close()

    def fmt(df, cols, hdr):
        df = df.copy()
        for c in ["snap_H1", "snap_H2", "d_snap"]:
            if c in df: df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.0f}%")
        out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"]*len(hdr)) + "|"]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(out)

    L = ["# 2025 First-Half vs Second-Half Split",
         f"\nH1 = weeks 1-{CUT}, H2 = weeks {CUT+1}-18. Scoring = your league (6-pt pass TD, PPR).",
         "`d_ppg` = H2 PPG − H1 PPG. `d_snap`/`d_tch`/`d_tgtsh` = role change (snap %, touches/g, target-share pts).",
         "A points jump *backed by* a snap/touch/target-share jump is a real role change — those carry forward.\n"]

    SUST = ["position", "player", "team", "g_H1", "ppg_H1", "g_H2", "ppg_H2", "d_ppg", "d_snap", "d_tch", "d_tgtsh"]
    HDR = ["Pos", "Player", "Tm", "G1", "PPG1", "G2", "PPG2", "ΔPPG", "Δsnap", "Δtch/g", "Δtgt%"]

    # 1) Sustained risers: played enough in both halves, biggest PPG jump
    a = piv[(piv.g_H1 >= 3) & (piv.g_H2 >= 3)].sort_values("d_ppg", ascending=False).head(30)
    L += ["## Biggest risers (≥3 games each half) — sustained role/production growth\n", fmt(a, SUST, HDR), ""]

    # 2) Late emergers: little/no H1, strong H2 (role found mid-season)
    b = piv[(piv.g_H2 >= 4) & (piv.ppg_H2 >= 9) & ((piv.g_H1 <= 2) | (piv.ppg_H1 < 6))].copy()
    b = b.sort_values("ppg_H2", ascending=False).head(25)
    L += ["## Late emergers — minimal H1 role, productive H2 (new role mid-season)\n", fmt(b, SUST, HDR), ""]

    # 3) Faders: opposite, for context (avoid drafting on first-half mirage)
    c = piv[(piv.g_H1 >= 3) & (piv.g_H2 >= 3)].sort_values("d_ppg").head(20)
    L += ["## Faders — lost role/production in H2 (don't overpay on H1 numbers)\n", fmt(c, SUST, HDR), ""]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L))

    print(f"Saved half_split_2025 ({len(piv)} players) and {os.path.relpath(OUT)}")
    print("\n=== Biggest sustained risers (>=3 G each half) ===")
    show = ["position", "player", "team", "ppg_H1", "ppg_H2", "d_ppg", "d_snap", "d_tch", "d_tgtsh"]
    pd.set_option("display.width", 160)
    print(a[show].head(20).to_string(index=False))
    print("\n=== Late emergers (minimal H1, productive H2) ===")
    print(b[show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
