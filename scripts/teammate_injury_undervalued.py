"""
Find 2025 pass-catchers whose production dropped when their starting QB missed an extended stretch.
If that QB is healthy for 2026, the player's 2025 rate (which projections lean on) understates him.

Method (2025 raw play-by-play — nflverse weekly endpoint 404s for 2025):
  - per team, find the season-primary QB (most attempts) and the weeks he was OUT (<10 att).
  - keep teams where he missed >= EXT_GAMES (an extended injury).
  - for each pass-catcher on those teams, split PPR points/game by QB-in vs QB-out.
  - flag real contributors (in-PPG >= MIN_IN) who dropped materially (out/in <= DROP).
  - join docs/data.js to show the 2026 projection + our board value for the undervaluation.
The Chase-Burrow check (scripts/chase_burrow_2025.py) is the control: it should NOT flag (Chase
held up without Burrow).
"""
import warnings; warnings.filterwarnings("ignore")
import json, os, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet"
EXT_GAMES, MIN_IN, DROP, MIN_GM = 3, 8.0, 0.78, 5   # QB played >=5 of the split games (i.e. the primary starter)
norm = lambda s: re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower())).replace("  ", " ").strip()


def main():
    cols = ["week", "posteam", "passer_player_id", "passer_player_name", "pass_attempt",
            "receiver_player_id", "rusher_player_id", "complete_pass", "yards_gained",
            "pass_touchdown", "rush_attempt", "rush_touchdown", "season_type"]
    p = pd.read_parquet(PBP, columns=cols); p = p[p["season_type"] == "REG"]

    # --- primary QB per team + the weeks he missed ---
    qa = p[p["pass_attempt"] == 1].groupby(["posteam", "week", "passer_player_id", "passer_player_name"]).size().reset_index(name="att")
    prim = qa.groupby(["posteam", "passer_player_id", "passer_player_name"])["att"].sum().reset_index()
    prim = prim.sort_values("att").groupby("posteam").tail(1).set_index("posteam")  # top QB per team
    team_weeks = p.groupby("posteam")["week"].unique()
    qb_wk_att = qa.groupby(["posteam", "week", "passer_player_id"])["att"].sum()
    inj = {}                       # team -> (qb_name, set(out_weeks), set(in_weeks))
    for tm, row in prim.iterrows():
        qid = row["passer_player_id"]; wks = set(team_weeks[tm])
        in_w = {w for w in wks if qb_wk_att.get((tm, w, qid), 0) >= 10}
        out_w = wks - in_w
        if len(out_w) >= EXT_GAMES and len(in_w) >= EXT_GAMES:
            inj[tm] = (row["passer_player_name"], out_w, in_w)

    # --- pass-catcher PPR points per player-week ---
    rc = p[p["complete_pass"] == 1].groupby(["receiver_player_id", "posteam", "week"]).agg(
        rec=("complete_pass", "sum"), ry=("yards_gained", "sum"), rtd=("pass_touchdown", "sum")).reset_index()
    ru = p[p["rush_attempt"] == 1].groupby(["rusher_player_id", "posteam", "week"]).agg(
        uy=("yards_gained", "sum"), utd=("rush_touchdown", "sum")).reset_index()
    rc = rc.rename(columns={"receiver_player_id": "pid"}); ru = ru.rename(columns={"rusher_player_id": "pid"})
    g = pd.merge(rc, ru, on=["pid", "posteam", "week"], how="outer").fillna(0)
    g["fp"] = g["rec"] + g["ry"] * 0.1 + g["rtd"] * 6 + g["uy"] * 0.1 + g["utd"] * 6

    # id -> name/pos map (players endpoint isn't year-gated)
    import nfl_data_py as nfl
    pl = nfl.import_players()[["gsis_id", "display_name", "position"]].dropna(subset=["gsis_id"])
    nm = dict(zip(pl["gsis_id"], pl["display_name"])); ps = dict(zip(pl["gsis_id"], pl["position"]))

    # 2026 projections/value
    P = json.loads(open(os.path.join(ROOT, "docs", "data.js"), encoding="utf-8").read().split("const PLAYERS = ")[1].rsplit(";", 1)[0])
    proj = {norm(x["name"]): x for x in P}

    rows = []
    for pid, sub in g.groupby("pid"):
        tm = sub["posteam"].mode().iat[0]
        if tm not in inj: continue
        qbname, out_w, in_w = inj[tm]
        ins = sub[sub["week"].isin(in_w)]; outs = sub[sub["week"].isin(out_w)]
        if len(ins) < MIN_GM or len(outs) < 2: continue
        ippg, oppg = ins["fp"].mean(), outs["fp"].mean()
        if ippg < MIN_IN or ippg <= 0 or oppg / ippg > DROP: continue
        pos = ps.get(pid, "?")
        if pos not in ("WR", "RB", "TE"): continue
        pr = proj.get(norm(nm.get(pid, "")))
        pj = round(pr.get("proj_ppg"), 1) if pr and pr.get("proj_ppg") else None
        if pj is None: continue
        rows.append({"name": nm.get(pid, pid), "pos": pos, "team": tm, "qb": qbname,
                     "in_ppg": round(ippg, 1), "out_ppg": round(oppg, 1), "drop": round((1 - oppg / ippg) * 100),
                     "gm": f"{len(ins)}/{len(outs)}", "proj26": pj, "gap": round(ippg - pj, 1)})
    rows.sort(key=lambda r: r["gap"], reverse=True)

    print(f"Pass-catchers who dropped when their starting QB missed >= {EXT_GAMES} games in 2025")
    print(f"(QB played the majority; in-PPG >= {MIN_IN}, dropped >= {round((1-DROP)*100)}%).")
    print("GAP = with-QB rate - 2026 projection (how much the projection may understate them if the QB is back).\n")
    print("  %-21s %-3s %-4s %-14s %-7s %6s %6s %5s %7s %5s" % ("player","pos","tm","starter QB","gm in/out","in","out","drop","proj26","GAP"))
    for r in rows:
        print("  %-21s %-3s %-4s %-14s %-7s %6.1f %6.1f %4d%% %7.1f %+5.1f"
              % (r["name"][:21], r["pos"], r["team"], r["qb"][:14], r["gm"], r["in_ppg"], r["out_ppg"], r["drop"], r["proj26"], r["gap"]))


if __name__ == "__main__":
    main()
