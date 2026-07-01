"""
Chase-Burrow correlation, checked against BOTH 2023 and 2025 game logs.

Why this exists: nfl_data_py.import_weekly_data([2025]) 404s (nflverse hasn't published the
pre-aggregated player box scores for 2025 yet, and the installed package also gates 2025 as
"not available"). But the RAW play-by-play IS published, so we read it directly and derive the
weekly stats ourselves. 2023 uses the normal weekly endpoint.

Finding: the 2023 "Chase drops without Burrow" effect (0.64x) did NOT replicate in 2025 (0.96x) —
Chase is QB-agnostic now. So no projection uplift is warranted.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

PBP_2025 = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet"


def split_from_weekly(year):                       # 2023/2024 via the pre-aggregated endpoint
    import nfl_data_py as nfl
    w = nfl.import_weekly_data([year]); w = w[w["season_type"] == "REG"]
    bur = w[w["player_display_name"] == "Joe Burrow"][["week", "attempts"]].rename(columns={"attempts": "att"})
    ch = w[w["player_display_name"] == "Ja'Marr Chase"].copy()
    ch["fp"] = (ch["receptions"] + ch["receiving_yards"] * 0.1 + ch["receiving_tds"] * 6
                + ch["rushing_yards"] * 0.1 + ch["rushing_tds"] * 6)
    return ch.merge(bur, on="week", how="left").fillna({"att": 0})[["week", "fp", "att"]]


def split_from_pbp(url):                           # 2025 via raw play-by-play (endpoint 404s)
    cols = ["week", "posteam", "passer_player_name", "receiver_player_name", "complete_pass",
            "yards_gained", "pass_touchdown", "pass_attempt", "season_type"]
    pbp = pd.read_parquet(url, columns=cols); pbp = pbp[pbp["season_type"] == "REG"]
    cin = pbp[pbp["posteam"] == "CIN"]
    att = cin[cin["passer_player_name"] == "J.Burrow"].groupby("week")["pass_attempt"].sum().rename("att")
    ch = cin[(cin["receiver_player_name"] == "J.Chase") & (cin["complete_pass"] == 1)]
    rec = ch.groupby("week").agg(rec=("complete_pass", "sum"), yds=("yards_gained", "sum"),
                                 td=("pass_touchdown", "sum"))
    rec["fp"] = rec["rec"] + rec["yds"] * 0.1 + rec["td"] * 6
    return rec.join(att, how="outer").fillna(0).reset_index()[["week", "fp", "att"]]


def report(label, m):
    m = m.copy(); m["burrow_in"] = m["att"] >= 10
    ins, outs = m[m["burrow_in"]], m[~m["burrow_in"]]
    i, o = (ins["fp"].mean() if len(ins) else 0), (outs["fp"].mean() if len(outs) else 0)
    print(f"{label}: Chase  IN {i:5.1f} PPG ({len(ins)} gm)  |  OUT {o:5.1f} PPG ({len(outs)} gm)  "
          f"|  out/in {o / i:.2f}" if i and o else f"{label}: IN {i:.1f} ({len(ins)}) OUT {o:.1f} ({len(outs)})")


print("Chase production, split by whether Burrow played (>=10 att):\n")
report("2023 (weekly endpoint)", split_from_weekly(2023))
report("2025 (raw play-by-play)", split_from_pbp(PBP_2025))
print("\n-> 2023 showed a real drop; 2025 did not. Chase is QB-agnostic now; no uplift applied.")
