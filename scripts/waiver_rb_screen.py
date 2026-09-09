"""Weekly waiver-wire RB screen for the current season — the rule from waiver_rb_study.py.

Two modes:
  * in-season (after week 1 box scores + snap counts post on nflverse, usually Tuesday):
    every RB outside the preseason FFA top-36, scored by the screen — snap share first,
    then touches, draft capital and age — with the historical hit rate of each tier.
  * pre-week-1 (nothing played yet): an OPPORTUNITY watchlist — available RBs whose team
    RB1 is Out/IR right now (ESPN + Sleeper) or whom the FFA weekly file already projects
    for a real role (>= 8 PPR), so you know who to grab before the box scores exist.

Usage: python scripts/waiver_rb_screen.py [--season 2026] [--week N] [--mine "Name,Name"]
   --mine  names already on your roster to exclude (optional)
Tiers (2013-25 back-test excl. 2014-15, RBs outside preseason top-36, base rate 6% RB2+):
                                          after wk1            after wk2            after wk3
   A  snap >= 60%                         20% RB2+ / 28% flex  32% / 42%            33% / 44%
   B  snap >= 50% & drafted rd 1-3        33% / 50% (n=12)     31% / 54%            31% / 38%
   C  snap >= 50%                         19% / 33%            21% / 35%            22% / 33%
   D  touches/g >= 15                     21% / 33%            32% / 42%            31% / 49%
   E  RB1 absent & carry share >= 40%      0% / 0%  (n=6)      15% / 31%            14% / 21%
   Snap share and touches are the signal; TDs, YPC, EPA, age and preseason rank add nothing.
   Injury fill-ins (E) are the WEAKEST tier early — the starter usually comes back.
"""
import argparse, json, os, re, sqlite3, urllib.request, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "WSH": "WAS"}
ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--week", type=int, default=None)
ap.add_argument("--mine", default="")
A = ap.parse_args()
S = A.season
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
mine = {norm(x) for x in A.mine.split(",") if x.strip()}

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
ffa = pd.read_sql(f"SELECT player_id, player, ffa_pos_rank FROM nflv_ffa_proj WHERE position='RB' AND season={S}", con)
ros = pd.read_sql(f"""SELECT gsis_id player_id, pfr_id, full_name, team, draft_number, birth_date
                      FROM nflv_rosters{'_2026' if S == 2026 else ''} WHERE position='RB' AND season={S}""", con) \
    if S != 2026 else pd.read_sql("""SELECT r.player_id, r.team, s.full_name, s.pfr_id, s.draft_number, s.birth_date
                      FROM nflv_rosters_2026 r LEFT JOIN (SELECT gsis_id, full_name, pfr_id, draft_number, birth_date
                      FROM nflv_rosters WHERE season=2025 AND position='RB') s ON s.gsis_id=r.player_id
                      WHERE r.position='RB'""", con)
con.close()
ros["team"] = ros.team.map(lambda t: FIX.get(t, t))
ros = ros.merge(ffa[["player_id", "ffa_pos_rank"]], on="player_id", how="left")
ros["pre_rank"] = ros.ffa_pos_rank.fillna(99)
ros["draft_round"] = np.where(pd.to_numeric(ros.draft_number, errors="coerce").isna(), 8,
                              np.ceil(pd.to_numeric(ros.draft_number, errors="coerce").fillna(0) / 32))
ros["age"] = S - pd.to_datetime(ros.birth_date, errors="coerce").dt.year
ros["nname"] = ros.full_name.fillna("").map(norm)
avail = ros[(ros.pre_rank > 36) & ~ros.nname.isin(mine)].copy()
print(f"{S}: {len(ros)} rostered RBs, {len(avail)} outside the preseason top-36 (available pool)")

# ---- in-season box scores + snaps from nflverse ----
def nflverse(path):
    return pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/" + path)
wk = None
for url in (f"player_stats/player_stats_{S}.parquet", f"player_stats/stats_player_week_{S}.parquet"):
    try:
        w = nflverse(url); w = w[(w.position == "RB") & (w.season_type == "REG")] if "season_type" in w.columns else w[w.position == "RB"]
        wk = w; break
    except Exception as e: err = str(e)[:50]
if wk is not None and len(wk):
    try:
        sn = nflverse(f"snap_counts/snap_counts_{S}.parquet"); sn = sn[sn.position == "RB"][["week", "pfr_player_id", "offense_pct"]]
    except Exception as e:
        print("  snap counts not available yet:", str(e)[:50]); sn = pd.DataFrame(columns=["week", "pfr_player_id", "offense_pct"])
    cur_week = A.week or int(wk.week.max())
    c = wk[wk.week <= cur_week].copy()
    teamcar = nflverse(f"player_stats/player_stats_{S}.parquet") if False else c   # team carries from RB rows is enough here
    tc = c.groupby(["week", "recent_team" if "recent_team" in c.columns else "team"]).carries.sum().rename("team_car").reset_index()
    tcol = "recent_team" if "recent_team" in c.columns else "team"
    c = c.merge(tc, on=["week", tcol], how="left"); c["car_share"] = c.carries / c.team_car.replace(0, np.nan)
    c = c.merge(ros[["player_id", "pfr_id"]], on="player_id", how="left")
    c = c.merge(sn.rename(columns={"pfr_player_id": "pfr_id"}), on=["week", "pfr_id"], how="left")
    g = c.groupby("player_id")
    f = pd.DataFrame({"snap_pct": g.offense_pct.mean(), "car_share": g.car_share.mean(),
                      "touches_g": (g.carries.mean() + g.receptions.mean()), "targets_g": g.targets.mean(),
                      "ppg": g.fantasy_points_ppr.mean(), "games": g.week.size()}).reset_index()
    f = avail.merge(f, on="player_id", how="inner")
    def tier(r):
        if pd.notna(r.snap_pct) and r.snap_pct >= 0.6: return "A"
        if pd.notna(r.snap_pct) and r.snap_pct >= 0.5 and r.draft_round <= 3: return "B"
        if pd.notna(r.snap_pct) and r.snap_pct >= 0.5: return "C"
        if r.touches_g >= 15: return "D"
        if r.car_share >= 0.4: return "E"
        return ""
    f["tier"] = f.apply(tier, axis=1)
    out = f[f.tier != ""].sort_values(["tier", "snap_pct"], ascending=[True, False])
    print(f"\nAfter week {cur_week} — available RBs that trip the screen (A best):\n")
    print(out[["tier", "full_name", "team", "snap_pct", "car_share", "touches_g", "targets_g", "ppg", "draft_round", "age", "pre_rank"]]
          .rename(columns={"full_name": "player"}).round(2).to_string(index=False))
    out.to_csv(os.path.join(ROOT, "outputs", f"waiver_rb_screen_{S}_wk{cur_week}.csv"), index=False)
else:
    # ---- pre-week-1: opportunity watchlist ----
    print(f"  no {S} box scores yet — opportunity watchlist instead")
    def http_json(u):
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
            return json.loads(r.read().decode())
    NAME2ABBR = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
    OUTW = {"out", "injured reserve", "ir", "pup", "suspension", "doubtful"}
    st = {}
    try:
        for t in http_json("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries").get("injuries", []):
            for i in t.get("injuries", []):
                if i.get("athlete", {}).get("position", {}).get("abbreviation") == "RB":
                    st[(norm(i["athlete"]["displayName"]), NAME2ABBR.get(t["displayName"], ""))] = str(i.get("status", "")).lower()
    except Exception as e: print("  ESPN unavailable", str(e)[:40])
    # FFA weekly wk1 projection = the role FFA already expects
    wf = None
    p1 = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", f"raw_stats_{S}_wk1.csv")
    if os.path.exists(p1):
        d = pd.read_csv(p1, na_values=["NA"]); d = d[d.position == "RB"]
        z = lambda c: d[c].fillna(0) if c in d.columns else 0
        d["wk1_ppr"] = z("rush_yds") * .1 + z("rush_tds") * 6 + z("rec_yds") * .1 + z("rec_tds") * 6 + z("rec")
        d["nname"] = d.player.map(norm); d["team"] = d.team.map(lambda t: FIX.get(t, t))
        wf = d[["nname", "team", "wk1_ppr", "injury_status"]]
    # team RB1 by preseason rank, and whether he is out
    rb1 = ros.sort_values("pre_rank").drop_duplicates("team")[["team", "full_name", "nname", "pre_rank"]].rename(
        columns={"full_name": "rb1", "nname": "rb1_n", "pre_rank": "rb1_rank"})
    rb1["rb1_status"] = [st.get((n, t), "") for n, t in zip(rb1.rb1_n, rb1.team)]
    a = avail.merge(rb1[["team", "rb1", "rb1_rank", "rb1_status"]], on="team", how="left")
    if wf is not None: a = a.merge(wf, on=["nname", "team"], how="left")
    a["own_status"] = [st.get((n, t), "") for n, t in zip(a.nname, a.team)]
    a = a[~a.own_status.isin(OUTW)]
    a["why"] = np.where(a.rb1_status.isin(OUTW), "team RB1 " + a.rb1.fillna("") + " is " + a.rb1_status, "")
    a.loc[(a.get("wk1_ppr", pd.Series(0, index=a.index)).fillna(0) >= 8), "why"] = a.why + np.where(a.why != "", "; ", "") + "FFA projects a real wk1 role"
    a = a[(a.why != "") & a.full_name.notna()]
    if "wk1_ppr" in a.columns:   # RB1-out teams: keep the two best-projected replacements only
        a = a.sort_values("wk1_ppr", ascending=False)
        a["rk"] = a.groupby("team").cumcount()
        a = a[(a.rk < 2) | a.why.str.contains("FFA projects")]
    cols = ["full_name", "team", "age", "draft_round", "pre_rank"] + (["wk1_ppr"] if "wk1_ppr" in a.columns else []) + ["why"]
    print(f"\nPre-week-1 opportunity watchlist (available RBs, {len(a)}):\n")
    print(a[cols].rename(columns={"full_name": "player"}).round(1).to_string(index=False))
    a.to_csv(os.path.join(ROOT, "outputs", f"waiver_rb_watchlist_{S}_pre_wk1.csv"), index=False)
