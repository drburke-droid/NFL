"""
GAME-TIME wind (open-meteo hourly archive, free) instead of GSOD daily averages.
Daily-mean wind under-classifies windy games (calm morning + 20kn evening averages to 10);
hourly lets us take the actual kickoff->+3h window. Expectation: many more games qualify
as windy; question: does the totals/passing-unders edge hold, grow, or vanish (books may
price GAME-TIME wind better than daily wind)?

Fetches per-game wind/gust/temp/precip into db nflv_game_wind (event_id keyed), then reruns:
  A. totals vs closing consensus by game-time wind bucket (outdoor)
  B. passing-prop unders by bucket (2023-25)
Appends to outputs/reports/sentiment_props.md.
"""
import os, json, re, sqlite3, time, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

from importlib.util import spec_from_file_location, module_from_spec
STAD = {  # same map as weather_props_study
 "Arizona Cardinals": (33.5276,-112.2626,1), "Atlanta Falcons": (33.7554,-84.4008,1),
 "Baltimore Ravens": (39.2780,-76.6227,0), "Buffalo Bills": (42.7738,-78.7870,0),
 "Carolina Panthers": (35.2258,-80.8528,0), "Chicago Bears": (41.8623,-87.6167,0),
 "Cincinnati Bengals": (39.0955,-84.5161,0), "Cleveland Browns": (41.5061,-81.6995,0),
 "Dallas Cowboys": (32.7473,-97.0945,1), "Denver Broncos": (39.7439,-105.0201,0),
 "Detroit Lions": (42.3400,-83.0456,1), "Green Bay Packers": (44.5013,-88.0622,0),
 "Houston Texans": (29.6847,-95.4107,1), "Indianapolis Colts": (39.7601,-86.1639,1),
 "Jacksonville Jaguars": (30.3239,-81.6373,0), "Kansas City Chiefs": (39.0489,-94.4839,0),
 "Las Vegas Raiders": (36.0909,-115.1833,1), "Los Angeles Chargers": (33.9535,-118.3392,1),
 "Los Angeles Rams": (33.9535,-118.3392,1), "Miami Dolphins": (25.9580,-80.2389,0),
 "Minnesota Vikings": (44.9736,-93.2575,1), "New England Patriots": (42.0909,-71.2643,0),
 "New Orleans Saints": (29.9511,-90.0812,1), "New York Giants": (40.8128,-74.0742,0),
 "New York Jets": (40.8128,-74.0742,0), "Philadelphia Eagles": (39.9008,-75.1675,0),
 "Pittsburgh Steelers": (40.4468,-80.0158,0), "San Francisco 49ers": (37.4032,-121.9698,0),
 "Seattle Seahawks": (47.5952,-122.3316,0), "Tampa Bay Buccaneers": (27.9759,-82.5033,0),
 "Tennessee Titans": (36.1665,-86.7713,0), "Washington Commanders": (38.9077,-76.8645,0)}

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
g = pd.read_sql("SELECT event_id, home_team, commence_time, season, week, home_score, away_score, completed FROM games", con)
g = g[g.completed == 1].copy()
g["kick"] = pd.to_datetime(g.commence_time).dt.tz_localize(None)
g["dome"] = g.home_team.map(lambda t: STAD.get(t, (0, 0, 1))[2])

if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_game_wind'").fetchone():
    con.execute("""CREATE TABLE nflv_game_wind (event_id TEXT PRIMARY KEY, wind_kn REAL,
                   gust_kn REAL, temp_c REAL, precip_mm REAL)""")
    outdoor = g[g.dome == 0]
    n_done = 0
    for team, sub in outdoor.groupby("home_team"):
        la, lo, _ = STAD[team]
        for yr, ysub in sub.groupby(sub.kick.dt.year):
            d0 = ysub.kick.min().strftime("%Y-%m-%d")
            d1 = (ysub.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}"
                   f"&start_date={d0}&end_date={d1}"
                   f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation"
                   f"&windspeed_unit=kn&timezone=UTC")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        js = json.loads(r.read())
                    break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(3)
            H = pd.DataFrame(js["hourly"]); H["t"] = pd.to_datetime(H.time)
            rows = []
            for _, gm in ysub.iterrows():
                w = H[(H.t >= gm.kick.floor("h")) & (H.t <= gm.kick.floor("h") + pd.Timedelta(hours=3))]
                if not len(w): continue
                rows.append((gm.event_id, float(w.wind_speed_10m.mean()), float(w.wind_gusts_10m.max()),
                             float(w.temperature_2m.mean()), float(w.precipitation.sum())))
            con.executemany("INSERT OR REPLACE INTO nflv_game_wind VALUES (?,?,?,?,?)", rows)
            con.commit(); n_done += len(rows)
            time.sleep(0.4)
    print(f"fetched game-time wind for {n_done} outdoor games")

gw = pd.read_sql("SELECT * FROM nflv_game_wind", con)
g = g.merge(gw, on="event_id", how="left")
g["total_pts"] = g.home_score + g.away_score

tot = pd.read_sql("""SELECT event_id, bookmaker, outcome_name, price, point, snapshot_time
                     FROM game_odds WHERE market='totals'""", con)
tot = tot.sort_values("snapshot_time").groupby(["event_id", "bookmaker", "outcome_name"], as_index=False).last()
ov = tot[tot.outcome_name == "Over"].rename(columns={"price": "over_p", "point": "line"})
un = tot[tot.outcome_name == "Under"].rename(columns={"price": "under_p", "point": "l2"})
t2 = ov.merge(un[["event_id", "bookmaker", "under_p", "l2"]], on=["event_id", "bookmaker"])
t2 = t2[t2.line == t2.l2]
for c in ("over_p", "under_p"):
    t2[c] = np.where(t2[c] < 0, 1 + 100 / t2[c].abs(), 1 + t2[c] / 100)
med = t2.groupby("event_id").line.median().rename("mline").reset_index()
t2 = t2.merge(med, on="event_id"); t2 = t2[t2.line == t2.mline]
cons = t2.groupby("event_id").agg(line=("line", "first"), under_dec=("under_p", "max")).reset_index()
out = g.merge(cons, on="event_id", how="inner")
out = out[(out.dome == 0) & out.wind_kn.notna()].dropna(subset=["total_pts", "line"])
print(f"\nA. TOTALS vs GAME-TIME wind — outdoor games: {len(out)}")
out["wb"] = pd.cut(out.wind_kn, [-1, 8, 12, 15, 99], labels=["<8kn", "8-12", "12-15", "15+"])
print(f"   {'wind':<8}{'n':>5}{'line':>7}{'actual':>8}{'act-line':>9}{'P(under)':>9}{'under ROI':>10}")
for b, sub in out.groupby("wb", observed=True):
    s2 = sub[sub.total_pts != sub.line]
    undw = s2.total_pts < s2.line
    roi = np.where(undw, s2.under_dec - 1, -1.0).mean()
    print(f"   {b:<8}{len(sub):>5}{sub.line.mean():>7.1f}{sub.total_pts.mean():>8.1f}"
          f"{(sub.total_pts-sub.line).mean():>+9.1f}{undw.mean():>9.0%}{roi:>10.1%}")
hi = out[out.gust_kn >= 25]
s2 = hi[hi.total_pts != hi.line]; undw = s2.total_pts < s2.line
print(f"   gust25+ {len(hi):>5}{hi.line.mean():>7.1f}{hi.total_pts.mean():>8.1f}"
      f"{(hi.total_pts-hi.line).mean():>+9.1f}{undw.mean():>9.0%}{np.where(undw, s2.under_dec-1, -1.0).mean():>10.1%}")

# B. passing props by game-time wind
nrm2 = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
pp = pd.read_sql("""SELECT p.event_id, p.bookmaker, p.market, p.player_name, p.outcome_type,
                           p.price, p.point, p.snapshot_time, g2.season, g2.week
                    FROM player_props p JOIN games g2 ON g2.event_id=p.event_id
                    WHERE p.market IN ('player_pass_yds','player_pass_completions')""", con)
pp["price"] = np.where(pp.price < 0, 1 + 100 / pp.price.abs(), 1 + pp.price / 100)
pp = pp.sort_values("snapshot_time").groupby(
    ["event_id", "market", "player_name", "bookmaker", "outcome_type"], as_index=False).last()
o = pp[pp.outcome_type == "Over"].rename(columns={"price": "over", "point": "pt"})
u = pp[pp.outcome_type == "Under"].rename(columns={"price": "under", "point": "pt2"})
m = o.merge(u[["event_id", "market", "player_name", "bookmaker", "under", "pt2"]],
            on=["event_id", "market", "player_name", "bookmaker"])
m = m[m.pt == m.pt2]
m = m.groupby(["event_id", "market", "player_name"]).agg(
    pt=("pt", "median"), under=("under", "max"), season=("season", "first"), week=("week", "first")).reset_index()
m = m.merge(g[["event_id", "wind_kn", "gust_kn", "dome"]], on="event_id", how="left")
wkst = pd.read_sql("""SELECT player_display_name nm, season, week, passing_yards, completions
                      FROM nflv_weekly WHERE season>=2023 AND season_type='REG'""", con)
wkst["nm"] = wkst.nm.map(nrm2)
m["nm"] = m.player_name.map(nrm2)
m["week"] = pd.to_numeric(m.week, errors="coerce"); m = m.dropna(subset=["week"]); m["week"] = m.week.astype(int)
m = m.merge(wkst, on=["nm", "season", "week"], how="left")
m["actual"] = np.where(m.market == "player_pass_yds", m.passing_yards, m.completions)
m = m.dropna(subset=["actual"]); m = m[m.actual != m.pt]
m["under_win"] = m.actual < m.pt
m2 = m[(m.dome == 0) & m.wind_kn.notna()].copy()
print(f"\nB. PASSING PROPS unders by GAME-TIME wind (outdoor 2023-25): n={len(m2)}")
m2["wb"] = pd.cut(m2.wind_kn, [-1, 8, 12, 15, 99], labels=["<8kn", "8-12", "12-15", "15+"])
for b, sub in m2.groupby("wb", observed=True):
    roi = np.where(sub.under_win, sub.under - 1, -1.0).mean()
    print(f"   wind {b:<6} n={len(sub):>5}  P(under) {sub.under_win.mean():.0%}  under ROI {roi:+.1%}")
hi = m2[m2.gust_kn >= 25]
roi = np.where(hi.under_win, hi.under - 1, -1.0).mean()
print(f"   gust25+     n={len(hi):>5}  P(under) {hi.under_win.mean():.0%}  under ROI {roi:+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## GAME-TIME hourly wind (`hourly_wind_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
