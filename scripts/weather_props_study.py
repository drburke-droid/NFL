"""
Game-day weather (NOAA GSOD via BigQuery) vs betting markets, 2020-2025.

Stadium -> nearest GSOD station -> daily wind/gust/temp/precip on game date (UTC kickoff
shifted -6h to local date). Domes/retractables excluded from wind tests.

Tests:
  A. TOTALS: is wind fully priced? actual total vs closing consensus total by wind bucket,
     and under-bet ROI at closing prices by bucket.
  B. PASSING PROPS (2023-25): pass-yds / completions unders in wind >= 15 kn.
Writes db nflv_weather (station-day per stadium); appends to outputs/reports/sentiment_props.md.
"""
import os, json, re, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

# home team -> (lat, lon, dome?)  dome=1 covers fixed roofs + retractables (usually closed)
STAD = {
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
if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_weather'").fetchone():
    from google.cloud import bigquery
    import pydata_google_auth
    creds = pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"])
    bq = bigquery.Client(project=json.load(open(os.path.join(ROOT, "config", "gdelt_bq.json")))["project"],
                         credentials=creds)
    # stations metadata is stale (max end 2021) but daily tables are current: take the
    # top-3 nearest per stadium and use whichever reports on a given date
    st = [dict(r) for r in bq.query("""
        SELECT usaf, wban, lat, lon FROM `bigquery-public-data.noaa_gsod.stations`
        WHERE country='US' AND lat IS NOT NULL AND ABS(lat) > 1
          AND CAST(`end` AS INT64) >= 20200101""").result()]
    st = pd.DataFrame(st)
    near = {}
    for team, (la, lo, dome) in STAD.items():
        dist = ((st.lat - la) ** 2 + ((st.lon - lo) * 0.77) ** 2) ** 0.5
        near[team] = [(st.loc[i].usaf, st.loc[i].wban) for i in dist.nsmallest(3).index]
    pairs = sorted(set(p for v in near.values() for p in v))
    cond = " OR ".join(f"(stn='{u}' AND wban='{w}')" for u, w in pairs)
    rows = []
    for yr in range(2020, 2026):
        q = f"""SELECT stn, wban, date, wdsp, gust, max AS tmax, min AS tmin, prcp, snow_ice_pellets
                FROM `bigquery-public-data.noaa_gsod.gsod{yr}` WHERE {cond}"""
        rows += [dict(r) for r in bq.query(q).result()]
    W = pd.DataFrame(rows)
    con.execute("""CREATE TABLE nflv_weather (team TEXT, date TEXT, wdsp REAL, gust REAL,
                   tmax REAL, tmin REAL, prcp REAL, snow INTEGER, PRIMARY KEY(team,date))""")
    ins = []
    for team, cands in near.items():
        seen = set()
        for u, w in cands:                                   # nearest first; skip dates already filled
            sub = W[(W.stn == u) & (W.wban == w)]
            for _, r in sub.iterrows():
                dt = str(r.date)
                if dt in seen: continue
                seen.add(dt)
                wd = None if r.wdsp in (999.9,) else r.wdsp
                gu = None if r.gust in (999.9,) else r.gust
                pc = None if r.prcp in (99.99,) else r.prcp
                ins.append((team, dt, wd, gu, r.tmax, r.tmin, pc, int(r.snow_ice_pellets or 0)))
    con.executemany("INSERT OR REPLACE INTO nflv_weather VALUES (?,?,?,?,?,?,?,?)", ins)
    con.commit()
    print(f"fetched weather: {len(ins)} station-days for {len(near)} stadiums (scan was tiny)")

g = pd.read_sql("SELECT event_id, home_team, away_team, commence_time, season, home_score, away_score, completed FROM games", con)
g = g[g.completed == 1].copy()
g["date"] = (pd.to_datetime(g.commence_time).dt.tz_localize(None) - pd.Timedelta(hours=6)).dt.date.astype(str)
g["dome"] = g.home_team.map(lambda t: STAD.get(t, (0, 0, 1))[2])
w = pd.read_sql("SELECT * FROM nflv_weather", con)
g = g.merge(w, left_on=["home_team", "date"], right_on=["team", "date"], how="left")
g["total_pts"] = g.home_score + g.away_score

# closing consensus total per game
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
t2 = t2.merge(med, on="event_id"); t2 = t2[t2.line == t2.mline]      # best price AT the consensus line only
cons = t2.groupby("event_id").agg(line=("line", "first"), under_dec=("under_p", "max"), over_dec=("over_p", "max")).reset_index()
g = g.merge(cons, on="event_id", how="inner").dropna(subset=["total_pts", "line"])
out = g[(g.dome == 0) & g.wdsp.notna()].copy()
print(f"\nA. TOTALS vs wind — outdoor games with closing total + weather: {len(out)}")
out["wb"] = pd.cut(out.wdsp, [-1, 8, 12, 15, 99], labels=["calm <8kn", "8-12", "12-15", "15+ kn"])
print(f"   {'wind':<12}{'n':>5}{'total line':>11}{'actual':>8}{'actual-line':>12}{'P(under)':>9}{'under ROI':>10}")
for b, sub in out.groupby("wb", observed=True):
    sub2 = sub[sub.total_pts != sub.line]
    undw = (sub2.total_pts < sub2.line)
    roi = np.where(undw, sub2.under_dec - 1, -1.0).mean()
    print(f"   {b:<12}{len(sub):>5}{sub.line.mean():>11.1f}{sub.total_pts.mean():>8.1f}"
          f"{(sub.total_pts-sub.line).mean():>+12.1f}{undw.mean():>9.0%}{roi:>10.1%}")
dome = g[g.dome == 1]
d2 = dome[dome.total_pts != dome.line]
droi = np.where(d2.total_pts > d2.line, d2.over_dec - 1, -1.0).mean()
print(f"   {'domes':<12}{len(dome):>5}{dome.line.mean():>11.1f}{dome.total_pts.mean():>8.1f}"
      f"{(dome.total_pts-dome.line).mean():>+12.1f}   P(over) {(d2.total_pts>d2.line).mean():.0%}  over ROI {droi:+.1%}")

# B. passing props in wind (2023-25)
pp = pd.read_sql("""SELECT p.event_id, p.bookmaker, p.market, p.player_name, p.outcome_type,
                           p.price, p.point, p.snapshot_time, g2.season, g2.week, g2.home_team
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
    pt=("pt", "median"), under=("under", "max"), season=("season", "first"),
    week=("week", "first"), home_team=("home_team", "first")).reset_index()
m = m.merge(g[["event_id", "wdsp", "dome"]], on="event_id", how="left")
nrm2 = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
wkst = pd.read_sql("""SELECT player_display_name nm, season, week, passing_yards, completions
                      FROM nflv_weekly WHERE season>=2023 AND season_type='REG'""", con)
wkst["nm"] = wkst.nm.map(nrm2)
m["nm"] = m.player_name.map(nrm2)
m["week"] = pd.to_numeric(m.week, errors="coerce"); m = m.dropna(subset=["week"])
m["week"] = m.week.astype(int)
m = m.merge(wkst, on=["nm", "season", "week"], how="left")
m["actual"] = np.where(m.market == "player_pass_yds", m.passing_yards, m.completions)
m = m.dropna(subset=["actual"]); m = m[m.actual != m.pt]
m["under_win"] = m.actual < m.pt
print(f"\nB. PASSING PROPS unders by wind (outdoor, 2023-25): n={len(m[m.dome==0])}")
m2 = m[(m.dome == 0) & m.wdsp.notna()].copy()
m2["wb"] = pd.cut(m2.wdsp, [-1, 8, 12, 15, 99], labels=["<8kn", "8-12", "12-15", "15+"])
for b, sub in m2.groupby("wb", observed=True):
    roi = np.where(sub.under_win, sub.under - 1, -1.0).mean()
    print(f"   wind {b:<6} n={len(sub):>5}  P(under) {sub.under_win.mean():.0%}  under ROI {roi:+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Weather vs totals & passing props (`weather_props_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
