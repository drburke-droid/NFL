"""
Beyond wind: do temp, precipitation, humidity, snow (solo or combined with wind) move
totals vs the closing line? Hourly game-window (kickoff->+3h) values from open-meteo
for all outdoor home games 2015-2025 (nflverse schedule + total_line), cached in
nflv_game_wx_hist. Wind joined from nflv_game_wind_hist.

Design guards:
 - every factor is ALSO tested within calm (<8kn) games, since storms are windy - a
   naive "rain effect" can be pure wind confounding
 - win rate vs 52.4% breakeven for 2015-25; real-price under ROI 2020-25 for anything
   that looks alive (via nflv_game_wind temp/precip, event_id-keyed to game_odds)
Appends to outputs/reports/sentiment_props.md.
"""
import io, os, json, sqlite3, time, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

COORD = {
 "BAL": (39.2780,-76.6227), "BUF": (42.7738,-78.7870), "CAR": (35.2258,-80.8528),
 "CHI": (41.8623,-87.6167), "CIN": (39.0955,-84.5161), "CLE": (41.5061,-81.6995),
 "DEN": (39.7439,-105.0201), "GB": (44.5013,-88.0622), "JAX": (30.3239,-81.6373),
 "KC": (39.0489,-94.4839), "MIA": (25.9580,-80.2389), "NE": (42.0909,-71.2643),
 "NYG": (40.8128,-74.0742), "NYJ": (40.8128,-74.0742), "PHI": (39.9008,-75.1675),
 "PIT": (40.4468,-80.0158), "SF": (37.4032,-121.9698), "SEA": (47.5952,-122.3316),
 "TB": (27.9759,-82.5033), "TEN": (36.1665,-86.7713), "WAS": (38.9077,-76.8645),
 "MIN": (44.9765,-93.2247), "SD": (32.7831,-117.1196), "OAK": (37.7516,-122.2005),
 "LAC": (33.8644,-118.2611), "LA": (34.0141,-118.2879)}

def kickoff_utc(row):
    hh, mm = str(row.gametime).split(":")[:2]
    t = pd.Timestamp(f"{row.gameday} {hh}:{mm}")
    try:
        from zoneinfo import ZoneInfo
        return t.tz_localize(ZoneInfo("America/New_York")).tz_convert("UTC").tz_localize(None)
    except Exception:
        off = 4 if t.month in (9, 10) or (t.month == 11 and t.day < 7 and t.weekday() != 6) else 5
        return t + pd.Timedelta(hours=off)

for url in ("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
            "http://www.habitatring.com/games.csv"):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            sched = pd.read_csv(io.BytesIO(r.read()), low_memory=False)
        break
    except Exception as e:
        err = e
else:
    raise err
sched = sched[(sched.season >= 2015) & (sched.season <= 2025)]
sched = sched[sched.roof.isin(["outdoors", "open"]) & (sched.location == "Home")]
sched = sched.dropna(subset=["total_line", "home_score", "away_score", "gametime"]).copy()
sched = sched[sched.home_team.isin(COORD)]
sched["kick"] = sched.apply(kickoff_utc, axis=1)
sched["total"] = sched.home_score + sched.away_score

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_game_wx_hist'").fetchone():
    con.execute("""CREATE TABLE nflv_game_wx_hist (game_id TEXT PRIMARY KEY, temp_c REAL,
                   precip_mm REAL, rh REAL, snow_cm REAL)""")
    n = 0
    for (team, yr), sub in sched.groupby(["home_team", sched.kick.dt.year]):
        la, lo = COORD[team]
        d0 = sub.kick.min().strftime("%Y-%m-%d")
        d1 = (sub.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        u = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}"
             f"&start_date={d0}&end_date={d1}"
             f"&hourly=temperature_2m,precipitation,relative_humidity_2m,snowfall&timezone=UTC")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(u, timeout=60) as r:
                    js = json.loads(r.read())
                break
            except Exception:
                if attempt == 3: raise
                time.sleep(5)
        H = pd.DataFrame(js["hourly"]); H["t"] = pd.to_datetime(H.time)
        rows = []
        for _, gm in sub.iterrows():
            w = H[(H.t >= gm.kick.floor("h")) & (H.t <= gm.kick.floor("h") + pd.Timedelta(hours=3))]
            if not len(w): continue
            rows.append((gm.game_id, float(w.temperature_2m.mean()), float(w.precipitation.sum()),
                         float(w.relative_humidity_2m.mean()), float(w.snowfall.sum())))
        con.executemany("INSERT OR REPLACE INTO nflv_game_wx_hist VALUES (?,?,?,?,?)", rows)
        con.commit(); n += len(rows)
        time.sleep(0.35)
    print(f"fetched temp/precip/humidity/snow for {n} games")

wx = pd.read_sql("SELECT * FROM nflv_game_wx_hist", con)
wd = pd.read_sql("SELECT * FROM nflv_game_wind_hist", con)
d = sched.merge(wx, on="game_id").merge(wd, on="game_id").dropna(subset=["wind_kn", "temp_c"])
d = d[d.total != d.total_line]
d["under"] = d.total < d.total_line
d["calm"] = d.wind_kn < 8
d["windy"] = d.wind_kn.between(8, 15)

def line(lab, s):
    if len(s) < 15:
        print(f"   {lab:<28} n={len(s):>4}  (too small)"); return
    print(f"   {lab:<28} n={len(s):>4}  act-line {(s.total-s.total_line).mean():+5.1f}  P(under) {s.under.mean():.1%}")

print(f"\n2015-2025 outdoor, n={len(d)} [breakeven 52.4%] - overall P(under) {d.under.mean():.1%}")
print("\nSOLO factors (all games / calm-only to strip wind confound):")
F = [("freezing (temp<=0C)",   d.temp_c <= 0),
     ("cold (0-5C)",           d.temp_c.between(0, 5)),
     ("hot (>27C)",            d.temp_c > 27),
     ("rain 1-5mm",            d.precip_mm.between(1, 5)),
     ("heavy rain 5mm+",       d.precip_mm >= 5),
     ("snow (>0.5cm)",         d.snow_cm > 0.5),
     ("humid (rh>85%)",        d.rh > 85),
     ("dry air (rh<40%)",      d.rh < 40)]
for lab, m in F:
    line(lab, d[m]); line("   ... calm games only", d[m & d.calm])
print("\nCOMBINED with moderate wind (8-15kn, base 58.3%/56.6% by era):")
line("windy alone (dry, >5C)", d[d.windy & (d.precip_mm < 1) & (d.temp_c > 5)])
line("windy + rain 1mm+",      d[d.windy & (d.precip_mm >= 1)])
line("windy + freezing",       d[d.windy & (d.temp_c <= 0)])
line("windy + hi gust (25+)",  d[d.windy & (d.gust_kn >= 25)])

# real-price ROI 2020-25 for the live-looking cells (event_id-keyed data)
g2 = pd.read_sql("SELECT event_id, season, home_score+away_score total FROM games WHERE completed=1", con)
gw = pd.read_sql("SELECT * FROM nflv_game_wind", con)
tot = pd.read_sql("""SELECT event_id, bookmaker, outcome_name, price, point, snapshot_time
                     FROM game_odds WHERE market='totals'""", con)
tot = tot.sort_values("snapshot_time").groupby(["event_id","bookmaker","outcome_name"], as_index=False).last()
ov = tot[tot.outcome_name=="Over"].rename(columns={"point":"line"})
un = tot[tot.outcome_name=="Under"].rename(columns={"price":"up","point":"l2"})
t2 = ov.merge(un[["event_id","bookmaker","up","l2"]], on=["event_id","bookmaker"])
t2 = t2[t2.line==t2.l2]
t2["up"] = np.where(t2.up<0, 1+100/t2.up.abs(), 1+t2.up/100)
med = t2.groupby("event_id").line.median().rename("ml").reset_index()
t2 = t2.merge(med, on="event_id"); t2 = t2[t2.line==t2.ml]
cons = t2.groupby("event_id").agg(line=("line","first"), ud=("up","max")).reset_index()
e = g2.merge(gw, on="event_id").merge(cons, on="event_id").dropna(subset=["total","line","wind_kn"])
e = e[e.total != e.line]; e["under"] = e.total < e.line
print("\nreal-price under ROI 2020-25 (best consensus-line price):")
for lab, m in [("rain 1mm+ (any wind)", e.precip_mm >= 1),
               ("rain 1mm+, calm",      (e.precip_mm >= 1) & (e.wind_kn < 8)),
               ("freezing, calm",       (e.temp_c <= 0) & (e.wind_kn < 8)),
               ("snowish (precip & <=1C)", (e.precip_mm >= 0.5) & (e.temp_c <= 1)),
               ("wind 8-15 + rain",     e.wind_kn.between(8, 15) & (e.precip_mm >= 1)),
               ("wind 8-15, dry",       e.wind_kn.between(8, 15) & (e.precip_mm < 1))]:
    s = e[m]
    if len(s) < 10:
        print(f"   {lab:<28} n={len(s):>4}  (too small)"); continue
    roi = np.where(s.under, s.ud - 1, -1.0).mean()
    print(f"   {lab:<28} n={len(s):>4}  P(under) {s.under.mean():.1%}  ROI {roi:+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Other weather factors (`weather_effects_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
