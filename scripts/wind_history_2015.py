"""
OUT-OF-SAMPLE replication of the moderate-wind under signal on 2015-2019 (and continuity
check 2020-25) using the nflverse schedule file, which carries Vegas closing TOTAL LINES
(no prices) back past 2015. Game-time wind = open-meteo hourly archive, kickoff->+3h,
kickoff converted from Eastern (nflverse gametime) to UTC. Outdoor games only (nflverse
roof column), home games only (excludes London/Mexico neutral sites).

No prices pre-2020, so the metric is under WIN RATE vs the 52.4% breakeven at -110,
plus actual-line. Wind cached in db nflv_game_wind_hist keyed by nflverse game_id.
Appends to outputs/reports/sentiment_props.md.
"""
import io, os, json, sqlite3, time, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

# coords by franchise abbr, with era overrides for moved teams
COORD = {
 "BAL": (39.2780,-76.6227), "BUF": (42.7738,-78.7870), "CAR": (35.2258,-80.8528),
 "CHI": (41.8623,-87.6167), "CIN": (39.0955,-84.5161), "CLE": (41.5061,-81.6995),
 "DEN": (39.7439,-105.0201), "GB": (44.5013,-88.0622), "JAX": (30.3239,-81.6373),
 "KC": (39.0489,-94.4839), "MIA": (25.9580,-80.2389), "NE": (42.0909,-71.2643),
 "NYG": (40.8128,-74.0742), "NYJ": (40.8128,-74.0742), "PHI": (39.9008,-75.1675),
 "PIT": (40.4468,-80.0158), "SF": (37.4032,-121.9698), "SEA": (47.5952,-122.3316),
 "TB": (27.9759,-82.5033), "TEN": (36.1665,-86.7713), "WAS": (38.9077,-76.8645),
 "MIN": (44.9765,-93.2247),          # TCF Bank 2015 (2016+ dome, roof filter handles)
 "SD": (32.7831,-117.1196),          # Qualcomm
 "OAK": (37.7516,-122.2005),         # Coliseum
 "LAC": (33.8644,-118.2611),         # Carson 2017-19 (SoFi 2020+ is dome)
 "LA": (34.0141,-118.2879),          # LA Coliseum 2016-19 (SoFi dome after)
 "CIN2": None}

def kickoff_utc(row):
    hh, mm = str(row.gametime).split(":")[:2]
    t = pd.Timestamp(f"{row.gameday} {hh}:{mm}")
    try:
        from zoneinfo import ZoneInfo
        return t.tz_localize(ZoneInfo("America/New_York")).tz_convert("UTC").tz_localize(None)
    except Exception:                               # no tzdata on this box: manual EDT/EST
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
print(f"nflverse outdoor home games with total_line 2015-2025: {len(sched)}")

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_game_wind_hist'").fetchone():
    con.execute("CREATE TABLE nflv_game_wind_hist (game_id TEXT PRIMARY KEY, wind_kn REAL, gust_kn REAL)")
    n = 0
    for (team, yr), sub in sched.groupby(["home_team", sched.kick.dt.year]):
        la, lo = COORD[team]
        d0 = sub.kick.min().strftime("%Y-%m-%d")
        d1 = (sub.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        u = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}"
             f"&start_date={d0}&end_date={d1}&hourly=wind_speed_10m,wind_gusts_10m"
             f"&windspeed_unit=kn&timezone=UTC")
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
            rows.append((gm.game_id, float(w.wind_speed_10m.mean()), float(w.wind_gusts_10m.max())))
        con.executemany("INSERT OR REPLACE INTO nflv_game_wind_hist VALUES (?,?,?)", rows)
        con.commit(); n += len(rows)
        time.sleep(0.35)
    print(f"fetched game-time wind for {n} games")

gw = pd.read_sql("SELECT * FROM nflv_game_wind_hist", con)
d = sched.merge(gw, on="game_id").dropna(subset=["wind_kn"])
d = d[d.total != d.total_line]
d["under"] = d.total < d.total_line
d["wb"] = pd.cut(d.wind_kn, [-1, 8, 12, 15, 99], labels=["<8kn", "8-12", "12-15", "15+"])

for era, sub in (("2015-2019 (OUT-OF-SAMPLE)", d[d.season <= 2019]),
                 ("2020-2025 (continuity chk)", d[d.season >= 2020])):
    print(f"\n{era}: n={len(sub)}  [breakeven at -110 = 52.4%]")
    print(f"   {'wind':<8}{'n':>6}{'line':>7}{'actual':>8}{'act-line':>9}{'P(under)':>9}")
    for b, s in sub.groupby("wb", observed=True):
        print(f"   {b:<8}{len(s):>6}{s.total_line.mean():>7.1f}{s.total.mean():>8.1f}"
              f"{(s.total-s.total_line).mean():>+9.1f}{s.under.mean():>9.1%}")
    hi = sub[sub.gust_kn >= 25]
    print(f"   gust25+ {len(hi):>6}{hi.total_line.mean():>7.1f}{hi.total.mean():>8.1f}"
          f"{(hi.total-hi.total_line).mean():>+9.1f}{hi.under.mean():>9.1%}")

cell = d[d.wind_kn.between(8, 15)]
print(f"\n8-15kn cell by season (n / act-line / P(under)):")
for s in sorted(cell.season.unique()):
    ss = cell[cell.season == s]
    print(f"   {s}: n={len(ss):>3}  {(ss.total-ss.total_line).mean():+.1f}  {ss.under.mean():.1%}")
b = cell[cell.season <= 2019]
z = (b.under.mean() - 0.5) / np.sqrt(0.25 / len(b))
print(f"\n2015-19 pooled 8-15kn: n={len(b)}  act-line {(b.total-b.total_line).mean():+.1f}"
      f"  P(under) {b.under.mean():.1%}  (z vs coin flip: {z:+.1f})")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Wind 2015-2019 out-of-sample (`wind_history_2015.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
