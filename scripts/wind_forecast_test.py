"""
THE DEPLOYABLE TEST: condition the wind-unders strategy on the FORECAST available ~24h
before kickoff (open-meteo previous-runs API: wind_speed_10m_previous_day1 = the value
that yesterday's model run predicted for each hour), not on actual wind.

If the edge survives forecast-conditioning, the market underreacts to public forecasts
(harvestable). If it dies, the actual-wind edge lived in forecast-miss games (not
harvestable). Coverage: previous-runs archive starts 2022. Also reports forecast skill
(MAE, band agreement) vs the actuals already in nflv_game_wind.
Appends to outputs/reports/sentiment_props.md.
"""
import os, json, sqlite3, time, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

STAD = {
 "Baltimore Ravens": (39.2780,-76.6227), "Buffalo Bills": (42.7738,-78.7870),
 "Carolina Panthers": (35.2258,-80.8528), "Chicago Bears": (41.8623,-87.6167),
 "Cincinnati Bengals": (39.0955,-84.5161), "Cleveland Browns": (41.5061,-81.6995),
 "Denver Broncos": (39.7439,-105.0201), "Green Bay Packers": (44.5013,-88.0622),
 "Jacksonville Jaguars": (30.3239,-81.6373), "Kansas City Chiefs": (39.0489,-94.4839),
 "Miami Dolphins": (25.9580,-80.2389), "New England Patriots": (42.0909,-71.2643),
 "New York Giants": (40.8128,-74.0742), "New York Jets": (40.8128,-74.0742),
 "Philadelphia Eagles": (39.9008,-75.1675), "Pittsburgh Steelers": (40.4468,-80.0158),
 "San Francisco 49ers": (37.4032,-121.9698), "Seattle Seahawks": (47.5952,-122.3316),
 "Tampa Bay Buccaneers": (27.9759,-82.5033), "Tennessee Titans": (36.1665,-86.7713),
 "Washington Commanders": (38.9077,-76.8645)}

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
g = pd.read_sql("""SELECT event_id, home_team, commence_time, season,
                          home_score+away_score AS total, completed FROM games""", con)
g = g[(g.completed == 1) & g.home_team.isin(STAD)].copy()
g["kick"] = pd.to_datetime(g.commence_time).dt.tz_localize(None)
g = g[g.kick >= "2022-03-01"]                                # previous-runs archive floor

if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_game_wind_fc'").fetchone():
    con.execute("CREATE TABLE nflv_game_wind_fc (event_id TEXT PRIMARY KEY, wind_fc_kn REAL)")
    n = 0
    for team, sub in g.groupby("home_team"):
        la, lo = STAD[team]
        for yr, ysub in sub.groupby(sub.kick.dt.year):
            d0 = ysub.kick.min().strftime("%Y-%m-%d")
            d1 = (ysub.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}"
                   f"&start_date={d0}&end_date={d1}&hourly=wind_speed_10m_previous_day1"
                   f"&windspeed_unit=kn&timezone=UTC")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        js = json.loads(r.read())
                    break
                except Exception:
                    if attempt == 2: js = None
                    time.sleep(4)
            if not js or "hourly" not in js: continue
            H = pd.DataFrame(js["hourly"]); H["t"] = pd.to_datetime(H.time)
            for _, gm in ysub.iterrows():
                w = H[(H.t >= gm.kick.floor("h")) & (H.t <= gm.kick.floor("h") + pd.Timedelta(hours=3))]
                v = w.wind_speed_10m_previous_day1.dropna()
                if not len(v): continue
                con.execute("INSERT OR REPLACE INTO nflv_game_wind_fc VALUES (?,?)",
                            (gm.event_id, float(v.mean())))
                n += 1
            con.commit(); time.sleep(0.4)
    print(f"fetched 24h-prior forecasts for {n} outdoor games (2022+)")

fc = pd.read_sql("SELECT * FROM nflv_game_wind_fc", con)
act = pd.read_sql("SELECT event_id, wind_kn FROM nflv_game_wind", con)
d = g.merge(fc, on="event_id").merge(act, on="event_id", how="left")

ok = d.dropna(subset=["wind_kn", "wind_fc_kn"])
mae = (ok.wind_kn - ok.wind_fc_kn).abs().mean()
band = lambda w: np.select([w < 8, w <= 15], ["calm", "moderate"], "high")
agree = (band(ok.wind_kn.values) == band(ok.wind_fc_kn.values)).mean()
print(f"forecast skill (n={len(ok)}): MAE {mae:.1f} kn | band agreement {agree:.0%}")

tot = pd.read_sql("""SELECT event_id, bookmaker, outcome_name, price, point, snapshot_time
                     FROM game_odds WHERE market='totals'""", con)
tot = tot.sort_values("snapshot_time").groupby(["event_id", "bookmaker", "outcome_name"], as_index=False).last()
ov = tot[tot.outcome_name == "Over"].rename(columns={"point": "line"})
un = tot[tot.outcome_name == "Under"].rename(columns={"price": "up", "point": "l2"})
t2 = ov.merge(un[["event_id", "bookmaker", "up", "l2"]], on=["event_id", "bookmaker"])
t2 = t2[t2.line == t2.l2]
t2["up"] = np.where(t2.up < 0, 1 + 100 / t2.up.abs(), 1 + t2.up / 100)
med = t2.groupby("event_id").line.median().rename("ml").reset_index()
t2 = t2.merge(med, on="event_id"); t2 = t2[t2.line == t2.ml]
cons = t2.groupby("event_id").agg(line=("line", "first"), ud=("up", "max")).reset_index()
d = d.merge(cons, on="event_id").dropna(subset=["total", "line", "wind_fc_kn"])
d = d[d.total != d.line]

print(f"\nTOTALS unders conditioned on the 24h-PRIOR FORECAST (2022-25, n={len(d)}):")
d["wb"] = pd.cut(d.wind_fc_kn, [-1, 8, 12, 15, 99], labels=["<8kn", "8-12", "12-15", "15+"])
for b, sub in d.groupby("wb", observed=True):
    undw = sub.total < sub.line
    roi = np.where(undw, sub.ud - 1, -1.0).mean()
    print(f"   fc {b:<6} n={len(sub):>4}  act-line {(sub.total-sub.line).mean():+.1f}  P(under) {undw.mean():.0%}  ROI {roi:+.1%}")
cell = d[d.wind_fc_kn.between(8, 15)]
print("\n   forecast 8-15kn cell by season:")
for s in sorted(cell.season.unique()):
    sub = cell[cell.season == s]
    roi = np.where(sub.total < sub.line, sub.ud - 1, -1.0).mean()
    print(f"     {s}: n={len(sub):>3}  ROI {roi:+.1%}")
# same games, ACTUAL-conditioned, for apples-to-apples
da = d.dropna(subset=["wind_kn"])
ca = da[da.wind_kn.between(8, 15)]
roi = np.where(ca.total < ca.line, ca.ud - 1, -1.0).mean()
print(f"\n   (same window, ACTUAL 8-15kn: n={len(ca)} ROI {roi:+.1%})")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Forecast-conditioned wind test (`wind_forecast_test.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
