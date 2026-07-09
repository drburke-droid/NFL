"""
Forecast-conditioning gate for the RAIN signal (same bar wind had to clear): bucket
2024-25 games by the 24h-prior PRECIP forecast for kickoff->+3h (open-meteo previous-runs,
precipitation_previous_day1) and check the under edge at real prices. Rain forecasts are
noisier than wind (timing within a 4h window), so this can genuinely fail.
Caches to nflv_game_rain_fc. Appends to outputs/reports/sentiment_props.md.
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
                          home_score+away_score AS total FROM games
                   WHERE completed=1 AND season>=2024""", con)
g = g[g.home_team.isin(STAD)].copy()
g["kick"] = pd.to_datetime(g.commence_time).dt.tz_localize(None)

if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_game_rain_fc'").fetchone():
    con.execute("CREATE TABLE nflv_game_rain_fc (event_id TEXT PRIMARY KEY, rain_fc_mm REAL)")
    n = 0
    for team, sub in g.groupby("home_team"):
        la, lo = STAD[team]
        for yr, ysub in sub.groupby(sub.kick.dt.year):
            d0 = ysub.kick.min().strftime("%Y-%m-%d")
            d1 = (ysub.kick.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            u = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}"
                 f"&start_date={d0}&end_date={d1}&hourly=precipitation_previous_day1&timezone=UTC")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(u, timeout=60) as r:
                        js = json.loads(r.read())
                    break
                except Exception:
                    if attempt == 2: js = None
                    time.sleep(4)
            if not js or "hourly" not in js: continue
            H = pd.DataFrame(js["hourly"]); H["t"] = pd.to_datetime(H.time)
            for _, gm in ysub.iterrows():
                w = H[(H.t >= gm.kick.floor("h")) & (H.t <= gm.kick.floor("h") + pd.Timedelta(hours=3))]
                v = w.precipitation_previous_day1.dropna()
                if not len(v): continue
                con.execute("INSERT OR REPLACE INTO nflv_game_rain_fc VALUES (?,?)",
                            (gm.event_id, float(v.sum())))
                n += 1
            con.commit(); time.sleep(0.4)
    print(f"fetched 24h-prior precip forecasts for {n} games (2024-25)")

fc = pd.read_sql("SELECT * FROM nflv_game_rain_fc", con)
act = pd.read_sql("SELECT event_id, wind_kn, precip_mm FROM nflv_game_wind", con)
d = g.merge(fc, on="event_id").merge(act, on="event_id", how="left")

ok = d.dropna(subset=["precip_mm", "rain_fc_mm"])
hit = ((ok.precip_mm >= 1) == (ok.rain_fc_mm >= 1)).mean()
print(f"forecast skill (n={len(ok)}): rain>=1mm band agreement {hit:.0%} "
      f"(actual rain rate {(ok.precip_mm>=1).mean():.0%}, forecast rate {(ok.rain_fc_mm>=1).mean():.0%})")

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
d = d.merge(cons, on="event_id").dropna(subset=["total","line","rain_fc_mm"])
d = d[d.total != d.line]; d["under"] = d.total < d.line

print("\nunders at real prices, conditioned on 24h-prior FORECAST (2024-25):")
for lab, m in [("fc rain >=1mm",            d.rain_fc_mm >= 1),
               ("fc rain >=1mm, fc calm",   (d.rain_fc_mm >= 1) & (d.wind_kn < 8)),
               ("fc dry (<0.2mm)",          d.rain_fc_mm < 0.2),
               ("ACTUAL rain >=1mm (ref)",  d.precip_mm >= 1)]:
    s = d[m]
    if len(s) < 10:
        print(f"   {lab:<26} n={len(s):>4}  (too small)"); continue
    roi = np.where(s.under, s.ud - 1, -1.0).mean()
    print(f"   {lab:<26} n={len(s):>4}  P(under) {s.under.mean():.1%}  ROI {roi:+.1%}")
for s_ in (2024, 2025):
    ss = d[(d.rain_fc_mm >= 1) & (d.season == s_)]
    if len(ss):
        roi = np.where(ss.under, ss.ud - 1, -1.0).mean()
        print(f"   fc rain {s_}: n={len(ss)}  P(under) {ss.under.mean():.1%}  ROI {roi:+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Rain forecast-conditioning gate (`rain_forecast_test.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
