"""
GDELT news volume + tone ingest (DOC 2.0 API; coverage floor Jan 2017).

Per player: one timelinevolraw call (daily article counts + norm denominator)
and one timelinetone call (daily avg tone), 2017->present, aggregated to WEEKLY
rows in nflv_gdelt_news. Query is '"Name" (nfl OR football)' to disambiguate
common names.

GDELT throttles hard (429s well below the advertised 1-per-5s), so this runs
with adaptive backoff and is fully resumable — safe to kill and re-run any
number of times; already-fetched players are skipped. Full priority list is
~700 players ~= overnight at the observed sustainable rate.

Priority order (so partial coverage is immediately usable):
  1. cheap-pool players 2017+ who were in the FFA file (the actionable darts)
  2. every cheap-pool HIT 2017+ (the outcomes we must explain)
  3. a fixed random sample of not-in-FFA misses (control group)
"""
import os, sys, time, json, sqlite3, urllib.request, urllib.parse
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
UA = {"User-Agent": "nfl-fantasy-research/1.0 (drburke@calgaryvisioncentre.com)"}
BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
SPAN = "&startdatetime=20170101000000&enddatetime=20260301000000"
PAUSE = 60          # starting inter-request pause (s); adapts upward on 429
MAX_PAUSE = 600


def gd(query, mode, pause_box):
    url = (BASE + "?query=" + urllib.parse.quote(query) + f"&mode={mode}&format=json" + SPAN)
    req = urllib.request.Request(url, headers=UA)
    while True:
        time.sleep(pause_box[0])
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode("utf-8", "replace")
            if body.lstrip().startswith("{"):
                pause_box[0] = max(PAUSE, pause_box[0] * 0.8)
                return json.loads(body)
            # HTML/plain-text throttle message
            pause_box[0] = min(MAX_PAUSE, pause_box[0] * 2)
            print(f"    throttle text; pause -> {pause_box[0]:.0f}s", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                pause_box[0] = min(MAX_PAUSE, pause_box[0] * 2)
                print(f"    429; pause -> {pause_box[0]:.0f}s", flush=True)
            else:
                return None
        except Exception as e:
            print(f"    err {e}; retrying", flush=True)
            time.sleep(30)


def weekly(points, field):
    """Aggregate GDELT daily timeline points to ISO-week rows."""
    if not points: return {}
    s = pd.DataFrame(points)
    s["date"] = pd.to_datetime(s["date"].str[:8])
    s["wk"] = s["date"].dt.strftime("%G-W%V")
    if field == "vol":
        g = s.groupby("wk").agg(value=("value", "sum"), norm=("norm", "sum"))
        return {w: (int(r.value), int(r.norm)) for w, r in g.iterrows()}
    g = s[s["value"] != 0].groupby("wk").agg(tone=("value", "mean"))
    return {w: float(r.tone) for w, r in g.iterrows()}


def targets():
    df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
    d = df[(df.season >= 2017) & (df.season <= 2025) & (df.cheap == 1)]
    g1 = d[d.in_ffa == 1][["player_id", "name"]]
    g2 = d[d.hit == 1][["player_id", "name"]]
    miss = d[(d.in_ffa == 0) & (d.hit == 0)][["player_id", "name"]].drop_duplicates("player_id")
    g3 = miss.sample(n=min(300, len(miss)), random_state=7)
    out = pd.concat([g1, g2, g3]).drop_duplicates("player_id").reset_index(drop=True)
    return out


def main():
    con = sqlite3.connect(DB, timeout=120)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=120000")
    cur.execute("""CREATE TABLE IF NOT EXISTS nflv_gdelt_news
                   (player_id TEXT, wk TEXT, articles INT, norm INT, tone REAL,
                    PRIMARY KEY (player_id, wk))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS nflv_gdelt_done
                   (player_id TEXT PRIMARY KEY, name TEXT, ok INT)""")
    con.commit()

    t = targets()
    done = {r[0] for r in cur.execute("SELECT player_id FROM nflv_gdelt_done")}
    todo = t[~t.player_id.isin(done)]
    print(f"{len(t)} targets, {len(done)} done, {len(todo)} to go", flush=True)

    pause_box = [PAUSE]
    for i, (_, row) in enumerate(todo.iterrows()):
        q = f'"{row["name"]}" (nfl OR football)'
        jv = gd(q, "timelinevolraw", pause_box)
        jt = gd(q, "timelinetone", pause_box)
        ok = 0
        if jv and jv.get("timeline"):
            vol = weekly(jv["timeline"][0]["data"], "vol")
            ton = weekly(jt["timeline"][0]["data"], "tone") if jt and jt.get("timeline") else {}
            rows = [(row.player_id, w, v[0], v[1], ton.get(w)) for w, v in vol.items()]
            if rows:
                cur.executemany("INSERT OR REPLACE INTO nflv_gdelt_news VALUES (?,?,?,?,?)", rows)
                ok = 1
        cur.execute("INSERT OR REPLACE INTO nflv_gdelt_done VALUES (?,?,?)",
                    (row.player_id, row["name"], ok))
        con.commit()
        if (i + 1) % 10 == 0:
            n = cur.execute("SELECT COUNT(DISTINCT player_id) FROM nflv_gdelt_news").fetchone()[0]
            print(f"  {i+1}/{len(todo)} processed ({n} players stored), pause {pause_box[0]:.0f}s", flush=True)
    print("all targets processed")


if __name__ == "__main__":
    main()
