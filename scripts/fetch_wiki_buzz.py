"""
Wikipedia pageview "buzz" ingest — the attention analog of a news/GDELT pull.

For every player in the late-breakout frame (2,313 unique), resolve their English
Wikipedia article and pull MONTHLY pageviews from 2015-07 (API floor) to present.
August-of-season views are the pre-draft camp-buzz signal (validated on Puka
Nacua 2023: Aug = 4x his June baseline, before he'd played a snap).

Resumable: players already in nflv_wiki_buzz (or marked unresolved in
nflv_wiki_titles) are skipped on re-run.

Tables:
  nflv_wiki_titles (player_id, name, wiki_title, resolved)
  nflv_wiki_buzz   (player_id, ym, views)
"""
import os, sys, time, json, sqlite3, urllib.request, urllib.parse
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
UA = {"User-Agent": "nfl-fantasy-research/1.0 (drburke@calgaryvisioncentre.com)"}
PAUSE = 0.12


def get(url, retries=3):
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 404: return None            # no pageview data / no article
            if e.code == 429: time.sleep(10 * (a + 1)); continue
            if a == retries - 1: return None
            time.sleep(3)
        except Exception:
            if a == retries - 1: return None
            time.sleep(3)
    return None


def resolve_title(name):
    """Best-effort en-wiki article for an NFL player; require last-name overlap."""
    q = urllib.parse.quote(f"{name} American football")
    j = get("https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={q}&srlimit=5&format=json")
    if not j: return None
    last = name.split()[-1].lower()
    first = name.split()[0].lower().rstrip(".")
    for h in j["query"]["search"]:
        t = h["title"]
        tl = t.lower()
        if last in tl and (first[:3] in tl or "." in name):
            return t
    return None


def fetch_views(title):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    j = get("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{t}/monthly/20150701/20260701")
    if not j: return []
    return [(i["timestamp"][:6], i["views"]) for i in j.get("items", [])]


def refresh_daily(con, year):
    """Pre-draft refresh: daily views Jun 1 -> today for the target season, for all
    resolved players (August-to-date drives the buzz features at draft time)."""
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS nflv_wiki_buzz_daily
                   (player_id TEXT, date TEXT, views INT, PRIMARY KEY (player_id, date))""")
    con.commit()
    rows = cur.execute("SELECT player_id, wiki_title FROM nflv_wiki_titles WHERE resolved=1").fetchall()
    print(f"daily refresh {year}: {len(rows)} players")
    for i, (pid, title) in enumerate(rows):
        t = urllib.parse.quote(title.replace(" ", "_"), safe="")
        j = get("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                f"en.wikipedia/all-access/user/{t}/daily/{year}0601/{year}0915")
        time.sleep(PAUSE)
        if j:
            cur.executemany("INSERT OR REPLACE INTO nflv_wiki_buzz_daily VALUES (?,?,?)",
                            [(pid, it["timestamp"][:8], it["views"]) for it in j.get("items", [])])
        if (i + 1) % 200 == 0:
            con.commit(); print(f"  {i+1}/{len(rows)}", flush=True)
    con.commit()
    print("daily refresh done")


def main():
    con = sqlite3.connect(DB, timeout=120)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=120000")
    cur.execute("""CREATE TABLE IF NOT EXISTS nflv_wiki_titles
                   (player_id TEXT PRIMARY KEY, name TEXT, wiki_title TEXT, resolved INT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS nflv_wiki_buzz
                   (player_id TEXT, ym TEXT, views INT, PRIMARY KEY (player_id, ym))""")
    con.commit()

    frame = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
    players = frame[["player_id", "name"]].drop_duplicates("player_id")
    done = {r[0] for r in cur.execute("SELECT player_id FROM nflv_wiki_titles")}
    todo = players[~players.player_id.isin(done)]
    print(f"{len(players)} players, {len(done)} done, {len(todo)} to fetch")

    n_ok = n_miss = 0
    for i, (_, row) in enumerate(todo.iterrows()):
        title = resolve_title(row["name"]); time.sleep(PAUSE)
        views = fetch_views(title) if title else []
        time.sleep(PAUSE)
        cur.execute("INSERT OR REPLACE INTO nflv_wiki_titles VALUES (?,?,?,?)",
                    (row.player_id, row["name"], title, 1 if views else 0))
        if views:
            cur.executemany("INSERT OR REPLACE INTO nflv_wiki_buzz VALUES (?,?,?)",
                            [(row.player_id, ym, v) for ym, v in views])
            n_ok += 1
        else:
            n_miss += 1
        if (i + 1) % 50 == 0:
            con.commit()
            print(f"  {i+1}/{len(todo)} (ok {n_ok}, miss {n_miss})", flush=True)
    con.commit()
    tot = cur.execute("SELECT COUNT(*), COUNT(DISTINCT player_id) FROM nflv_wiki_buzz").fetchone()
    print(f"done: +{n_ok} players fetched, {n_miss} unresolved; table now {tot[0]} rows / {tot[1]} players")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        import datetime
        yr = int(sys.argv[2]) if len(sys.argv) > 2 else datetime.date.today().year
        c = sqlite3.connect(DB, timeout=120)
        c.execute("PRAGMA busy_timeout=120000")
        refresh_daily(c, yr)
    else:
        main()
