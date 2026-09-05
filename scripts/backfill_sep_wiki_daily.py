"""Backfill daily wiki views Aug 25 - Sep 7 for historical cheap-pool (dart)
player-seasons 2016-2025, into a separate table nflv_wiki_buzz_daily_sep so the
production buzz_features() pipeline is untouched."""
import sqlite3, os, time, urllib.parse, sys
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location(
    "fwb", r"C:\Users\drbur\Documents\GitHub\NFL\scripts\fetch_wiki_buzz.py")
fwb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fwb)

root = r"C:\Users\drbur\Documents\GitHub\NFL"
frame = pd.read_pickle(os.path.join(root, r"outputs\models\late_breakout_frame.pkl"))
print("has cheap col:", "cheap" in frame.columns, "| has hit:", "hit" in frame.columns)
pool = frame[frame.cheap == 1][["player_id", "season", "name"]].drop_duplicates()
print("cheap player-seasons:", len(pool))

con = sqlite3.connect(os.path.join(root, "db", "nfl_odds.db"), timeout=120)
cur = con.cursor()
cur.execute("PRAGMA busy_timeout=120000")
cur.execute("""CREATE TABLE IF NOT EXISTS nflv_wiki_buzz_daily_sep
               (player_id TEXT, date TEXT, views INT, PRIMARY KEY (player_id, date))""")
con.commit()

titles = pd.read_sql("SELECT player_id, wiki_title FROM nflv_wiki_titles WHERE resolved=1", con)
pool = pool.merge(titles, on="player_id", how="inner")
# skip player-seasons already backfilled
done = pd.read_sql("SELECT DISTINCT player_id, substr(date,1,4) yr FROM nflv_wiki_buzz_daily_sep", con)
done_set = set(zip(done.player_id, done.yr.astype(int))) if len(done) else set()
pool = pool[~pool.apply(lambda r: (r.player_id, r.season) in done_set, axis=1)]
print("to fetch (resolved, not done):", len(pool))

n_ok = n_empty = 0
for i, (_, row) in enumerate(pool.iterrows()):
    y = int(row.season)
    t = urllib.parse.quote(row.wiki_title.replace(" ", "_"), safe="")
    j = fwb.get("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                f"en.wikipedia/all-access/user/{t}/daily/{y}0825/{y}0907")
    time.sleep(fwb.PAUSE)
    if j and j.get("items"):
        cur.executemany("INSERT OR REPLACE INTO nflv_wiki_buzz_daily_sep VALUES (?,?,?)",
                        [(row.player_id, it["timestamp"][:8], it["views"]) for it in j["items"]])
        n_ok += 1
    else:
        n_empty += 1
    if (i + 1) % 250 == 0:
        con.commit()
        print(f"  {i+1}/{len(pool)} (ok {n_ok}, empty {n_empty})", flush=True)
con.commit()
print(f"done: {n_ok} fetched, {n_empty} empty/missing")
con.close()
