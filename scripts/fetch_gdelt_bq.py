"""
Player news sentiment at scale from GDELT on BigQuery (no rate limits, tone included).

Replaces the drip-fed GDELT DOC API (fetch_gdelt_news.py: 570 players, 2017+) with the
public BigQuery datasets:
  - gdelt-bq.gdeltv2.gkg_partitioned   (Feb 2015 -> now; V2Persons + V2Tone) [source='gkg']
  - gdelt-bq.full.events_partitioned   (2012 -> Feb 2015 backfill via CAMEO actor names +
    per-event article AvgTone) [source='events'] - noisier person tagging; GKG 1.0 was
    never loaded into BigQuery, so this is the only pre-2015 person-ish signal.

One-time setup (no gcloud SDK needed):
  1. console.cloud.google.com -> create a project (any name), note the PROJECT ID
     (BigQuery API is on by default; the free tier is 1 TiB of query per month)
  2. python scripts/fetch_gdelt_bq.py auth --project YOUR_PROJECT_ID   (opens a browser)
Then:
  python scripts/fetch_gdelt_bq.py check          # FREE dry-run: exact bytes each query costs
  python scripts/fetch_gdelt_bq.py fetch --yes    # runs the queries, writes db nflv_gdelt_bq

Output table nflv_gdelt_bq: (name, player_id, ym, articles, avg_tone, avg_pos, avg_neg)
per player-month. CAVEAT: GDELT persons are raw name strings - "Josh Allen" conflates the
Bills QB with the Jaguars edge rusher; validate any single-player analysis. --nfl-only adds
an NFL-organization co-mention filter (cleaner, but scans the organizations column too:
roughly doubles bytes).
"""
import argparse, json, os, re, sqlite3, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "gdelt_bq.json")
SCOPES = ["https://www.googleapis.com/auth/bigquery"]
V2_START = 2015          # gkg_partitioned coverage begins 2015-02-19
FREE_GUARD_GB = 400      # refuse to run past this without --yes (free tier = 1024 GB/mo)

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()

def creds():
    import pydata_google_auth
    return pydata_google_auth.get_user_credentials(SCOPES, use_local_webserver=True)

def client():
    if not os.path.exists(CFG):
        sys.exit("No config/gdelt_bq.json - run:  python scripts/fetch_gdelt_bq.py auth --project YOUR_PROJECT_ID")
    from google.cloud import bigquery
    project = json.load(open(CFG))["project"]
    return bigquery.Client(project=project, credentials=creds())

def player_names():
    con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
    rows = con.execute("SELECT DISTINCT player_display_name FROM nflv_season WHERE season>=2012").fetchall()
    names = sorted(set(norm(r[0]) for r in rows if r[0] and len(norm(r[0]).split()) >= 2))
    ids = {}
    for pid_, nm in con.execute("SELECT DISTINCT player_id, player_display_name FROM nflv_season"):
        ids.setdefault(norm(nm), pid_)
    return names, ids

def queries(names, start, end, nfl_only, weekly=False):
    """(label, sql, params) per year - partition-pruned v2 queries + one v1 backfill."""
    from google.cloud import bigquery
    P = lambda: [bigquery.ArrayQueryParameter("names", "STRING", names)]
    out = []
    org2 = "AND LOWER(Organizations) LIKE '%national football league%'" if nfl_only else ""
    if start < V2_START:
        d0 = f"{start}-01-01"
        out.append((f"events {start}-2015/02", f"""
            SELECT person, CAST(MonthYear AS STRING) AS ym, COUNT(*) n,
                   AVG(AvgTone) avg_tone, CAST(NULL AS FLOAT64) avg_pos, CAST(NULL AS FLOAT64) avg_neg,
                   'events' AS source
            FROM (
              SELECT MonthYear, LOWER(Actor1Name) AS person, AvgTone
              FROM `gdelt-bq.full.events_partitioned`
              WHERE _PARTITIONTIME >= TIMESTAMP('{d0}') AND _PARTITIONTIME < TIMESTAMP('2015-02-19')
              UNION ALL
              SELECT MonthYear, LOWER(Actor2Name), AvgTone
              FROM `gdelt-bq.full.events_partitioned`
              WHERE _PARTITIONTIME >= TIMESTAMP('{d0}') AND _PARTITIONTIME < TIMESTAMP('2015-02-19')
            )
            WHERE person IN UNNEST(@names)
            GROUP BY person, ym""", P()))
    if weekly:
        # WEEKLY grain, in-season only (Sep 1 -> Jan 31 spanning the year boundary): the
        # props/lineup tests need in-season weeks, and the date filter cuts scan ~60%
        out2 = []
        for yr in range(max(start, V2_START), end + 1):
            out2.append((f"wk {yr} season", f"""
            SELECT person, FORMAT_DATE('%Y-%m-%d', DATE_TRUNC(DATE(_PARTITIONTIME), WEEK(TUESDAY))) AS wk,
                   COUNT(*) n,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(0)] AS FLOAT64)) avg_tone,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(1)] AS FLOAT64)) avg_pos,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(2)] AS FLOAT64)) avg_neg
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`, UNNEST(SPLIT(LOWER(Persons),';')) person
            WHERE _PARTITIONTIME >= TIMESTAMP('{yr}-09-01') AND _PARTITIONTIME < TIMESTAMP('{yr+1}-02-01')
              AND person IN UNNEST(@names)
            GROUP BY person, wk""", P()))
        return out2
    for yr in range(max(start, V2_START), end + 1):
        out.append((f"v2 {yr}", f"""
            SELECT person, FORMAT_TIMESTAMP('%Y%m', _PARTITIONTIME) AS ym, COUNT(*) n,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(0)] AS FLOAT64)) avg_tone,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(1)] AS FLOAT64)) avg_pos,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(2)] AS FLOAT64)) avg_neg,
                   'gkg' AS source
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`, UNNEST(SPLIT(LOWER(Persons),';')) person
            WHERE _PARTITIONTIME >= TIMESTAMP('{yr}-01-01') AND _PARTITIONTIME < TIMESTAMP('{yr+1}-01-01')
              AND person IN UNNEST(@names) {org2}
            GROUP BY person, ym""", P()))
    return out

def dry_run(bq, qs):
    from google.cloud import bigquery
    total = 0
    print(f"{'query':<22}{'scans':>12}")
    for label, sql, params in qs:
        job = bq.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False, query_parameters=params))
        gb = job.total_bytes_processed / 1e9
        total += gb
        print(f"{label:<22}{gb:>10.1f} GB")
    print(f"{'TOTAL':<22}{total:>10.1f} GB   (free tier: 1024 GB/month; on-demand ~$6.25/TiB beyond)")
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["auth", "check", "fetch"])
    ap.add_argument("--project")
    ap.add_argument("--start", type=int, default=2013)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--nfl-only", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--weekly", action="store_true", help="weekly grain, in-season Sep-Jan, -> nflv_gdelt_wk")
    a = ap.parse_args()

    if a.cmd == "auth":
        if not a.project: sys.exit("need --project YOUR_PROJECT_ID")
        creds()                                          # opens browser, caches token locally
        os.makedirs(os.path.dirname(CFG), exist_ok=True)
        json.dump({"project": a.project}, open(CFG, "w"))
        print(f"authenticated; project '{a.project}' saved to config/gdelt_bq.json")
        return

    bq = client()
    names, ids = player_names()
    print(f"matching {len(names)} distinct player names (nflv_season 2012+)")
    qs = queries(names, a.start, a.end, a.nfl_only, a.weekly)
    total = dry_run(bq, qs)
    if a.cmd == "check":
        return
    if total > FREE_GUARD_GB and not a.yes:
        sys.exit(f"total scan {total:.0f} GB > {FREE_GUARD_GB} GB guard - rerun with --yes to proceed")

    con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
    if a.weekly:
        con.execute("""CREATE TABLE IF NOT EXISTS nflv_gdelt_wk
            (name TEXT, player_id TEXT, wk TEXT, articles INTEGER,
             avg_tone REAL, avg_pos REAL, avg_neg REAL, PRIMARY KEY(name, wk))""")
    con.execute("""CREATE TABLE IF NOT EXISTS nflv_gdelt_bq
        (name TEXT, player_id TEXT, ym TEXT, articles INTEGER,
         avg_tone REAL, avg_pos REAL, avg_neg REAL, source TEXT, PRIMARY KEY(name, ym, source))""")
    from google.cloud import bigquery
    wrote = 0
    for label, sql, params in qs:
        print(f"running {label} ...", flush=True)
        job = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        if a.weekly:
            rows = [(r["person"], ids.get(r["person"]), r["wk"], r["n"],
                     r["avg_tone"], r["avg_pos"], r["avg_neg"]) for r in job.result()]
            con.executemany("INSERT OR REPLACE INTO nflv_gdelt_wk VALUES (?,?,?,?,?,?,?)", rows)
        else:
            rows = [(r["person"], ids.get(r["person"]), r["ym"], r["n"],
                     r["avg_tone"], r["avg_pos"], r["avg_neg"], r["source"]) for r in job.result()]
            con.executemany("INSERT OR REPLACE INTO nflv_gdelt_bq VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        wrote += len(rows)
        print(f"  {label}: {len(rows)} player-months (billed {job.total_bytes_billed/1e9:.1f} GB)")
    n_named = con.execute("SELECT COUNT(DISTINCT name) FROM nflv_gdelt_bq").fetchone()[0]
    print(f"done: {wrote} rows upserted; {n_named} distinct players in nflv_gdelt_bq")

if __name__ == "__main__":
    main()
