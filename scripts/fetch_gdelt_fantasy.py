"""
Fantasy-football-specific news mentions from GDELT GKG on BigQuery.

Filters the draft-prep window (Jun 1 - Sep 1) to articles that are plausibly fantasy
content: URL contains 'fantasy' OR the source is a dedicated fantasy site. Coverage is
known-thin (GDELT barely crawls the fantasy ecosystem - Aug 2024 sample: ~700 player-
mentions total, dominated by radio syndication) so this is a SECONDARY signal for
well-covered names only. Writes per player-season rows to nflv_gdelt_fantasy.

Usage:  python scripts/fetch_gdelt_fantasy.py check    # free dry-run
        python scripts/fetch_gdelt_fantasy.py fetch
"""
import json, os, re, sqlite3, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FANTASY_DOMAINS = ("fantasypros.com", "rotowire.com", "rotoballer.com", "numberfire.com",
                   "footballguys.com", "draftsharks.com", "4for4.com", "fantasyalarm.com",
                   "dynastyleaguefootball.com", "thefantasyfootballers.com")
SEASONS = range(2016, 2026)

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    import pydata_google_auth
    from google.cloud import bigquery
    creds = pydata_google_auth.get_user_credentials(
        ["https://www.googleapis.com/auth/bigquery"], auth_local_webserver=False,
        credentials_cache=pydata_google_auth.cache.ReadWriteCredentialsCache())
    project = json.load(open(os.path.join(ROOT, "config", "gdelt_bq.json")))["project"]
    bq = bigquery.Client(project=project, credentials=creds)

    con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
    rows = con.execute(
        "SELECT DISTINCT player_display_name FROM nflv_season WHERE season>=2015").fetchall()
    names = sorted(set(norm(r[0]) for r in rows if r[0] and len(norm(r[0]).split()) >= 2))
    print(f"matching {len(names)} names", flush=True)

    doms = ",".join(f"'{d}'" for d in FANTASY_DOMAINS)
    qs = []
    for yr in SEASONS:
        qs.append((yr, f"""
            SELECT person, COUNT(*) n,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(0)] AS FLOAT64)) avg_tone,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(1)] AS FLOAT64)) avg_pos,
                   AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(2)] AS FLOAT64)) avg_neg
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`, UNNEST(SPLIT(LOWER(Persons),';')) person
            WHERE _PARTITIONTIME >= TIMESTAMP('{yr}-06-01') AND _PARTITIONTIME < TIMESTAMP('{yr}-09-01')
              AND person IN UNNEST(@names)
              AND (LOWER(DocumentIdentifier) LIKE '%fantasy%' OR SourceCommonName IN ({doms}))
            GROUP BY person"""))

    total = 0
    for yr, sql in qs:
        job = bq.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False,
            query_parameters=[bigquery.ArrayQueryParameter("names", "STRING", names)]))
        gb = job.total_bytes_processed / 1e9
        total += gb
        print(f"  {yr}: {gb:.1f} GB", flush=True)
    print(f"TOTAL {total:.0f} GB (free tier 1024 GB/mo)", flush=True)
    if cmd == "check":
        return

    con.execute("""CREATE TABLE IF NOT EXISTS nflv_gdelt_fantasy
        (name TEXT, season INTEGER, mentions INTEGER,
         avg_tone REAL, avg_pos REAL, avg_neg REAL, PRIMARY KEY(name, season))""")
    for yr, sql in qs:
        job = bq.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("names", "STRING", names)]))
        out = [(r["person"], yr, r["n"], r["avg_tone"], r["avg_pos"], r["avg_neg"])
               for r in job.result()]
        con.executemany("INSERT OR REPLACE INTO nflv_gdelt_fantasy VALUES (?,?,?,?,?,?)", out)
        con.commit()
        print(f"  {yr}: {len(out)} players (billed {job.total_bytes_billed/1e9:.1f} GB)", flush=True)
    n = con.execute("SELECT COUNT(*), SUM(mentions) FROM nflv_gdelt_fantasy").fetchone()
    print(f"done: {n[0]} player-seasons, {n[1]} total mentions", flush=True)


if __name__ == "__main__":
    main()
