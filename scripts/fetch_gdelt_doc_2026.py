"""
August-2026 news tone for every NFL team's RB1, via the free GDELT DOC 2.0 API.

Feeds the HANDCUFF-STORM flag (gdelt_buzz_interaction_study.py, 2016-25: RB1 with
bottom-quartile August news tone x RB2 with top-quartile wiki buzz -> RB2 hit 33%
vs 19-21% base, +22 pts over FFA; ~2.4 qualifying pairs/yr). The BigQuery GKG table
(nflv_gdelt_bq) ends at 2025, so the current August comes from the DOC API instead:
per-name article volume (timelinevolraw) + average tone (timelinetone) since Aug 1.

RB1 = each NFL team's top RB by market price (max of FFA AAV / ESPN $), falling
back to projected points. Writes nflv_gdelt_doc2026(team, name, articles, avg_tone).

    python scripts/fetch_gdelt_doc_2026.py
"""
import json, os, re, sqlite3, subprocess, sys, time, urllib.parse, urllib.request

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
UA = {"User-Agent": "nfl-fantasy-research/1.0 (drburke@calgaryvisioncentre.com)"}
API = "https://api.gdeltproject.org/api/v2/doc/doc"
PACE = 1.5   # seconds between API calls; fixup mode slows this way down


def load_players():
    js = ("const s=require('fs').readFileSync(" + json.dumps(os.path.join(ROOT, "docs", "data.js"))
          + ",'utf8');const v=new Function(s+';return PLAYERS;')();process.stdout.write(JSON.stringify(v));")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit("node data.js load failed: " + out.stderr[:300])
    return json.loads(out.stdout)


def get(params, retries=3):
    url = API + "?" + urllib.parse.urlencode(params)
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                raw = r.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip().startswith("{") else None
        except Exception:
            time.sleep(4 * (a + 1))
    return None


def fetch_name(name):
    """(articles since Aug 1, avg tone). Query is scoped with the word `football` so
    namesake noise (James Cook the explorer...) doesn't pollute tone; every relevant
    NFL article contains it. Volume call re-tried when it comes back empty while the
    tone call succeeded (the API rate-limits intermittently)."""
    q = f'"{name}" football sourcelang:eng'
    base = {"query": q, "format": "json", "startdatetime": "20260801000000"}
    def read_vol():
        vol = get({**base, "mode": "timelinevolraw"})
        if vol and vol.get("timeline"):
            return int(sum(d.get("value", 0) for d in vol["timeline"][0].get("data", [])))
        return None
    n_art = read_vol()
    time.sleep(PACE)
    tone = get({**base, "mode": "timelinetone"})
    time.sleep(PACE)
    avg_tone = None
    if tone and tone.get("timeline"):
        pts = [d.get("value") for d in tone["timeline"][0].get("data", []) if d.get("value") is not None]
        if pts: avg_tone = sum(pts) / len(pts)
    if n_art is None and avg_tone is not None:      # vol call flaked — one more try
        time.sleep(PACE * 3); n_art = read_vol(); time.sleep(PACE)
    return (n_art or 0), avg_tone


def main():
    players = load_players()
    rbs = [p for p in players if p.get("position") == "RB"]
    best = {}
    for p in rbs:
        cost = max(p.get("ffa_aav") or 0, p.get("espn_av") or 0)
        key = p.get("team") or "?"
        rank = (cost, p.get("proj_pts") or 0)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, p)
    rb1s = {t: p for t, (rk, p) in best.items() if t and t != "?"}
    print(f"{len(rb1s)} team RB1s (top RB by market $, fallback proj)")

    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS nflv_gdelt_doc2026
                   (team TEXT PRIMARY KEY, name TEXT, articles INT, avg_tone REAL, fetched TEXT)""")
    from datetime import date
    fixup = len(sys.argv) > 1 and sys.argv[1] == "fixup"
    if fixup:
        # gap-fill: re-fetch only rows the flaky API shortchanged (0 articles or no
        # tone), at a much slower cadence that stays under the rate limit
        global PACE; PACE = 12.0
        bad = {r[0] for r in con.execute(
            "SELECT team FROM nflv_gdelt_doc2026 WHERE articles=0 OR avg_tone IS NULL")}
        done = {t for t in rb1s if t not in bad}
        print(f"fixup: re-fetching {len(bad)} gap teams slowly")
    else:
        done = {r[0] for r in con.execute("SELECT team FROM nflv_gdelt_doc2026 WHERE fetched=?",
                                          (date.today().isoformat(),))}
        if done: print(f"resuming: {len(done)} teams already fetched today")
    for i, (team, p) in enumerate(sorted(rb1s.items())):
        if team in done: continue
        n, t = fetch_name(p["name"])
        con.execute("INSERT OR REPLACE INTO nflv_gdelt_doc2026 VALUES (?,?,?,?,?)",
                    (team, p["name"], n, t, date.today().isoformat()))
        con.commit()
        print(f"  [{i+1:>2}/{len(rb1s)}] {team:4s} {p['name']:24s} articles={n:<5} tone={'--' if t is None else f'{t:+.2f}'}",
              flush=True)
    rows = con.execute("SELECT team, name, articles, avg_tone FROM nflv_gdelt_doc2026 "
                       "WHERE avg_tone IS NOT NULL ORDER BY avg_tone").fetchall()
    tones = [r[3] for r in rows]
    if tones:
        q1 = sorted(tones)[max(0, len(tones) // 4 - 1)]
        print(f"\nbottom-quartile tone cut: <= {q1:+.2f}; storm candidates (tone<=Q1, >=20 articles):")
        for tm, nm, n, tn in rows:
            if tn <= q1 and n >= 20:
                print(f"  🚨 {tm:4s} {nm:24s} articles={n} tone={tn:+.2f}")
    con.close()


if __name__ == "__main__":
    main()
