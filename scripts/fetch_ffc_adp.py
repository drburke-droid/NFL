"""
FantasyFootballCalculator crowd ADP (live, from real mock drafts) — free public
API, refreshed continuously through draft season.

    https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2026

Writes nflv_ffc_adp (name/pos/team/adp/stdev/times_drafted). Re-run any time
before a draft to freshen; build_draft_tool.py bakes it into data.js.
"""
import os, re, sqlite3, json, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2026"
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(s):
    s = str(s).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("-", " ").replace(",", "")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    players = data.get("players", [])
    if not players:
        raise SystemExit("FFC API returned no players")
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS nflv_ffc_adp")
    con.execute("""CREATE TABLE nflv_ffc_adp
                   (name TEXT, nm TEXT, position TEXT, team TEXT,
                    adp REAL, stdev REAL, times_drafted INTEGER)""")
    pos_map = {"DEF": "DST", "PK": "K"}
    rows = [(p["name"], norm(p["name"]), pos_map.get(p["position"], p["position"]),
             p.get("team"), p.get("adp"), p.get("stdev"), p.get("times_drafted"))
            for p in players]
    con.executemany("INSERT INTO nflv_ffc_adp VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM nflv_ffc_adp").fetchone()[0]
    top = con.execute("SELECT name, position, adp FROM nflv_ffc_adp ORDER BY adp LIMIT 5").fetchall()
    con.close()
    print(f"nflv_ffc_adp: {n} players (12-team PPR, 2026)")
    print("top 5:", " | ".join(f"{a} {b} {c}" for a, b, c in top))


if __name__ == "__main__":
    main()
