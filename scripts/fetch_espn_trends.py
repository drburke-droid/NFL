"""
Parse an ESPN Live Draft Trends paste (data/espn_trends_2026.txt) into
nflv_espn_trends: sitewide ADP + average auction $ per player.

Same tolerant block parser as the draft tool's in-browser upload
(index.html parseEspnTrends): rank / doubled-name / name / team / POS /
ADP / adp-trend / AVG AUCTION $ / $-trend / %rostered — anchored on the
doubled-name line, position scanned forward. Re-run after saving a fresh
paste; build_draft_tool.py bakes the values into data.js (espn_av/espn_adp).
"""
import os, re, sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
SRC = os.path.join(ROOT, "data", "espn_trends_2026.txt")
POSSET = {"QB", "RB", "WR", "TE", "K", "D/ST"}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
NUM = re.compile(r"^[+-]?\d+(\.\d+)?$")


def norm(s):
    s = str(s).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("-", " ").replace(",", "")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    lines = [l.strip() for l in open(SRC, encoding="utf-8") if l.strip()]
    rows = []
    i = 0
    while i < len(lines) - 6:
        if not lines[i].isdigit():
            i += 1; continue
        d = lines[i + 1]
        half = d[: len(d) // 2]
        if len(d) < 4 or half != d[len(d) // 2:]:
            i += 1; continue
        j = -1
        for k in range(i + 2, min(i + 8, len(lines))):
            if lines[k] in POSSET:
                j = k; break
        if j < 0 or not NUM.match(lines[j + 1]) or not NUM.match(lines[j + 3]):
            i += 1; continue
        pos = "DST" if lines[j] == "D/ST" else lines[j]
        rows.append((half, norm(half), pos, lines[j - 1],
                     float(lines[j + 1]), float(lines[j + 3])))
        i = j + 3
    if len(rows) < 10:
        raise SystemExit(f"only parsed {len(rows)} players — is {SRC} an ESPN Live Draft Trends paste?")
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS nflv_espn_trends")
    con.execute("""CREATE TABLE nflv_espn_trends
                   (name TEXT, nm TEXT, position TEXT, team TEXT, adp REAL, av REAL)""")
    con.executemany("INSERT INTO nflv_espn_trends VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    top = con.execute("SELECT name, position, av FROM nflv_espn_trends ORDER BY av DESC LIMIT 5").fetchall()
    con.close()
    print(f"nflv_espn_trends: {len(rows)} players")
    print("top 5 by auction $:", " | ".join(f"{a} {b} ${c}" for a, b, c in top))


if __name__ == "__main__":
    main()
