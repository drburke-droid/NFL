"""
Historical DraftKings NFL salaries (and DK points) from RotoGuru, one CSV per season.  HOME-PC job.

RotoGuru's archive stops after 2021, so seasons 2022 on come from DailyFantasyFuel instead: its
/nfl/projections/draftkings/<date> page still serves past Sundays, and its default view there is the
DK Sunday main slate (no TNF/SNF/MNF players). Each row carries data-week/-salary/-team/-opp, so we
read those attributes. DFF has no actual DK points, so dk_points is blank for those seasons (join
them from nflverse). --source picks one explicitly; the default is rotoguru <= 2021, dff after.

The cloud sessions cannot reach rotoguru1.com (network policy), so run this where the internet is
open. RotoGuru's "fyday" page lists every player's DK salary and DK points for one week; with
&scsv=1 the table comes as semicolon-separated text inside a <pre> block. Seasons 2014 onward.

    python scripts/fetch_dk_salaries.py                     # 2014 .. this season, skipping files already saved
    python scripts/fetch_dk_salaries.py --start 2024 --end 2025
    python scripts/fetch_dk_salaries.py --season 2026 --refresh     # re-pull the current season (weekly)
    python scripts/fetch_dk_salaries.py --season 2021 --source dff  # DFF instead of RotoGuru

Writes data/dk_salaries/dk_salaries_<season>.csv with columns:
    season, week, player, pos, team, home_away, opp, dk_points, dk_salary, gid, source
(gid is the source's player id -- RotoGuru's numeric id or DFF's hex id, each stable across seasons
but not across sources; D/ST rows have pos "Def" on RotoGuru, "DST" on DFF).
The header is read from the page itself, so a renamed or reordered column does not shift the data.
"""
import argparse, csv, datetime, io, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "dk_salaries")
URL = "http://rotoguru1.com/cgi-bin/fyday.pl?week={w}&year={y}&game=dk&scsv=1"
# RotoGuru header names -> ours (matched case-insensitively, ignoring spaces and punctuation)
COLS = {"week": "week", "year": "season", "gid": "gid", "name": "player", "pos": "pos", "team": "team",
        "ha": "home_away", "oppt": "opp", "dkpoints": "dk_points", "dksalary": "dk_salary"}
DFF_URL = "https://www.dailyfantasyfuel.com/nfl/projections/draftkings/{d}"
FIELDS = ["season", "week", "player", "pos", "team", "home_away", "opp", "dk_points", "dk_salary", "gid", "source"]
key = lambda s: re.sub(r"[^a-z]", "", s.lower())


def parse(page):
    """The semicolon table inside the page's <pre> block -> list of dicts in our column names."""
    m = re.search(r"<pre[^>]*>(.*?)</pre>", page, re.S | re.I)
    text = m.group(1) if m else page
    lines = [l.strip() for l in text.replace("\r", "").split("\n") if l.count(";") >= 5]
    if not lines: return []
    head = [key(h) for h in lines[0].split(";")]
    if "name" not in head or not any("salary" in h for h in head): return []
    idx = {COLS[h]: i for i, h in enumerate(head) if h in COLS}
    out = []
    for row in csv.reader(io.StringIO("\n".join(lines[1:])), delimiter=";"):
        if len(row) < len(head): continue
        r = {f: row[idx[f]].strip() if f in idx else "" for f in FIELDS}
        if "," in r["player"]:                                  # "Last, First" -> "First Last"
            last, first = [x.strip() for x in r["player"].split(",", 1)]; r["player"] = f"{first} {last}"
        r["dk_salary"] = r["dk_salary"].replace("$", "").replace(",", "")
        if not r["player"] or not r["dk_salary"]: continue       # no salary that week: not on the DK slate
        out.append(r)
    return out


def week_sunday(y, w):
    """Sunday of week w: week 1 opens the Thursday after Labor Day (first Monday of September)."""
    sep1 = datetime.date(y, 9, 1)
    labor = sep1 + datetime.timedelta(days=(7 - sep1.weekday()) % 7)
    return labor + datetime.timedelta(days=6 + 7 * (w - 1))


def parse_dff(page, y, w):
    """DFF projection rows -> our columns. Rows tagged with another week are dropped (date drift guard)."""
    out = []
    for tag in re.findall(r"<tr[^>]*data-salary=[^>]*>", page):
        a = dict(re.findall(r'data-([a-z_]+)=\s*"([^"]*)"', tag))
        if not a.get("name") or not a.get("salary") or a.get("week", str(w)) != str(w): continue
        out.append({"season": str(y), "week": str(w), "player": a["name"].strip(), "pos": a.get("pos", ""),
                    "team": a.get("team", ""), "home_away": "a" if a.get("loc") == "@" else "h",
                    "opp": a.get("opp", ""), "dk_points": "", "dk_salary": a["salary"],
                    "gid": a.get("player_id", ""), "source": "dff"})
    return out


def fetch(y, w, tries=3, url=None):
    for i in range(tries):
        try:
            req = urllib.request.Request(url or URL.format(w=w, y=y), headers={"User-Agent": "Mozilla/5.0 (salary history; personal use)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("latin-1")
        except Exception as e:
            if i == tries - 1: print(f"  {y} wk{w}: {str(e)[:80]}"); return ""
            time.sleep(3 * (i + 1))


def main():
    this_season = datetime.date.today().year if datetime.date.today().month >= 9 else datetime.date.today().year - 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2014); ap.add_argument("--end", type=int, default=this_season)
    ap.add_argument("--season", type=int, help="just this one season")
    ap.add_argument("--refresh", action="store_true", help="re-pull seasons whose file already exists")
    ap.add_argument("--source", choices=["auto", "rotoguru", "dff"], default="auto",
                    help="auto = rotoguru through 2021 (its archive ends there), dff from 2022")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests (be polite)")
    A = ap.parse_args()
    seasons = [A.season] if A.season else list(range(A.start, A.end + 1))
    os.makedirs(OUT, exist_ok=True)
    for y in seasons:
        path = os.path.join(OUT, f"dk_salaries_{y}.csv")
        if os.path.exists(path) and not A.refresh:
            print(f"{y}: already saved ({path}); --refresh to re-pull"); continue
        src = A.source if A.source != "auto" else ("rotoguru" if y <= 2021 else "dff")
        rows, empty = [], 0
        for w in range(1, 19):                                   # 17 or 18 regular-season weeks
            if src == "dff":
                if week_sunday(y, w) >= datetime.date.today(): break     # salaries for a slate not yet played
                got = parse_dff(fetch(y, w, url=DFF_URL.format(d=week_sunday(y, w))), y, w)
            else:
                got = parse(fetch(y, w))
            for r in got:
                r["season"] = r["season"] or str(y); r["week"] = r["week"] or str(w); r["source"] = src
            rows += got; print(f"  {y} wk{w}: {len(got)} players"); time.sleep(A.pause)
            empty = empty + 1 if not got else 0
            if empty >= 2 and w >= 3: break                      # past the last week played so far
        if not rows: print(f"{y}: nothing parsed -- not written"); continue
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
        print(f"{y}: {len(rows):,} player-weeks ({src}) -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
