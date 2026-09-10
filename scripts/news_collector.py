"""Point-in-time news collector for the "beat writer / practice report vs market" study.

Every record is append-only JSONL with a `fetched_at` stamp and the source's own timestamp, so a
backtest can ask "what was known at time t?" without a latest-value join.

Sources (all free, no auth):
  bsky      posts from the curated beat-writer list data/news/beat_writers_bsky.csv
            (public.api.bsky.app getAuthorFeed; history back to each account's first post,
            most NFL beat writers joined Nov 2024)
  espn      ESPN NFL news API (site.api.espn.com), articles carry athlete/team tags
  nflinj    NFL.com official injury report page (practice status Wed/Thu/Fri + game status)
  sleeper   Sleeper trending adds/drops (24h counts) — the crowd's attention
  wiki      Wikipedia daily pageviews per player (Wikimedia REST, 2015-07 onward) — the
            historical attention proxy; needs a player list (nflv_wiki_titles in the odds DB)

Usage:
  python scripts/news_collector.py snapshot                 # bsky (new posts only) + espn + nflinj + sleeper
  python scripts/news_collector.py backfill-bsky            # page every listed account back to its first post
  python scripts/news_collector.py wiki 2025-09-01 2026-01-10 [--limit N]   # daily views, players in nflv_wiki_titles
Files: data/news/<source>_<YYYY-MM>.jsonl ; state in data/news/state.json (last uri per account).
Run `snapshot` on a schedule (Task Scheduler / cron): hourly Wed-Sun in season is plenty.
"""
import os, sys, json, csv, re, time, sqlite3, urllib.request, urllib.parse
from datetime import datetime, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = os.path.join(ROOT, "data", "news"); os.makedirs(ND, exist_ok=True)
STATE = os.path.join(ND, "state.json")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
TEAMS = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
now_iso = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def get(url, retries=3, sleep=2.0, raw=False, ua=None):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=ua or UA), timeout=40) as r:
                b = r.read()
            return b.decode("utf-8", "ignore") if raw else json.loads(b)
        except Exception as e:
            err = e; time.sleep(sleep * (i + 1))
    print(f"  fetch failed: {url[:90]} ({str(err)[:60]})"); return None

def append(source, rows):
    if not rows: return 0
    by_month = {}
    for r in rows: by_month.setdefault(r["fetched_at"][:7], []).append(r)
    for ym, rr in by_month.items():
        with open(os.path.join(ND, f"{source}_{ym}.jsonl"), "a", encoding="utf-8") as f:
            for r in rr: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

def load_state():
    return json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
def save_state(s): json.dump(s, open(STATE, "w", encoding="utf-8"), indent=1)

# ---------------- Bluesky ----------------
def bsky_accounts():
    p = os.path.join(ND, "beat_writers_bsky.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return [r for r in rows if r.get("use", "1") not in ("0", "n", "no")]

def bsky_pull(handle, stop_uri=None, max_pages=50):
    out, cur, pages = [], None, 0
    while pages < max_pages:
        u = (f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={handle}&limit=100&filter=posts_no_replies"
             + (f"&cursor={urllib.parse.quote(cur)}" if cur else ""))
        d = get(u); pages += 1
        if not d: break
        feed = d.get("feed", [])
        for x in feed:
            p = x["post"]; rec = p.get("record", {})
            if stop_uri and p["uri"] == stop_uri: return out, True
            if x.get("reason"):   # repost — keep, flagged
                pass
            out.append({"uri": p["uri"], "handle": handle, "created_at": rec.get("createdAt"), "text": rec.get("text", ""),
                        "repost": bool(x.get("reason")), "likes": p.get("likeCount"), "reposts": p.get("repostCount")})
        cur = d.get("cursor")
        if not cur or not feed: break
        time.sleep(0.25)
    return out, False

def cmd_bsky(backfill=False):
    st = load_state(); acc = bsky_accounts(); n = 0; fa = now_iso()
    for a in acc:
        h = a["handle"]; last = None if backfill else st.get("bsky", {}).get(h)
        posts, _ = bsky_pull(h, stop_uri=last, max_pages=200 if backfill else 5)
        if posts:
            st.setdefault("bsky", {})[h] = posts[0]["uri"]     # newest first
            for p in posts: p.update({"team": a["team"], "source": "bsky", "fetched_at": fa})
            n += append("bsky", posts)
        print(f"  {a['team']} {h}: {len(posts)} new")
    save_state(st); print(f"bsky: {n} posts written")

# ---------------- ESPN news ----------------
def cmd_espn():
    st = load_state(); seen = set(st.get("espn_ids", [])[-5000:]); rows = []; fa = now_iso()
    d = get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=100", ua={"User-Agent": "curl/8.4.0"})
    for a in (d or {}).get("articles", []):
        i = str(a.get("id") or a.get("dataSourceIdentifier") or a.get("links", {}).get("web", {}).get("href"))
        if i in seen: continue
        seen.add(i)
        ath = [c.get("description") for c in a.get("categories", []) if c.get("type") == "athlete"]
        tm = [TEAMS.get(c.get("description"), c.get("description")) for c in a.get("categories", []) if c.get("type") == "team"]
        rows.append({"id": i, "published": a.get("published"), "headline": a.get("headline"), "description": a.get("description"),
                     "athletes": ath, "teams": tm, "type": a.get("type"), "source": "espn", "fetched_at": fa})
    st["espn_ids"] = list(seen)[-8000:]; save_state(st)
    print(f"espn: {append('espn', rows)} new articles")

# ---------------- NFL.com injury report ----------------
def cmd_nflinj():
    h = get("https://www.nfl.com/injuries/", raw=True)
    if not h: return
    fa = now_iso(); rows = []
    # each team table is preceded by <div class="d3-o-section-sub-title"><span>Nickname</span></div>
    NICK = {v: k for k, v in TEAMS.items()}
    nick2abbr = {name.split()[-1]: ab for name, ab in TEAMS.items()}
    blocks = re.split(r'<table class="d3-o-table', h)
    for i, b in enumerate(blocks[1:]):
        trs = re.findall(r"<tr>(.*?)</tr>", b, re.S)
        m = re.findall(r'd3-o-section-sub-title"><span>([^<]+)</span>', blocks[i])
        team = nick2abbr.get(m[-1].strip(), m[-1].strip()) if m else None
        for tr in trs:
            tds = [re.sub(r"<[^>]+>", " ", t).strip() for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            tds = [re.sub(r"\s+", " ", t) for t in tds]
            if len(tds) >= 5 and tds[0]:
                rows.append({"team": team, "player": tds[0], "pos": tds[1], "injury": tds[2], "practice": tds[3], "game_status": tds[4],
                             "source": "nflinj", "fetched_at": fa})
    print(f"nflinj: {append('nflinj', rows)} rows ({len(set(r['team'] for r in rows))} teams)")

# ---------------- Sleeper trending ----------------
def cmd_sleeper():
    fa = now_iso(); rows = []
    for kind in ("add", "drop"):
        d = get(f"https://api.sleeper.app/v1/players/nfl/trending/{kind}?lookback_hours=24&limit=100") or []
        for r in d: rows.append({"kind": kind, "sleeper_id": r.get("player_id"), "count": r.get("count"), "source": "sleeper", "fetched_at": fa})
    print(f"sleeper: {append('sleeper', rows)} rows")

# ---------------- Wikipedia daily ----------------
def cmd_wiki(start, end, limit=None):
    con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
    titles = con.execute("SELECT player_id, wiki_title FROM nflv_wiki_titles WHERE resolved=1 AND wiki_title IS NOT NULL").fetchall()
    con.close()
    if limit: titles = titles[:int(limit)]
    s, e = start.replace("-", ""), end.replace("-", ""); fa = now_iso(); n = 0
    for k, (pid, t) in enumerate(titles):
        u = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{urllib.parse.quote(t.replace(' ', '_'))}/daily/{s}/{e}"
        d = get(u, retries=2)
        rows = [{"player_id": pid, "title": t, "date": x["timestamp"][:8], "views": x["views"], "source": "wiki", "fetched_at": fa} for x in (d or {}).get("items", [])]
        n += append("wiki", rows); time.sleep(0.1)
        if k % 100 == 0: print(f"  wiki {k}/{len(titles)}")
    print(f"wiki: {n} player-days")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    if cmd == "snapshot":
        cmd_bsky(); cmd_espn(); cmd_nflinj(); cmd_sleeper()
    elif cmd == "backfill-bsky": cmd_bsky(backfill=True)
    elif cmd == "wiki":
        lim = sys.argv[sys.argv.index("--limit") + 1] if "--limit" in sys.argv else None
        cmd_wiki(sys.argv[2], sys.argv[3], lim)
    elif cmd in ("bsky", "espn", "nflinj", "sleeper"): globals()["cmd_" + cmd]()
    else: raise SystemExit(__doc__)
