"""Why does ESPN refuse the runners, and can any header or host get through?

On 2026-09-14 both ESPN endpoints returned 403 from GitHub Actions while Sleeper answered
normally from the same runner — with Mozilla/5.0, the exact User-Agent sabersim_weekly.py's
http_json sends. So live_status() has been failing on every send and the lineup status has been
coming from Sleeper alone. This is a diagnostic, not part of the pipeline: it tries a handful of
header profiles and hosts and prints what each one returns, so the question is settled by a status
code rather than by guessing which header ESPN dislikes.

Run it from the espn-diag workflow. stdlib only.
"""
import json, sys, urllib.request, urllib.error
from datetime import datetime, timezone

CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/129.0.0.0 Safari/537.36")
LEAGUE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
CASES = [
    ("A urllib default UA",        LEAGUE, {}),
    ("B Mozilla/5.0 (generator)",  LEAGUE, {"User-Agent": "Mozilla/5.0"}),
    ("C full Chrome UA",           LEAGUE, {"User-Agent": CHROME}),
    ("D Chrome + Accept + Referer", LEAGUE, {"User-Agent": CHROME, "Accept": "application/json, text/plain, */*",
                                             "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.espn.com/"}),
    ("E site.web.api host",        "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
                                   {"User-Agent": CHROME}),
    ("F core API teams/12/injuries",
     "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/12/injuries?limit=200",
     {"User-Agent": CHROME}),
    ("G scoreboard (control)",     "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
                                   {"User-Agent": CHROME}),
    ("H sleeper (control)",        "https://api.sleeper.app/v1/state/nfl", {"User-Agent": CHROME}),
]
print(f"espn diagnostic at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n")
print(f"{'case':30s} {'status':>7} {'bytes':>9}  note")
print("-" * 78)
res = {}
for name, url, hdr in CASES:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=25) as r:
            body = r.read()
            note = ""
            try:
                d = json.loads(body)
                if isinstance(d, dict) and "injuries" in d: note = f"{len(d['injuries'])} teams"
                elif isinstance(d, dict) and "items" in d: note = f"{len(d['items'])} items"
                elif isinstance(d, dict): note = f"keys {list(d)[:3]}"
            except Exception: note = "non-JSON"
            print(f"{name:30s} {200:>7} {len(body):>9}  {note}")
            res[name] = 200
    except urllib.error.HTTPError as e:
        print(f"{name:30s} {e.code:>7} {'-':>9}  {str(e.reason)[:34]}")
        res[name] = e.code
    except Exception as e:
        print(f"{name:30s} {'ERR':>7} {'-':>9}  {str(e)[:34]}")
        res[name] = None
ok = [k for k, v in res.items() if v == 200]
print()
print("reachable:", ", ".join(ok) if ok else "NONE")
espn_ok = [k for k in ok if not k.startswith("H")]
print("verdict:", "ESPN reachable — a header/host change fixes the generator"
      if espn_ok else
      "every ESPN host refused while Sleeper answered — this is IP-level, headers will not fix it")

# What does a core-API team injuries item actually look like? espn_team in inactives_probe.py
# counted 0 OUT for KC while the league digest showed 9, so its parse is reading the wrong shape.
try:
    with urllib.request.urlopen(urllib.request.Request(
            "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/12/injuries?limit=200",
            headers={}), timeout=25) as r:
        items = json.loads(r.read()).get("items", [])
    print("\ncore API item shape:", json.dumps(items[0], indent=2)[:400] if items else "no items")
except Exception as e:
    print("\ncore API shape probe failed:", e)
