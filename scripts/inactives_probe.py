"""Sample how many players ESPN lists as OUT, once per tick, around the inactives release.

Why this exists: official inactives are released at T-90, but the feeds the generator reads do not
carry them for another ~17 minutes. On 2026-09-13 the 4:25 slate showed 3 OUT at T-74 and 9 at
T-72, and the 1pm slate 8 at T-87.6 and 21 by T-66. Two data points per slate, scraped by hand.
This records the whole curve so the crossover is measured rather than inferred, and so a candidate
faster feed can be scored against the same clock.

Deliberately writes NOTHING to the repo. A commit to main triggers sabersim_send.yml, and a run
queued during the send window is exactly the concurrency hazard that cancelled a dispatch on
2026-09-13. One JSON line per tick to stdout; the Actions log is the store.

stdlib only, one HTTP request, ~1 s. Called from a continue-on-error step so it can never affect
a send.

Usage: python scripts/inactives_probe.py --minutes-to 83.4 --slate 2026-09-14T00:20_1g
"""
import json, argparse, urllib.request
from datetime import datetime, timezone

OUT_WORDS = {"out", "injured reserve", "ir", "suspension", "sus", "pup", "dnr", "nfi", "inactive"}
ap = argparse.ArgumentParser()
ap.add_argument("--minutes-to", type=float, required=True, help="minutes to the slate's first kickoff")
ap.add_argument("--slate", default="", help="slate key, for grouping samples later")
ap.add_argument("--band", default="60,100", help="only sample inside this minutes-to-kickoff band")
A = ap.parse_args()
lo, hi = (float(x) for x in A.band.split(","))
if not (lo <= A.minutes_to <= hi):
    print(json.dumps({"probe": "skip", "minutes_to": A.minutes_to, "reason": "outside band"}))
    raise SystemExit(0)

url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "model-burke-probe"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.load(r)
except Exception as e:
    print(json.dumps({"probe": "error", "minutes_to": A.minutes_to, "error": str(e)[:120]}))
    raise SystemExit(0)

per_team, total, q = {}, 0, 0
for t in data.get("injuries", []):
    team = t.get("abbreviation") or t.get("displayName") or "?"
    n = 0
    for inj in t.get("injuries", []):
        st = str((inj.get("status") or "")).strip().lower()
        if st in OUT_WORDS: n += 1; total += 1
        elif st == "questionable": q += 1
    if n: per_team[team] = n
print(json.dumps({"probe": "sample", "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "slate": A.slate, "minutes_to": round(A.minutes_to, 1),
                  "out_total": total, "questionable_total": q, "teams_with_out": len(per_team),
                  "per_team": dict(sorted(per_team.items()))}, separators=(",", ":")))
