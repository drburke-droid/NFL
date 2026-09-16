"""Sample several candidate inactives feeds each tick, so their latency can be compared.

Official inactives are released at T-90. The feed the generator reads — ESPN's league-wide
/injuries aggregate — did not carry them until ~T-73 on 2026-09-13, and the last send that still
lands inside SaberSim's T-75 cutoff must start at T-80. Reading the docs does not settle which
alternative is faster; only sampling them side by side against the same clock does.

Sources sampled (all free, no auth):
  espn_league  site.api.espn.com/.../nfl/injuries       — the current source, the baseline
  espn_team    sports.core.api.espn.com/.../teams/{id}/injuries — per-team; DISABLED as a count,
               it returns $ref stubs rather than inline injuries (see the note in espn_team), so it
               reports its shape only. Only the teams playing inside the band are queried.
  sleeper      api.sleeper.app/v1/players/nfl           — injury_status; already a dependency

Known dead ends, from research on 2026-09-14: ESPN publishes no pregame inactives endpoint — the
core API's roster didNotPlay/active fields are populated from game participation, not from the
inactive list, and a pregame endpoint is a standing unanswered request from its users. Of the
commercial feeds, SportsDataIO is the one that explicitly documents an Inactive flag available
"around 90 minutes before kickoff", but its free tier returns scrambled data, so it cannot be
evaluated without buying it.

Writes nothing to the repo: a commit to main triggers sabersim_send.yml, and a run queued during
the send window is the concurrency hazard that cancelled a dispatch on 2026-09-13. One JSON line
per tick to stdout; the Actions log is the store.

Usage: python scripts/inactives_probe.py --minutes-to 83.4 --slate 2026-09-14T00:20_1g
"""
import json, argparse, time, urllib.request
from datetime import datetime, timezone

OUT_WORDS = {"out", "injured reserve", "ir", "suspension", "sus", "pup", "dnr", "nfi", "inactive"}
# No User-Agent at all, which is what the generator's espn_json now sends. ESPN 403s a browser UA
# from a datacenter IP and serves the urllib default one — measured on a runner 2026-09-14 by
# scripts/espn_diag.py, which is why the earlier "Mozilla/5.0" here (matched to the generator) got
# 403 on every ESPN tick. Sleeper answers either way. Keep this identical to espn_json's headers so
# a refusal here stays evidence about the generator rather than about the probe.
UA = {}
ap = argparse.ArgumentParser()
ap.add_argument("--minutes-to", type=float, required=True)
ap.add_argument("--slate", default="")
ap.add_argument("--band", default="60,100", help="only sample inside this minutes-to-kickoff band")
ap.add_argument("--sources", default="espn_league,espn_team,sleeper")
A = ap.parse_args()
lo, hi = (float(x) for x in A.band.split(","))
if not (lo <= A.minutes_to <= hi):
    print(json.dumps({"probe": "skip", "minutes_to": A.minutes_to, "reason": "outside band"}))
    raise SystemExit(0)

def jget(url, timeout=25):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.load(r)
def is_out(s): return str(s or "").strip().lower() in OUT_WORDS

def espn_league():
    d = jget("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries")
    per = {}
    for t in d.get("injuries", []):
        team = t.get("abbreviation") or t.get("displayName") or "?"
        n = sum(1 for i in t.get("injuries", []) if is_out(i.get("status")))
        if n: per[team] = n
    return {"out": sum(per.values()), "teams": len(per), "per_team": dict(sorted(per.items()))}

def espn_team():
    """Only the teams whose game kicks off inside the band — a handful of requests, not 32."""
    sb = jget("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard")
    now = datetime.now(timezone.utc); ids = {}
    for ev in sb.get("events", []):
        try: k = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except Exception: continue
        mins = (k - now).total_seconds() / 60
        if not (lo <= mins <= hi): continue
        for c in ev.get("competitions", [{}])[0].get("competitors", []):
            t = c.get("team") or {}
            if t.get("id"): ids[t["id"]] = t.get("abbreviation") or t["id"]
    per = {}
    for tid, abbr in ids.items():
        try:
            d = jget(f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{tid}/injuries?limit=200")
        except Exception: continue
        n = 0
        for it in d.get("items", []):
            st = it.get("status") or (it.get("type") or {}).get("name")
            if is_out(st): n += 1
        per[abbr] = {"out": n, "items": len(d.get("items", [])), "inline": any("status" in i or "type" in i for i in d.get("items", []))}
    # The endpoint returns {"$ref": ".../athletes/{id}/injuries/{id}"} stubs, not inline injuries —
    # dumped from a runner 2026-09-14, when this reported 0 OUT for KC while the league digest showed
    # 9. Reading statuses needs one more fetch per item (~57 per team), so this cannot answer the
    # "does per-team refresh before the digest?" question without ~114 requests a tick, which risks
    # the league source that was just recovered. Report the shape instead of a zero that reads like a
    # measurement: out is meaningless while inline is false.
    inline = all(v["inline"] for v in per.values()) if per else False
    return {"usable": inline, "out": sum(v["out"] for v in per.values()) if inline else None,
            "teams_queried": len(ids), "per_team": dict(sorted(per.items()))}

def sleeper():
    # Origin copy, not the edge. Measured 2026-09-16: the CDN served this file with Age up to
    # 555s, and a ?t= param reaches the origin (1530ms vs 156ms) while Cache-Control is ignored.
    # Sampling the cached copy would measure the CDN's refresh, not the feed's.
    d = jget(f"https://api.sleeper.app/v1/players/nfl?t={int(time.time())}", timeout=90)
    per = {}
    for p in (d or {}).values():
        if not isinstance(p, dict) or not is_out(p.get("injury_status")): continue
        t = p.get("team")
        if t: per[t] = per.get(t, 0) + 1
    return {"out": sum(per.values()), "teams": len(per), "per_team": dict(sorted(per.items()))}

FN = {"espn_league": espn_league, "espn_team": espn_team, "sleeper": sleeper}
res = {}
for name in [x.strip() for x in A.sources.split(",") if x.strip()]:
    fn = FN.get(name)
    if not fn: continue
    t0 = datetime.now(timezone.utc)
    try: res[name] = fn()
    except Exception as e: res[name] = {"error": str(e)[:110]}
    res[name]["ms"] = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
print(json.dumps({"probe": "sample", "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "slate": A.slate, "minutes_to": round(A.minutes_to, 1), "sources": res},
                 separators=(",", ":")))
