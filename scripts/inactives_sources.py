"""Hunt for a faster free inactives source, tested against a game whose inactives are already known.

The problem this is for: official inactives post at T-90, the feed we read carries them ~T-73, and
the last send that lands inside SaberSim's T-75 cutoff must start at T-80. Week 1 showed the gap is
not injuries — 74 of the 82 skill players who never took a snap were not on the Friday injury
report at all. They are healthy scratches, and only an actual inactives list shows them.

Two hypotheses worth separating:
  H1  Sleeper's lag is its own cache, not the NFL's release. api.sleeper.app/v1/players/nfl is a
      ~5MB dump Sleeper asks you to call at most once a day; if it regenerates slowly, ~T-73 is
      Sleeper's refresh cadence rather than reality. Last-Modified/Age headers say which.
  H2  ESPN's per-event competitor roster carries an explicit `active` flag. Earlier research in
      this repo concluded ESPN has no pregame inactives endpoint because didNotPlay/active are
      participation-based — but that was read, not measured.

Testing against a COMPLETED game is the point: if an endpoint can mark the players who really were
inactive, the data exists, and Sunday's probe only has to measure WHEN it appears. An endpoint that
cannot even do it in hindsight is not worth timing.

Usage: python scripts/inactives_sources.py [--week 1]
"""
import argparse, json, re, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--week", type=int, default=1)
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--games", type=int, default=2, help="how many completed games to inspect")
A = ap.parse_args()

# No User-Agent: ESPN 403s a browser UA from a datacenter IP and serves the urllib default one
# (scripts/espn_diag.py, measured on a runner 2026-09-14).
def get(url, raw=False, hdr=None):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr or {}), timeout=30)
    b = r.read()
    return (b, dict(r.headers)) if raw else json.loads(b)

print(f"=== H1: is Sleeper's players dump stale by design? ===")
try:
    b, h = get("https://api.sleeper.app/v1/players/nfl", raw=True)
    for k in ("Last-Modified", "Age", "Date", "Cache-Control", "ETag", "CF-Cache-Status", "X-Cache"):
        if k in h: print(f"  {k}: {h[k]}")
    print(f"  size: {len(b)/1e6:.1f} MB")
except Exception as e:
    print("  failed:", str(e)[:70])

print(f"\n=== H2: does any ESPN endpoint expose an explicit active/inactive flag? ===")
try:
    sb = get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={A.week}")
    evs = [e for e in sb.get("events", []) if e.get("status", {}).get("type", {}).get("completed")]
    print(f"  completed week-{A.week} games found: {len(evs)}")
    for ev in evs[:A.games]:
        print(f"\n  --- {ev['shortName']} (event {ev['id']}) ---")
        # (a) the site summary's roster block
        try:
            s = get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={ev['id']}")
            ros = s.get("rosters") or []
            print(f"    summary.rosters: {len(ros)} teams", end="")
            if ros:
                ent = ros[0].get("roster") or []
                keys = sorted({k for e in ent[:40] for k in e})
                print(f", {len(ent)} entries, keys {keys[:8]}")
                for flag in ("active", "didNotPlay", "starter"):
                    vals = [e.get(flag) for e in ent if flag in e]
                    if vals:
                        print(f"      {flag}: present on {len(vals)}/{len(ent)}; False on {sum(v is False for v in vals)}")
            else:
                print(" — empty")
        except Exception as e:
            print(f"    summary failed: {str(e)[:60]}")
        # (b) the core API per-competitor roster
        try:
            comp = ev["competitions"][0]
            for c in comp.get("competitors", []):
                u = (f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{ev['id']}"
                     f"/competitions/{comp['id']}/competitors/{c['id']}/roster?limit=200")
                d = get(u)
                items = d.get("items", [])
                inline = [i for i in items if isinstance(i, dict) and "active" in i]
                print(f"    core roster {c.get('team',{}).get('abbreviation','?')}: {len(items)} items, "
                      f"{len(inline)} with an 'active' key"
                      + (f", inactive={sum(i.get('active') is False for i in inline)}" if inline else
                         f"  (keys: {sorted(items[0])[:5] if items else 'none'})"))
        except Exception as e:
            print(f"    core roster failed: {str(e)[:60]}")
except Exception as e:
    print("  scoreboard failed:", str(e)[:70])

print(f"\n=== H3: does nfl.com publish the inactives list on the game page? ===")
try:
    b, _ = get(f"https://www.nfl.com/scores/{A.season}/reg{A.week}", raw=True, hdr={"User-Agent": "Mozilla/5.0"})
    html = b.decode("utf8", "replace")
    hits = len(re.findall(r"(?i)inactive", html))
    print(f"  scores page: {len(html)/1000:.0f} KB, the word 'inactive' appears {hits}x")
except Exception as e:
    print("  nfl.com failed:", str(e)[:70])
