"""Is the ~T-73 inactives lag Sleeper's CDN cache, or Sleeper's origin data?

The source hunt on 2026-09-16 found api.sleeper.app/v1/players/nfl served with Age: 555 — a copy
nine minutes old — on a 14.7 MB file Sleeper asks you to call at most once a day. If the CDN is the
bottleneck, busting it recovers most of the ~4.5 minute deficit between the T-73 the feed carries
and the T-80 the send must start at. If the origin is simply regenerated slowly, busting the cache
changes nothing and the deficit is real.

The ETag settles it. Force a fresh fetch (cache-buster plus Cache-Control: no-cache); if Age drops
and the ETag is UNCHANGED, we already had the origin's latest and the cache was never the problem.
"""
import urllib.request, time, hashlib

URL = "https://api.sleeper.app/v1/players/nfl"
BODIES = {}
def fetch(label, url, hdr=None):
    s = time.time()
    r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr or {}), timeout=60)
    b = r.read(); h = dict(r.headers)
    print(f"  {label:<26} {len(b)/1e6:>5.1f}MB  {(time.time()-s)*1000:>6.0f}ms  "
          f"Age={h.get('Age','-'):<6} ETag={h.get('ETag','-')[:22]}  sha={hashlib.sha256(b).hexdigest()[:12]}")
    BODIES[label] = b
    return h.get("ETag", ""), hashlib.sha256(b).hexdigest()

print("Sleeper players feed — is the CDN the bottleneck?")
e1, s1 = fetch("plain", URL)
b1 = BODIES["plain"]
e2, s2 = fetch("cache-buster ?t=", f"{URL}?t={int(time.time())}")
b2 = BODIES["cache-buster ?t="]
e3, s3 = fetch("no-cache header", URL, {"Cache-Control": "no-cache", "Pragma": "no-cache"})
print()
print(f"  plain == no-cache header : {s1 == s3}   (if True, Cache-Control is ignored by their CDN)")
print(f"  plain == cache-buster    : {s1 == s2}   (if False, ?t= reaches the origin)")
print("""
  What this can and cannot show. A byte difference on a 14.7 MB dump of every NFL player does NOT
  prove we gained injury information — the blob can differ for unrelated reasons, and when this was
  first run the cached copy was only 9-11 seconds old. The question that matters is whether the
  ORIGIN carries a lineup status the CACHE does not, and that is only answerable on a game day when
  statuses are actually moving. The status-level diff below is the real measurement; on a Tuesday it
  is expected to be empty, and an empty result here means nothing either way.""")
try:
    import json as _j
    a = {k: (v or {}).get("injury_status") for k, v in _j.loads(b1).items()}
    b = {k: (v or {}).get("injury_status") for k, v in _j.loads(b2).items()}
    diff = [(k, a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)]
    print(f"\n  players whose injury_status differs between cached and origin: {len(diff)}")
    for k, x, y in diff[:15]:
        print(f"    {k}: cached={x!r} origin={y!r}")
except Exception as e:
    print("  status diff failed:", str(e)[:70])

