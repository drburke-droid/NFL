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
def fetch(label, url, hdr=None):
    s = time.time()
    r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr or {}), timeout=60)
    b = r.read(); h = dict(r.headers)
    print(f"  {label:<26} {len(b)/1e6:>5.1f}MB  {(time.time()-s)*1000:>6.0f}ms  "
          f"Age={h.get('Age','-'):<6} ETag={h.get('ETag','-')[:22]}  sha={hashlib.sha256(b).hexdigest()[:12]}")
    return h.get("ETag", ""), hashlib.sha256(b).hexdigest()

print("Sleeper players feed — is the CDN the bottleneck?")
e1, s1 = fetch("plain", URL)
e2, s2 = fetch("cache-buster ?t=", f"{URL}?t={int(time.time())}")
e3, s3 = fetch("no-cache header", URL, {"Cache-Control": "no-cache", "Pragma": "no-cache"})
print()
same = (s1 == s2 == s3)
print(f"  all three byte-identical: {same}")
print("  verdict:", "CDN is NOT the bottleneck — the origin file itself is what it is; "
      "busting the cache buys nothing" if same else
      "the forced fetch returned DIFFERENT bytes — the cache was serving us stale data, "
      "and cache-busting is worth real minutes")
