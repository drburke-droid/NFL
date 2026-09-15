"""Does the ODDS_API_KEY Actions secret hold every key in data/odds_api_key.txt?

Secret values are never readable, so this reads the *evidence*: every SaberSim-send tick on main logs
`"keys": N, "key_tails": "195c,1154,..."` (the last 4 chars of each key the runner parsed out of the
secret).  This script pulls the newest completed tick's log through the GitHub API, prints that line,
and diffs the fingerprints against the local key file.  Also shows credits left per local key.

    python scripts/secret_keys_check.py

Needs: a github.com credential in the git credential store (the one `git push` uses); no gh CLI.
"""
import io, json, os, subprocess, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "drburke-droid/NFL"; WF = "sabersim_send.yml"

def token():
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True, cwd=ROOT).stdout
    for line in out.splitlines():
        if line.startswith("password="): return line[len("password="):]
    raise SystemExit("no github.com credential in the git credential store")

def api(url, tok, raw=False):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if raw else json.load(r)

tok = token()
runs = api(f"https://api.github.com/repos/{REPO}/actions/workflows/{WF}/runs?status=completed&per_page=1", tok)["workflow_runs"]
if not runs: raise SystemExit("no completed runs")
run = runs[0]
z = zipfile.ZipFile(io.BytesIO(api(f"https://api.github.com/repos/{REPO}/actions/runs/{run['id']}/logs", tok, raw=True)))
line = None
for n in z.namelist():
    if "Check send window" in n:
        for l in z.read(n).decode("utf8", "replace").splitlines():
            if '"in_window"' in l: line = l[l.index("{"):]; break
if line is None: raise SystemExit(f"run {run['id']} has no check-step output")
info = json.loads(line)
print(f"latest tick: run {run['id']} at {run['created_at']} ({run['event']})")
if "keys" not in info:
    raise SystemExit("that tick ran a sabersim_auto.py without the keys line yet; wait for the next tick after the merge")
secret_tails = [t for t in info["key_tails"].split(",") if t]
print(f"secret parsed into {info['keys']} key(s): {secret_tails}")

kf = os.path.join(ROOT, "data", "odds_api_key.txt")
local = [k.strip() for k in open(kf) if k.strip() and not k.startswith("#")] if os.path.exists(kf) else []
if not local:
    print("no local data/odds_api_key.txt to compare against"); sys.exit(0)
print(f"\nlocal key file: {len(local)} key(s)")
print(f"{'key':>8} {'in secret':>10} {'remaining':>10} {'used':>6}")
for k in local:
    try:
        with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={k}", timeout=30) as r:
            rem, used = r.headers.get("x-requests-remaining", "?"), r.headers.get("x-requests-used", "?")
    except Exception as e:
        rem, used = f"ERR {getattr(e, 'code', '')}", "-"
    print(f"...{k[-4:]:>5} {'yes' if k[-4:] in secret_tails else 'NO':>10} {rem:>10} {used:>6}")
extra = [t for t in secret_tails if t not in {k[-4:] for k in local}]
missing = [k[-4:] for k in local if k[-4:] not in secret_tails]
print()
if not missing and not extra: print(f"OK: the secret holds all {len(local)} local keys")
else:
    if missing: print(f"MISSING from the secret: {missing}  -> paste them into ODDS_API_KEY (one per line)")
    if extra: print(f"in the secret but not in the local file: {extra}")
