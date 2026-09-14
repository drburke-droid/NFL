"""Commit the send log and grade outputs onto a main that may have moved underneath us.

The 2026-09-13 19:15 send emailed its CSV to SaberSim and then failed this step. `git pull
--rebase` hit a conflict in sent_log.json, left the rebase half-applied, and the `&& git push`
never ran, so the run went red. I read that red as "the send failed" and told Robert SaberSim had
not received the 9-OUT file; he found it in his Gmail sent box. The email goes out two steps
earlier, so a failure here can never mean the send failed — and the old one-liner gave no way to
tell the two apart.

The conflict was not a real one. sent_log.json is a dict keyed by slate, so two runs recording
different slates disagree about nothing; only the surrounding JSON punctuation collides, and a
text rebase cannot see that. So merge it by key and replay, the same approach
sabersim_scenarios.yml already uses for its generated file, rather than rebasing text.

Everything else in the list is regenerated wholesale each run (grade outputs, idempotently parsed
pastes), so on a replay ours simply wins.

Usage: python scripts/record_send.py -m "message" path [path ...]
"""
import argparse, json, os, subprocess, sys, time

MERGE_BY_KEY = {"data/sabersim/sent_log.json"}   # dict keyed by slate; union, ours wins per key

def git(*a, check=True, quiet=False):
    r = subprocess.run(("git",) + a, capture_output=True, text=True)
    if not quiet and r.stdout.strip(): print(r.stdout.strip())
    if check and r.returncode: raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()[:300]}")
    return r

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--message", required=True)
ap.add_argument("--attempts", type=int, default=5)
ap.add_argument("--remote", default="origin")
ap.add_argument("paths", nargs="+")
A = ap.parse_args()

branch = git("rev-parse", "--abbrev-ref", "HEAD", quiet=True).stdout.strip()
if branch in ("", "HEAD"):
    print("record: detached HEAD, nothing to push to"); sys.exit(0)

def stage_and_commit():
    for p in A.paths:
        git("add", "--", p, check=False, quiet=True)   # globs may match nothing; that is fine
    if not git("diff", "--cached", "--quiet", check=False, quiet=True).returncode:
        return False
    git("commit", "-m", A.message, quiet=True)
    return True

if not stage_and_commit():
    print("record: nothing to commit"); sys.exit(0)

def fail(why):
    """Never let this step look like a failed send. The email goes out two steps earlier."""
    print(f"record: {why}")
    print("record: the CSV was emailed to SaberSim two steps earlier — what failed here is the")
    print("record: LOG WRITE ONLY. A red run at this step does NOT mean the send was missed;")
    print("record: check the Generate + email step and the Gmail sent box before concluding that.")
    sys.exit(1)

for attempt in range(1, A.attempts + 1):
    if git("push", A.remote, f"HEAD:{branch}", check=False, quiet=True).returncode == 0:
        print(f"record: pushed on attempt {attempt}"); sys.exit(0)
    print(f"record: push rejected (attempt {attempt}) — {A.remote}/{branch} moved, replaying")
    try:
        # Capture our versions before rewinding, then rebuild them on top of the remote.
        ours = {p: open(p, "rb").read() for p in A.paths if os.path.isfile(p)}
        git("fetch", A.remote, branch, quiet=True)
        git("reset", "--hard", "FETCH_HEAD", quiet=True)   # leaves untracked files (pkg/) alone
        for p, blob in ours.items():
            if p in MERGE_BY_KEY:
                try:
                    base = json.loads(open(p, encoding="utf-8").read()) if os.path.isfile(p) else {}
                    mine = json.loads(blob.decode())
                    if isinstance(base, dict) and isinstance(mine, dict):
                        merged = {**base, **mine}
                        kept = [k for k in base if k not in mine]
                        print(f"record: merged {p} by key — kept {len(kept)} entry(s) from the remote: {kept}")
                        # Slate keys are ISO-prefixed, so sorting keeps the file chronological and
                        # the diffs readable instead of appending replayed entries out of order.
                        merged = {k: merged[k] for k in sorted(merged)}
                        blob = json.dumps(merged, indent=1).encode()   # no trailing \n: matches sabersim_auto.py
                except Exception as e:
                    print(f"record: key-merge of {p} failed ({e}); keeping ours")
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            open(p, "wb").write(blob)
        if not stage_and_commit():
            print("record: replay left nothing to commit"); sys.exit(0)
    except Exception as e:
        fail(f"replay failed on attempt {attempt}: {e}")
    time.sleep(min(2 ** attempt, 16))

fail(f"could not push after {A.attempts} attempts")
