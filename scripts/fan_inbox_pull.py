"""Pull Fan Picks submissions out of the drop box (a Google Form's response sheet) into the inbox.

The game page (docs/fan.html) posts every submission -- username + FAN1 code -- to a Google Form
in the background, so fans need no account and nobody has to forward emails. The form's linked
sheet is the inbox. This reads it, writes every code into data/fan_adjustments/inbox/dropbox.txt
(the recorder is idempotent per fan + submitted-at, so re-pulling is free), bakes a small public
summary of who is in (docs/fan/submissions.json, what the page shows before games are graded),
and optionally records and grades in the same breath.

Config: docs/fan/dropbox.json, shared with the page:
    {"form_action": "https://docs.google.com/forms/d/e/<id>/formResponse",
     "entry_user": "entry.NNN", "entry_code": "entry.MMM",
     "sheet_csv": "https://docs.google.com/spreadsheets/d/<id>/gviz/tq?tqx=out:csv&sheet=Form%20Responses%201"}
Until it is filled in the page falls back to the old email/copy flow and this script does nothing.

    python scripts/fan_inbox_pull.py [--record] [--grade]
"""
import argparse, base64, csv, io, json, os, re, subprocess, sys, urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "docs", "fan", "dropbox.json")
INBOX = os.path.join(ROOT, "data", "fan_adjustments", "inbox", "dropbox.txt")
SUBS = os.path.join(ROOT, "docs", "fan", "submissions.json")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def decode_head(code):
    """(season, week, name, submitted_at, fan_id, n_arrows) from a code, without the bake."""
    m = re.match(r"^\s*FAN1\.(\d{4})\.(\d{1,2})\.([A-Za-z0-9_\-=]+)\s*$", code)
    if not m:
        return None
    b = m.group(3); b += "=" * (-len(b) % 4)
    j = json.loads(base64.urlsafe_b64decode(b).decode("utf-8"))
    return int(m.group(1)), int(m.group(2)), (j.get("n") or "anonymous").strip()[:60], j.get("t") or "", \
        (j.get("u") or "")[:40], len(j.get("a") or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="run fan_adjust_record.py --inbox afterwards")
    ap.add_argument("--grade", action="store_true", help="run fan_grade.py afterwards")
    ap.add_argument("--sends", nargs="*", default=None, help="passed through to fan_grade.py")
    a = ap.parse_args()
    cfg = json.load(open(CFG, encoding="utf-8")) if os.path.exists(CFG) else {}
    url = cfg.get("sheet_csv")
    if not url:
        print("no drop box configured (docs/fan/dropbox.json has no sheet_csv) -- nothing to pull")
        return 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fan picks pull)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        print("empty sheet"); return 0
    head = [h.strip().lower() for h in rows[0]]
    def col(*names):
        for i, h in enumerate(head):
            if any(n in h for n in names):
                return i
        return None
    ci, cu, ct = col("code"), col("user", "name", "handle"), col("timestamp", "time")
    if ci is None:
        print(f"no code column in the sheet header {rows[0]}"); return 1
    codes, subs = [], []
    for r in rows[1:]:
        if len(r) <= ci or not r[ci].strip():
            continue
        code = r[ci].strip()
        h = decode_head(code)
        if not h:
            continue
        S, W, name, sub, fid, n = h
        if n == 0:
            continue                      # a setup test or an empty submission: nothing to grade, nobody to list
        if cu is not None and len(r) > cu and r[cu].strip():
            name = r[cu].strip()[:60]
        codes.append(code)
        subs.append({"season": S, "week": W, "fan": name, "submitted_at": sub, "arrows": n,
                     "received": r[ct].strip() if ct is not None and len(r) > ct else ""})
    os.makedirs(os.path.dirname(INBOX), exist_ok=True)
    with open(INBOX, "w", encoding="utf-8") as fh:
        fh.write("# pulled from the Fan Picks drop box; the recorder is idempotent\n" + "\n".join(codes) + "\n")
    # who is in, latest submission per fan per week (the page shows this before anything is graded)
    latest = {}
    for s in subs:
        k = (s["season"], s["week"], s["fan"])
        if k not in latest or s["submitted_at"] > latest[k]["submitted_at"]:
            latest[k] = s
    entries = sorted(latest.values(), key=lambda s: (-s["week"], s["submitted_at"]))
    json.dump({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "n_codes": len(codes),
               "entries": entries}, open(SUBS, "w", encoding="utf-8"), indent=1)
    print(f"pulled {len(codes)} codes from the drop box; {len(entries)} fan-weeks -> {os.path.relpath(SUBS, ROOT)}")
    rc = 0
    if a.record and codes:
        rc |= subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fan_adjust_record.py"), "--inbox"], cwd=ROOT).returncode
    if a.grade:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "fan_grade.py")]
        if a.sends:
            cmd += ["--sends"] + a.sends
        rc |= subprocess.run(cmd, cwd=ROOT).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
