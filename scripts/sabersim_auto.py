"""Unattended SaberSim send: run the generator in the window between official inactives
(T-90 min) and the submission deadline (T-75 min), then email the CSV.

Designed for a scheduler that fires every ~15 minutes (GitHub Actions cron, Task Scheduler,
cron-job.org). Each tick:
  1. --check  : stdlib only, no pip — is the next slate's earliest kickoff 72-90 min away
                and not yet sent? Prints JSON and (in Actions) sets step outputs.
  2. run      : python scripts/sabersim_weekly.py <pkg>  -> CSV for the next slate
  3. email    : SMTP (Gmail app password or any SMTP) with the CSV attached
  4. log      : data/sabersim/sent_log.json (slate key -> sent_at, rows, file) so a second
                tick in the same window is a no-op.

Env (all optional except the key and SMTP creds when actually sending):
  ODDS_API_KEY      written to data/odds_api_key.txt if that file is absent (Actions secret); may hold
                    several keys separated by commas or newlines; the generator rotates on 401/429/0 credits
  MODEL_BURKE_PKG   path to the folder CONTAINING model_burke/  (default ./pkg)
  SMTP_USER / SMTP_PASS   sender login (Gmail: 2-step verification + App Password)
  SMTP_HOST / SMTP_PORT   default smtp.gmail.com / 465 (SSL)
  MAIL_TO           default analysts+robert@sabersim.com ; MAIL_CC optional (yourself)
  PUBLISH_DIR       if set, the CSV is also copied there (the workflow points it at the private repo
                    checkout, pkg/sends, and pushes — the Pages "Run now" button downloads from there)
  CREDIT_WARN       Odds API credits threshold (default 120): below it the send email's subject starts
                    with [LOW ODDS API CREDITS: n]; every internal copy lists credits remaining
  SEND_WINDOW       "72,90" minutes-to-kickoff bounds at check time (default). Inactives post at
                    T-90; the run takes ~2.5 min; SaberSim needs the CSV by T-75, so a check at T-78 or
                    later lands late — the 5-min cron makes that rare, and 72 is the last-resort cutoff.
Usage:
  python scripts/sabersim_auto.py --check
  python scripts/sabersim_auto.py            # check + run + email (no-op outside the window)
  python scripts/sabersim_auto.py --force    # ignore window + sent log
  python scripts/sabersim_auto.py --dry-run  # everything but the email
  python scripts/sabersim_auto.py --force --dry-run -- --no-market   # generator flags after --
"""
import os, sys, json, argparse, subprocess, smtplib, ssl, urllib.request
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")
from email.message import EmailMessage
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true"); ap.add_argument("--force", action="store_true")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--pkg", default=os.environ.get("MODEL_BURKE_PKG", os.path.join(ROOT, "pkg")))
ap.add_argument("gen_args", nargs="*", help="extra args for sabersim_weekly.py after --  (e.g. -- --no-market)")
A = ap.parse_args()
LOG = os.path.join(ROOT, "data", "sabersim", "sent_log.json"); os.makedirs(os.path.dirname(LOG), exist_ok=True)
KEYF = os.path.join(ROOT, "data", "odds_api_key.txt")
if not os.path.exists(KEYF) and os.environ.get("ODDS_API_KEY"):   # the secret may hold several keys, comma/newline separated
    import re as _re
    open(KEYF, "w").write("\n".join(k for k in _re.split(r"[,;\s]+", os.environ["ODDS_API_KEY"]) if k) + "\n")
lo, hi = (float(x) for x in os.environ.get("SEND_WINDOW", "72,90").split(","))

def out(**kw):
    print(json.dumps({k: v for k, v in kw.items() if k != "log_tail"}))
    if kw.get("log_tail"): print("--- generator log tail ---\n" + kw["log_tail"])
    if os.environ.get("GITHUB_OUTPUT"):      # scalar, single-line values only
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for k, v in kw.items():
                if isinstance(v, (bool, int, float)) or (isinstance(v, str) and "\n" not in v and len(v) < 200):
                    f.write(f"{k}={json.dumps(v) if not isinstance(v, str) else v}\n")

def credits():
    """Odds API credits left, summed over every key in the key file (one free /events call per key)."""
    keys = [k.strip() for k in open(KEYF) if k.strip() and not k.startswith("#")] if os.path.exists(KEYF) else []
    if not keys: return None, None, []
    left = used = 0; per = []
    for k in keys:
        try:
            with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={k}", timeout=30) as r:
                h = r.headers
            l, u = int(float(h.get("x-requests-remaining", 0))), int(float(h.get("x-requests-used", 0)))
        except Exception: l, u = 0, 0
        left += l; used += u; per.append(l)
    return left, used, per

def next_slate():
    keys = [k.strip() for k in open(KEYF) if k.strip() and not k.startswith("#")] if os.path.exists(KEYF) else []
    ev = None
    for k in keys:
        try:
            with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={k}", timeout=30) as r:
                ev = json.load(r); break
        except Exception as e: err = str(e)[:60]
    if ev is None: return None, f"events fetch failed ({err if keys else 'no key'})"
    now = datetime.now(timezone.utc)
    up = sorted(((datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")), e) for e in ev
                 if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > now), key=lambda x: x[0])
    if not up: return None, "no upcoming events"
    k0 = up[0][0]; slate = [e for t, e in up if t <= k0 + timedelta(minutes=90)]
    key = k0.strftime("%Y-%m-%dT%H:%M") + f"_{len(slate)}g"
    return {"kick": k0, "minutes_to": (k0 - now).total_seconds() / 60, "games": len(slate), "key": key,
            "label": k0.astimezone(ET).strftime("%a %m/%d %I:%M %p ET") + f" slate ({len(slate)} games)"}, None

sent = json.load(open(LOG)) if os.path.exists(LOG) else {}
s, err = next_slate()
if err: out(in_window=False, reason=err); sys.exit(0)
already = s["key"] in sent
in_win = lo <= s["minutes_to"] <= hi
out(in_window=bool(in_win and not already) or A.force, minutes_to=round(s["minutes_to"], 1), slate=s["label"], slate_key=s["key"], already_sent=already)
if A.check or not (A.force or (in_win and not already)): sys.exit(0)

# ---- run the generator (quiet: its stdout goes to a local log, not the scheduler's log) ----
stamp = datetime.now(timezone.utc).strftime("%m%d_%H%M")
csv_path = os.path.join(ROOT, "outputs", "sabersim", f"Burke_Model_Burke_{s['kick']:%Y}_{s['kick'].astimezone(ET):%a%I%p}_{stamp}.csv".lower().replace("burke_model_burke", "Burke_Model_Burke"))
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
logp = csv_path.replace(".csv", ".log")
with open(logp, "w", encoding="utf-8") as lf:
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "sabersim_weekly.py"), A.pkg, "--out", csv_path] + A.gen_args,
                         stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
if rc != 0 or not os.path.exists(csv_path):
    txt = open(logp, encoding="utf-8", errors="ignore").read()
    tail = txt[txt.rfind("Traceback"):] if "Traceback" in txt else txt[-1500:]
    out(sent=False, reason=f"generator exit {rc}", log_tail=tail[-3000:]); sys.exit(1)
n_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
if os.environ.get("PUBLISH_DIR"):          # e.g. the private repo checkout: pkg/sends
    import shutil; os.makedirs(os.environ["PUBLISH_DIR"], exist_ok=True)
    shutil.copy(csv_path, os.path.join(os.environ["PUBLISH_DIR"], os.path.basename(csv_path)))
left, used, per_key = credits(); warn = int(os.environ.get("CREDIT_WARN", "120"))
low = left is not None and left < warn

# ---- email ----
user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
to = os.environ.get("MAIL_TO", "analysts+robert@sabersim.com"); cc = os.environ.get("MAIL_CC", "")
if A.dry_run or not (user and pw):
    out(sent=False, dry_run=True, file=os.path.basename(csv_path), rows=n_rows, credits_left=left, low_credits=low, reason="dry run" if A.dry_run else "SMTP_USER/SMTP_PASS not set")
    sys.exit(0)
msg = EmailMessage()
msg["From"] = user; msg["To"] = to
if cc: msg["Cc"] = cc
msg["Subject"] = (f"[LOW ODDS API CREDITS: {left}] " if low else "") + f"Model_Burke projections — {s['label']}"
body = f"Model_Burke v1 projections for the {s['label']}.\n{n_rows} players, generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.\n\nRobert Burke"
ops = f"Odds API credits remaining: {left if left is not None else 'unknown'} across {len(per_key)} key(s) {per_key}; used {used}; warning threshold {warn}."
external = to.strip().lower() == "analysts+robert@sabersim.com"
if not external: body += "\n\n--\n" + ops            # test phase: the send itself comes to you, ops line included
msg.set_content(body)
with open(csv_path, "rb") as f:
    msg.add_attachment(f.read(), maintype="text", subtype="csv", filename=os.path.basename(csv_path))
host, port = os.environ.get("SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("SMTP_PORT", "465"))
with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as smtp:
    smtp.login(user, pw); smtp.send_message(msg)
    if external and cc:                       # live phase: SaberSim gets a clean mail, you get a receipt with the ops line
        rcpt = EmailMessage(); rcpt["From"] = user; rcpt["To"] = cc
        rcpt["Subject"] = (f"[LOW ODDS API CREDITS: {left}] " if low else "") + f"[sent to SaberSim] {s['label']} — {n_rows} rows"
        rcpt.set_content(f"Sent {os.path.basename(csv_path)} to {to} at {datetime.now(timezone.utc):%H:%M} UTC.\n{ops}")
        smtp.send_message(rcpt)
sent[s["key"]] = {"sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": n_rows, "file": os.path.basename(csv_path), "slate": s["label"], "credits_left": left}
json.dump(sent, open(LOG, "w"), indent=1)
out(sent=True, file=os.path.basename(csv_path), rows=n_rows, to=to, credits_left=left, low_credits=low)
