"""Unattended SaberSim send: run the generator once the official inactives have actually reached the
ESPN/Sleeper feeds (a few minutes after their T-90 release) and before the T-75 submission deadline,
then email the CSV.

Designed for a scheduler that fires every ~15 minutes (GitHub Actions cron, Task Scheduler,
cron-job.org). Each tick:
  1. --check  : stdlib only, no pip — is the next slate's earliest kickoff 78-82 min away
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
  MAIL_TO           default analysts+robert@sabersim.com. If MAIL_CC is set, MAIL_TO gets the clean external
                    email and MAIL_CC gets a receipt (ops line + CSV copy); without MAIL_CC the send itself
                    carries the ops line (test phase)
  PUBLISH_DIR       if set, the CSV is also copied there (the workflow points it at the private repo
                    checkout, pkg/sends, and pushes — the Pages "Run now" button downloads from there)
  CREDIT_WARN       Odds API credits threshold (default 120): below it the send email's subject starts
                    with [LOW ODDS API CREDITS: n]; every internal copy lists credits remaining
  SEND_WINDOW       "78,82" minutes-to-kickoff bounds at check time (default): one attempt, at T-80. Inactives are released at
                    T-90, but the feeds lag: the 2026-09-13 T-90 tick emailed 13 inactives short. Kickoffs
                    sit on a 5-minute mark so ticks land at T-90/T-85/T-80/T-75; the 82 cap makes T-80 the
                    first eligible tick (email ~T-77.5, inside the T-75 cutoff), with T-75 as a late
                    fallback. 72 stays the last-resort floor.
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
ap.add_argument("--slate", default="", help="kickoff key of a specific slate to send (or just its ISO "
                "kickoff prefix, e.g. 2026-09-20T20:25). Overrides the automatic pick, which is the "
                "earliest UNSENT slate — with a split late window that is not necessarily the one you "
                "mean. Pair with --force to resend a slate already in the log.")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--pkg", default=os.environ.get("MODEL_BURKE_PKG", os.path.join(ROOT, "pkg")))
ap.add_argument("gen_args", nargs="*", help="extra args for sabersim_weekly.py after --  (e.g. -- --no-market)")
A = ap.parse_args()
LOG = os.path.join(ROOT, "data", "sabersim", "sent_log.json"); os.makedirs(os.path.dirname(LOG), exist_ok=True)
KEYF = os.path.join(ROOT, "data", "odds_api_key.txt")
if not os.path.exists(KEYF) and os.environ.get("ODDS_API_KEY"):   # the secret may hold several keys, comma/newline separated
    import re as _re
    open(KEYF, "w").write("\n".join(k for k in _re.split(r"[,;\s]+", os.environ["ODDS_API_KEY"]) if k) + "\n")
lo, hi = (float(x) for x in os.environ.get("SEND_WINDOW", "78,82").split(","))
# how many keys the run can see and a 4-char fingerprint of each, so the Actions log of every tick answers
# "did the ODDS_API_KEY secret parse into all the keys I pasted?" (scripts/secret_keys_check.py reads it back)
_keys = [k.strip() for k in open(KEYF) if k.strip() and not k.startswith("#")] if os.path.exists(KEYF) else []
KEYS_INFO = {"keys": len(_keys), "key_tails": ",".join(k[-4:] for k in _keys)}

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

GROUP_MIN = float(os.environ.get("SLATE_GROUP_MIN", "15"))

def group_slates(times, now):
    """Upcoming kickoffs bucketed into slates, earliest first.

    Grouped on a 15-minute tolerance, not the old 90. Week 2 is the case that forces it: the late
    window splits 16:05 ET (2 games) and 16:25 ET (3), twenty minutes apart. Under 90 minutes they
    became one slate keyed on 16:05, so the send fired at T-80 for 16:05 — which is T-100 for the
    16:25 games, ten minutes before their inactives are even released. A tolerance still beats an
    exact match, so a feed reporting 13:00 and 13:01 for the same window stays one slate.
    """
    out = []
    for t in sorted(x for x in times if x > now):
        if out and t <= out[-1][0] + timedelta(minutes=GROUP_MIN): out[-1][1] += 1
        else: out.append([t, 1])
    return [{"kick": t, "minutes_to": (t - now).total_seconds() / 60, "games": n,
             "key": t.strftime("%Y-%m-%dT%H:%M") + f"_{n}g",
             "label": t.astimezone(ET).strftime("%a %m/%d %I:%M %p ET") + f" slate ({n} games)"}
            for t, n in out]

def pick_slate(slates, sent, wanted=""):
    """The slate to act on: a named one if asked for, else the earliest not yet sent."""
    if wanted:
        w = wanted.split("_")[0]                     # tolerate a full key or a bare ISO kickoff
        for x in slates:
            if x["key"] == wanted or x["key"].startswith(w): return x, None
        return None, f"no upcoming slate matching {wanted!r}"
    for x in slates:
        if x["key"] not in sent: return x, None
    return dict(slates[0], all_sent=True), None

ALL_SLATES = []

def next_slate(sent, wanted=""):
    """The earliest slate that has NOT been sent yet.

    Returning simply the earliest upcoming slate was the second half of the week-2 problem: once
    16:05 is in the sent log, every later tick still sees 16:05 as earliest, finds it already sent
    and skips — so 16:25 would never get a send of its own, at any point.
    """
    keys = [k.strip() for k in open(KEYF) if k.strip() and not k.startswith("#")] if os.path.exists(KEYF) else []
    ev = err = None
    for k in keys:
        try:
            with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={k}", timeout=30) as r:
                ev = json.load(r); break
        except Exception as e: err = str(e)[:60]
    if ev is None: return None, f"events fetch failed ({err if keys else 'no key'})"
    now = datetime.now(timezone.utc)
    times = [datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) for e in ev]
    sl = group_slates(times, now)
    if not sl: return None, "no upcoming events"
    ALL_SLATES[:] = sl
    return pick_slate(sl, sent, wanted)

sent = json.load(open(LOG)) if os.path.exists(LOG) else {}
s, err = next_slate(sent, A.slate)
if err: out(in_window=False, reason=err, **KEYS_INFO); sys.exit(0)
already = bool(s.get("all_sent")) or s["key"] in sent
in_win = lo <= s["minutes_to"] <= hi
# daily grade: the tick that lands in the first 10 min of GRADE_HOUR UTC (default 14 = 8 am MDT) also
# re-grades finished games (GitHub's own cron never fires for this repo, so the ticks carry it)
_now = datetime.now(timezone.utc); gh_ = int(os.environ.get("GRADE_HOUR", "14"))
grade_due = _now.hour == gh_ and _now.minute < 10
imm = ALL_SLATES[0] if ALL_SLATES else s
out(imminent_minutes_to=round(imm["minutes_to"], 1), imminent_slate=imm["key"],
    in_window=bool(in_win and not already) or A.force, slate_requested=A.slate or None, minutes_to=round(s["minutes_to"], 1), slate=s["label"], slate_key=s["key"], already_sent=already, grade_due=grade_due, **KEYS_INFO)
if A.check or not (A.force or (in_win and not already)): sys.exit(0)

# ---- run the generator (quiet: its stdout goes to a local log, not the scheduler's log) ----
stamp = datetime.now(timezone.utc).strftime("%m%d_%H%M")
csv_path = os.path.join(ROOT, "outputs", "sabersim", f"Burke_Model_Burke_{s['kick']:%Y}_{s['kick'].astimezone(ET):%a%I%p}_{stamp}.csv".lower().replace("burke_model_burke", "Burke_Model_Burke"))
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
logp = csv_path.replace(".csv", ".log")
with open(logp, "w", encoding="utf-8") as lf:
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "sabersim_weekly.py"), A.pkg,
                          "--out", csv_path, "--kickoff", s["kick"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "--slate-tol", str(GROUP_MIN)] + A.gen_args,
                         stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
if rc != 0 or not os.path.exists(csv_path):
    txt = open(logp, encoding="utf-8", errors="ignore").read()
    tail = txt[txt.rfind("Traceback"):] if "Traceback" in txt else txt[-1500:]
    out(sent=False, reason=f"generator exit {rc}", log_tail=tail[-3000:]); sys.exit(1)
n_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
# The generator's stdout goes to csv_path.log, which never reaches the Actions output, so a feed
# quietly failing looked identical to a feed reporting nothing. On 2026-09-14 the probe found ESPN
# returning 403 from the runners while live_status() was catching it and printing to that log.
WARN_MARKERS = ("unavailable", "no DK props posted", "failed", "Traceback")
try:
    _log = open(logp, encoding="utf-8", errors="ignore").read().splitlines()
    warns = [l.strip()[:160] for l in _log if any(m.lower() in l.lower() for m in WARN_MARKERS)][:6]
except Exception:
    warns = []
if warns:
    print("generator warnings:")
    for w in warns: print("   " + w)
# the generator's health record (what FFA / DK lines / DK props / lineups actually delivered) + the wrapper's warnings
health_p = csv_path.replace(".csv", ".health.json")
try: health = json.load(open(health_p, encoding="utf-8"))
except Exception: health = {}
health.update({"warnings": warns, "dry_run": bool(A.dry_run), "slate": s["label"], "slate_key": s["key"],
               "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
json.dump(health, open(health_p, "w", encoding="utf-8"), indent=1, default=str)
if os.environ.get("PUBLISH_DIR"):          # e.g. the private repo checkout: pkg/sends
    import shutil; os.makedirs(os.environ["PUBLISH_DIR"], exist_ok=True)
    # a dry run is a PREVIEW: published so the page can show and download it, but under a name the
    # graders' Burke_Model_Burke_* globs never match, so it is never mistaken for a send
    pub = ("preview_" if A.dry_run else "") + os.path.basename(csv_path)
    shutil.copy(csv_path, os.path.join(os.environ["PUBLISH_DIR"], pub))
    shutil.copy(health_p, os.path.join(os.environ["PUBLISH_DIR"], pub.replace(".csv", ".health.json")))
    if not A.dry_run:
        import glob as _glob          # the generator's full run parquet (FFA baseline, DK market, quantiles) rides along for grading
        runs = sorted(_glob.glob(os.path.join(ROOT, "outputs", "sabersim", "run_*.parquet")), key=os.path.getmtime)
        if runs: shutil.copy(runs[-1], os.path.join(os.environ["PUBLISH_DIR"], os.path.basename(csv_path).replace(".csv", ".parquet")))
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
# Two modes. With MAIL_CC set: MAIL_TO gets the clean external email (what SaberSim sees) and MAIL_CC
# gets a receipt with the ops line + a copy of the CSV. Without MAIL_CC: the send itself carries the ops line.
external = bool(cc)
if not external: body += "\n\n--\n" + ops
msg.set_content(body)
with open(csv_path, "rb") as f:
    msg.add_attachment(f.read(), maintype="text", subtype="csv", filename=os.path.basename(csv_path))
host, port = os.environ.get("SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("SMTP_PORT", "465"))
with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as smtp:
    smtp.login(user, pw); smtp.send_message(msg)
    if external and cc:                       # live phase: SaberSim gets a clean mail, you get a receipt with the ops line
        rcpt = EmailMessage(); rcpt["From"] = user; rcpt["To"] = cc
        rcpt["Subject"] = (f"[LOW ODDS API CREDITS: {left}] " if low else "") + f"[sent to SaberSim] {s['label']} — {n_rows} rows"
        rcpt.set_content(f"Sent {os.path.basename(csv_path)} to {to} at {datetime.now(timezone.utc):%H:%M} UTC ({n_rows} rows).\n{ops}\n\nThe attached CSV is the exact file that went out.")
        with open(csv_path, "rb") as f:
            rcpt.add_attachment(f.read(), maintype="text", subtype="csv", filename=os.path.basename(csv_path))
        smtp.send_message(rcpt)
try:    # the generator's own lineup summary ("lineups: N statuses pulled {espn: .., sleeper: ..}; Q, OUT {..}") and injury-report line
    _keep = [l.strip()[:300] for l in open(logp, encoding="utf-8", errors="ignore").read().splitlines()
             if l.strip().startswith(("lineups:", "injury report", "schedule:", "week ", "ESPN injuries unavailable", "Sleeper unavailable", "NFL.com injury"))][:6]
except Exception:
    _keep = []
sent[s["key"]] = {"sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": n_rows, "file": os.path.basename(csv_path), "slate": s["label"], "credits_left": left,
                  "lineups": _keep, "health": {k: health.get(k) for k in ("ffa", "dk_lines", "dk_props", "kdst", "dk_kickers", "injury_report", "lineups", "warnings") if k in health}}
json.dump(sent, open(LOG, "w"), indent=1)
out(sent=True, file=os.path.basename(csv_path), rows=n_rows, to=to, credits_left=left,
    low_credits=low, generator_warnings=len(warns))
