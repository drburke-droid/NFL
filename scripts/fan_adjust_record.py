"""Record Fan Picks submissions (the codes fans send back from docs/fan.html) into a long table.

A code looks like  FAN1.2026.2.<base64url json>  where the json is
    {"n": fan name, "t": submitted-at ISO, "b": bake_id, "c": optional comment, "r": arrow rule,
     "a": [[player_index, stat_index, arrows], ...]}
player_index is the row in docs/fan/proj_<S>_wk<W>.json (the file the fan was looking at), stat_index
indexes that file's stat_keys, arrows is the signed press count (never 0). What a press means depends
on "r" (scripts/fan_rules.py): "u1" = a set amount in the stat's units (half a TD, 10 yards...),
floored at zero; no "r" = the original rule, 10% of the baseline per press, -10 = zero. Decoding
against the frozen weekly file means the recorded row also carries the baseline the fan saw and the
adjusted value the page showed.

Recorded, NOT used: data/fan_adjustments/fan_adjustments_long.csv
    received_at, season, week, fan, submitted_at, comment, player_index, player_id, player, team, pos,
    opp, stat, arrows, pct, baseline, adjusted, proj_pts, rule, delta
(pct = the move as a percent of the baseline under either rule; delta = the move in stat units)
Idempotent per (season, week, fan, submitted_at): re-recording the same code changes nothing.

Usage:
    python scripts/fan_adjust_record.py <code> [<code> ...]
    python scripts/fan_adjust_record.py --file codes.txt        # one code per line (blank / # lines ignored)
    python scripts/fan_adjust_record.py --inbox                  # every *.txt under data/fan_adjustments/inbox/
"""
import os, sys, csv, json, base64, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fan_rules
from datetime import datetime, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "fan_adjustments", "fan_adjustments_long.csv")
FIELDS = ["received_at", "season", "week", "fan", "submitted_at", "bake_id", "comment", "fan_id", "device_id", "fan_team", "player_index", "player_id", "player", "team", "pos",
          "opp", "stat", "arrows", "pct", "baseline", "adjusted", "proj_pts", "rule", "delta"]


def migrate(path):
    """Rewrite an older long table so its header carries every field (fan_id arrived 2026-09-22)."""
    if not os.path.exists(path): return
    with open(path, encoding="utf-8") as fh:
        rd = csv.DictReader(fh); old = rd.fieldnames or []; rows = list(rd)
    if old == FIELDS: return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    print(f"  migrated {os.path.relpath(path, ROOT)} to the current columns")

def decode(code):
    m = re.match(r"^\s*FAN1\.(\d{4})\.(\d{1,2})\.([A-Za-z0-9_\-=]+)\s*$", code)
    if not m: raise ValueError("not a FAN1 code")
    S, W, b = int(m.group(1)), int(m.group(2)), m.group(3)
    b += "=" * (-len(b) % 4)
    j = json.loads(base64.urlsafe_b64decode(b).decode("utf-8"))
    return S, W, j

def rows_for(S, W, j):
    fp = os.path.join(ROOT, "docs", "fan", f"proj_{S}_wk{W}_{j.get('b')}.json") if j.get("b") else ""
    if not fp or not os.path.exists(fp):
        if j.get("b"): print(f"  note: no frozen bake {j.get('b')} for {S} wk{W}; decoding against the week's latest bake")
        fp = os.path.join(ROOT, "docs", "fan", f"proj_{S}_wk{W}.json")
    if not os.path.exists(fp): raise FileNotFoundError(f"no frozen projection file for {S} wk{W} ({fp}); bake it first")
    bake = json.load(open(fp, encoding="utf-8"))
    keys, players = bake["stat_keys"], bake["players"]
    fan = (j.get("n") or "anonymous").strip()[:60]; sub = j.get("t") or ""; com = (j.get("c") or "").strip()[:300]
    fid = str(j.get("u") or "")[:40]      # identity: a hash of username + PIN the page computes (same person, any device)
    did = str(j.get("d") or "")[:40]      # the browser's own random id, kept for the data mining
    fav = str(j.get("f") or "")[:4]       # the team the fan says he knows best: are fans sharper on their own team?
    rule = fan_rules.rule_of(j.get("r"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for a in j.get("a", []):
        try: pi, si, v = int(a[0]), int(a[1]), int(a[2])
        except (TypeError, ValueError, IndexError): continue
        if not (0 <= pi < len(players)) or not (0 <= si < len(keys)) or v == 0 or v > 100: continue
        p = players[pi]; stat = keys[si]
        if stat not in p["stats"]: continue
        base = p["stats"][stat] or 0.0
        # a press past zero means nothing; the page stops there, so a code that goes further is
        # hand-built or from an older page -- keep the arrow, floored at the rule's zero
        v = max(v, fan_rules.min_presses(rule, stat, base))
        if v == 0: continue
        raw = fan_rules.adjusted(rule, stat, base, v); adj = round(raw, 2)
        # pct keeps its old meaning exactly for pct10 codes (10 x presses); under u1 it is the same
        # move expressed as a percent of the baseline, from the unrounded value
        pct = 10 * v if rule == "pct10" else (round(100 * (raw - base) / base, 1) if base else "")
        out.append({"received_at": now, "season": S, "week": W, "fan": fan, "submitted_at": sub, "bake_id": bake.get("bake_id", ""), "comment": com, "fan_id": fid, "device_id": did, "fan_team": fav, "player_index": pi,
                    "player_id": p["id"], "player": p["name"], "team": p["team"], "pos": p["pos"], "opp": p["opp"], "stat": stat,
                    "arrows": v, "pct": pct, "baseline": p["stats"][stat], "adjusted": adj,
                    "proj_pts": p["proj"], "rule": rule, "delta": round(raw - base, 3)})
    return fan, sub, out

codes = []
args = sys.argv[1:]
if "--file" in args:
    codes += [l.strip() for l in open(args[args.index("--file") + 1], encoding="utf-8") if l.strip() and not l.startswith("#")]
if "--inbox" in args:
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "fan_adjustments", "inbox", "*.txt"))):
        codes += [l.strip() for l in open(fp, encoding="utf-8") if l.strip().startswith("FAN1.")]
codes += [a for a in args if a.startswith("FAN1.")]
if not codes: raise SystemExit("no codes given")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
migrate(OUT)
have = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as fh:
        for r in csv.DictReader(fh): have.add((r["season"], r["week"], r["fan"], r["submitted_at"]))
new_rows, skipped = [], 0
for c in codes:
    try:
        S, W, j = decode(c); fan, sub, rows = rows_for(S, W, j)
    except Exception as e:
        print(f"  skipped a code: {e}"); skipped += 1; continue
    if (str(S), str(W), fan, sub) in have: print(f"  already recorded: {fan} {S} wk{W} {sub}"); continue
    have.add((str(S), str(W), fan, sub)); new_rows += rows
    print(f"  {fan}: {len(rows)} arrows for {S} wk{W}" + (f' — "{j.get("c")[:60]}"' if j.get("c") else ""))
if new_rows:
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerows(new_rows)
print(f"recorded {len(new_rows)} rows -> {os.path.relpath(OUT, ROOT)}" + (f" ({skipped} codes skipped)" if skipped else ""))
