"""Parse a Subvertadown paste (all tables) into a tidy point-in-time table, and derive the
K / D-ST override CSV the SaberSim generator reads.

Input: the site's tables copied as text (any subset, any order). Two kinds are recognised:
  projections   header "Player/Team" then 1..17, AVG, Playoffs; rows "Jaguars 10.0 2.7 …" (D/ST)
                or "Cameron Dicker | LAC 10.0 9.8 …" (K / QB); "-" = bye
  matchup bonus header adds "Baseline" and "Next 4"; each team spans four lines
                (name / baseline / next4 / weekly values). Tables are labelled by position in the
                order they appear among the bonus tables (default WR, RB, TE; --bonus-order).
The old short K/DST format ("Jaguars vs. Browns 10.0", "Cam Little JAC @ CIN 8.1") still works.

Outputs:
  data/subvertadown/subvertadown_long.csv   append-only: snapshot, season, week_of_paste, table,
                                            team, player, week, value  (+ baseline/next4/avg/playoffs
                                            as week = "baseline"/"next4"/"avg"/"playoffs")
  data/kdst/kdst_{S}_wk{W}.csv              position, team, player, proj for week W (K + DST tables)
Usage:  python scripts/parse_subvertadown.py <season> <week> [paste.txt] [--bonus-order WR,RB,TE]
        default paste = data/subvertadown/raw/paste_{S}_wk{W}.txt, else data/kdst/paste_{S}_wk{W}.txt
"""
import os, re, sys, csv
from datetime import date
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
S, W = int(args[0]), int(args[1])
order = (sys.argv[sys.argv.index("--bonus-order") + 1] if "--bonus-order" in sys.argv else "WR,RB,TE").split(",")
src = args[2] if len(args) > 2 else next((p for p in (os.path.join(ROOT, "data", "subvertadown", "raw", f"paste_{S}_wk{W}.txt"),
                                                       os.path.join(ROOT, "data", "kdst", f"paste_{S}_wk{W}.txt")) if os.path.exists(p)), None)
if not src: raise SystemExit("no paste file found")
NICK = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI",
    "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC",
    "Rams": "LA", "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF", "Seahawks": "SEA", "Buccaneers": "TB",
    "Titans": "TEN", "Commanders": "WAS"}
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "WSH": "WAS"}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
text = open(src, encoding="utf-8").read()
rows = []           # long rows
kd = []             # k/dst override rows for week W

def is_num_tok(t): return t == "-" or NUM.match(t) is not None

if "Player/Team" not in text:                       # ---- old short K/DST format ----
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        m = re.match(r"^(\S+)\s+(vs\.|@)\s+(\S+)\s+([\d.]+)$", line)
        if m and m.group(1) in NICK:
            kd.append({"position": "DST", "team": NICK[m.group(1)], "player": m.group(1), "proj": float(m.group(4))}); continue
        m = re.match(r"^(.+?)\s+([A-Z]{2,3})\s+(vs\.|@)\s+([A-Z]{2,3})\s+([\d.]+)$", line)
        if m: kd.append({"position": "K", "team": FIX.get(m.group(2), m.group(2)), "player": m.group(1).strip(), "proj": float(m.group(5))})
else:                                               # ---- full site tables ----
    chunks = re.split(r"Player/Team", text)[1:]
    bonus_i = 0
    for ch in chunks:
        ch = ch.split("Matchup Bonuses can help")[0]
        lines = [l.strip() for l in ch.replace(",", " ").splitlines()]
        lines = [l for l in lines if l and l not in ("AVG", "Playoffs", "Baseline", "Next 4") and not NUM.match(l) or (l and NUM.match(l) and "." in l)]
        # header ints (1..17) were dropped by the NUM filter (no '.'); numeric lines with '.' are data
        is_bonus = "Baseline" in ch and "Next 4" in ch
        recs = []
        i = 0
        while i < len(lines):
            l = lines[i]
            toks = l.split()
            if is_num_tok(toks[0]) and not (len(toks) == 1 and "." in toks[0] and not is_bonus):
                i += 1; continue
            # name line: tokens up to the first numeric token
            k = next((j for j, t in enumerate(toks) if is_num_tok(t)), len(toks))
            name = " ".join(toks[:k]); vals = toks[k:]
            i += 1
            if is_bonus:                            # baseline, next4, then the weekly line
                extra = []
                while i < len(lines) and len(extra) < 2 and NUM.match(lines[i]): extra.append(lines[i]); i += 1
                if i < len(lines) and all(is_num_tok(t) for t in lines[i].split()): vals = lines[i].split(); i += 1
                recs.append((name, extra, vals))
            else:
                recs.append((name, [], vals))
        if not recs: continue
        if is_bonus:
            table = (order[bonus_i] if bonus_i < len(order) else f"bonus{bonus_i}").strip().lower() + "_bonus"; bonus_i += 1
        else:
            has_player = any("|" in n for n, _, _ in recs)
            if not has_player: table = "dst"
            else:
                med = sorted(float(v[0]) for _, _, v in recs if v and v[0] != "-")[len(recs) // 2]
                table = "k" if med < 12 else "qb"
        for name, extra, vals in recs:
            if "|" in name:
                player, team = [x.strip() for x in name.split("|", 1)]; team = FIX.get(team, team)
            else:
                player, team = "", NICK.get(name, name)
            weeks = vals[:17]; tail = vals[17:]
            def add(wk, v):
                if v != "-" and NUM.match(v): rows.append({"snapshot": date.today().isoformat(), "season": S, "week_of_paste": W, "table": table,
                                                         "team": team, "player": player, "week": wk, "value": float(v)})
            for j, v in enumerate(weeks, 1): add(j, v)
            if len(tail) >= 1: add("avg", tail[0])
            if len(tail) >= 2: add("playoffs", tail[1])
            if extra: add("baseline", extra[0]);
            if len(extra) > 1: add("next4", extra[1])
            if table in ("k", "dst") and len(weeks) >= W and weeks[W - 1] != "-":
                kd.append({"position": table.upper(), "team": team, "player": player or NICK.get(name, name) and name, "proj": float(weeks[W - 1])})

# ---- write ----
if rows:
    out = os.path.join(ROOT, "data", "subvertadown", "subvertadown_long.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    if not new:      # idempotent: a paste for this season/week is appended once, however often the runner re-parses it
        with open(out, encoding="utf-8") as f:
            if any(r["season"] == str(S) and r["week_of_paste"] == str(W) for r in csv.DictReader(f)):
                print("long table already has this season/week paste; not appending"); rows = []
if rows:
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"appended {len(rows)} rows to {os.path.relpath(out, ROOT)}: " + ", ".join(f"{t} {n}" for t, n in sorted(Counter(r['table'] for r in rows).items())))
if kd:
    seen = set(); kd2 = []
    for r in kd:
        if (r["position"], r["team"]) not in seen: seen.add((r["position"], r["team"])); kd2.append(r)
    out = os.path.join(ROOT, "data", "kdst", f"kdst_{S}_wk{W}.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["position", "team", "player", "proj"]); w.writeheader(); w.writerows(kd2)
    from collections import Counter
    print(f"wrote {os.path.relpath(out, ROOT)}: {dict(Counter(r['position'] for r in kd2))}")
else:
    print("no K/DST rows for this week in the paste")
