"""Parse a pasted K / D-ST projection list (Subvertadown-style) into the override CSV
the SaberSim generator reads:  data/kdst/kdst_{S}_wk{W}.csv  (position, team, player, proj)

Paste format, one per line (the site's layout collapsed to single lines is fine):
  D/ST : "<Nickname> vs. <Nickname> 7.4"  or  "<Nickname> @ <Nickname> 5.9"
  K    : "<First Last> <TEAM> vs. <TEAM> 9.3"  /  "<First Last> <TEAM> @ <TEAM> 8.1"
Usage: python scripts/parse_kdst_paste.py <season> <week> [paste.txt]
"""
import os, re, sys, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S, W = int(sys.argv[1]), int(sys.argv[2])
src = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "data", "kdst", f"paste_{S}_wk{W}.txt")
NICK = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR",
    "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN",
    "Lions": "DET", "Packers": "GB", "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX",
    "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG", "Jets": "NYJ",
    "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF", "Seahawks": "SEA", "Buccaneers": "TB",
    "Titans": "TEN", "Commanders": "WAS"}
FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "WSH": "WAS"}
rows = []
for line in open(src, encoding="utf-8"):
    line = line.strip()
    if not line: continue
    m = re.match(r"^(\S+)\s+(vs\.|@)\s+(\S+)\s+([\d.]+)$", line)          # D/ST
    if m and m.group(1) in NICK:
        rows.append({"position": "DST", "team": NICK[m.group(1)], "player": m.group(1),
                     "proj": float(m.group(4))}); continue
    m = re.match(r"^(.+?)\s+([A-Z]{2,3})\s+(vs\.|@)\s+([A-Z]{2,3})\s+([\d.]+)$", line)  # K
    if m:
        t = FIX.get(m.group(2), m.group(2))
        rows.append({"position": "K", "team": t, "player": m.group(1).strip(), "proj": float(m.group(5))}); continue
    print("  unparsed:", line)
df = pd.DataFrame(rows).drop_duplicates(["position", "team"])
out = os.path.join(ROOT, "data", "kdst", f"kdst_{S}_wk{W}.csv")
df.to_csv(out, index=False)
print(f"wrote {os.path.relpath(out, ROOT)}: {df.position.value_counts().to_dict()}")
