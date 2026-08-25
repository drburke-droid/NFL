"""
Convert an ESPN Live Draft Trends CSV (Google-Sheets export: Rank / Player /
AVG PICK / 7 DAY +/- / AVG SALARY / 7 DAY +/- / %ROST, player cell =
"Name\\nTEAMPOS") into the paste-format data/espn_trends_2026.txt that
fetch_espn_trends.py and the draft tool's in-browser upload both parse
(rank / doubled-name / name / team / POS / ADP / trend / AVG$ / trend / %rost).

    python scripts/convert_espn_trends_csv.py "C:\\path\\to\\export.csv"
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "espn_trends_2026.txt")
POS2 = ("QB", "RB", "WR", "TE")


def split_teampos(tp):
    tp = tp.strip()
    if tp.endswith("D/ST"):
        return tp[:-4], "D/ST"
    if tp[-2:] in POS2:
        return tp[:-2], tp[-2:]
    if tp.endswith("K"):
        return tp[:-1], "K"
    return None, None


def main():
    src = sys.argv[1]
    rows = list(csv.reader(open(src, encoding="utf-8-sig")))
    out, skipped = [], 0
    for r in rows:
        if len(r) < 7 or not r[0].strip().isdigit() or "\n" not in r[1]:
            continue                                   # page-break header rows etc.
        name, teampos = r[1].split("\n", 1)
        team, pos = split_teampos(teampos)
        pick, sal = r[2].strip(), r[4].strip()
        if not (team and _num(pick) and _num(sal)):
            skipped += 1; continue
        out += [r[0].strip(), name + name, name, team, pos, pick,
                r[3].strip() or "0", sal, r[5].strip() or "0", r[6].strip() or "0"]
    if len(out) < 100:
        raise SystemExit(f"only {len(out)//10} players parsed — wrong CSV?")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"{len(out)//10} players -> {os.path.relpath(OUT, ROOT)} (skipped {skipped})")


def _num(v):
    try:
        float(v); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
