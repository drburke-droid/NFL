"""
Per-position auction price curves from the league's real drafts (outputs/espn_drafts.csv).

For WR / RB / TE / QB, per season and pooled: the top $, the ramp-down by positional rank, and the
rank where prices hit $1 (the replacement level — how many at the position are worth real money).

Open-auction (non-keeper) prices are the clean market signal for TOP $ and RAMP shape (2023 had 0
keepers = a pure full auction). For the REPLACEMENT rank we also count keepers (a stud kept cheap
still occupies a "worth-money" roster spot), using keeper price when present.
"""
import csv, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
SEASONS = ["2023", "2024", "2025"]; POS = ["WR", "RB", "TE", "QB"]
RANKS = [1, 2, 3, 5, 8, 10, 12, 16, 20, 24, 30]
def fbid(r):
    try: return int(float(r["bid"]))
    except: return 0
iskeep = lambda r: str(r["keeper"]).lower() in ("true", "1")
# 12-team starter demand (1QB/2RB/2WR/1TE/1FLEX split ~RB/WR)
START = {"QB": 12, "RB": 30, "WR": 30, "TE": 12}

def curve(pos, season, keepers=False):
    picks = [r for r in rows if r["pos"] == pos and r["season"] == season and (keepers or not iskeep(r))]
    return sorted([fbid(r) for r in picks], reverse=True)

print("=" * 92)
print("OPEN-AUCTION (non-keeper) price by positional rank — top $, ramp, and where it hits $1")
print("=" * 92)
for pos in POS:
    print(f"\n### {pos}   (12-team starter demand ~{START[pos]})")
    print("  year   n>$1   " + "  ".join(f"r{r:<2}" for r in RANKS) + "   last>$1")
    for s in SEASONS:
        c = curve(pos, s)
        npaid = sum(1 for b in c if b > 1)
        cells = "  ".join(f"{(c[r-1] if r-1 < len(c) else 0):>3}" for r in RANKS)
        last = npaid  # rank of the last player above $1
        tag = "  (0 keepers)" if s == "2023" else ""
        print(f"  {s}   {npaid:>3}    {cells}    WR{last}"[:88].replace("WR", pos) + tag)

# pooled recency-weighted average by rank (like the bidcurve, per position)
print("\n" + "=" * 92)
print("POOLED (recency-weighted 2023x1 / 2024x2 / 2025x3) average $ by positional rank")
print("=" * 92)
BW = {"2023": 1, "2024": 2, "2025": 3}
print("  pos   " + "  ".join(f"r{r:<2}" for r in RANKS) + "    ~$1 rank (avg #>$1)")
for pos in POS:
    avg = []
    for r in RANKS:
        num = den = 0
        for s in SEASONS:
            c = curve(pos, s)
            if r - 1 < len(c): num += c[r-1] * BW[s]; den += BW[s]
        avg.append(round(num/den) if den else 0)
    # replacement rank: avg count of players (incl keepers) acquired for >$1
    paid_counts = []
    for s in SEASONS:
        cc = curve(pos, s, keepers=True)
        paid_counts.append(sum(1 for b in cc if b > 1))
    repl = round(sum(paid_counts) / len(paid_counts))
    cells = "  ".join(f"{v:>3}" for v in avg)
    print(f"  {pos:3s}   {cells}    ~{pos}{repl}  (incl keepers)")
print("\nStartable cutoffs the value model uses: QB12 / RB~29 / WR~29 / TE~13;")
print("bench floor currently ramps to $1 by ~1.6x that (QB19 / RB47 / WR47 / TE21).")
