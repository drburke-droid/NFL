"""
Does a multi-book CONSENSUS prop line beat DraftKings' own line?  (home-PC job: needs db/nfl_odds.db)

Why: the SaberSim send blends DraftKings player props into the projection. The Odds API bills an
event-odds call per market per "region", and up to 10 named bookmakers count as ONE region, so
pulling FanDuel, BetMGM, Caesars, ... alongside DraftKings costs no more credits. Worth doing only
if the consensus line is closer to what happens than DK's line alone. book_sharpness_study.py
ranked books against the average BOOK error; this asks the question the send needs answered.

For each prop market, each book's last pre-kickoff line (the same closing snapshot the prop
frames use), rows with a DK line and at least MIN_BOOKS books quoting it, scored against the
player's actual stat (nflverse weekly, played rows only):

  dk        |actual - DK line|
  cons      |actual - median line over the 10 most-quoted books|   (what one call could fetch)
  dk_adj    DK line moved by its own devigged over/under price     (same number, different juice)
  cons_adj  median of the books' price-adjusted lines

Price adjustment: a book at 25.5 with the over at -140 expects more than 25.5. The shift is
sd * z(p_over), with p_over devigged from the book's two prices and sd the market's residual
spread (a normal approximation; it only has to rank the books, not be exact).

Paired differences (cons - dk) with standard errors, by market and season; negative = the
consensus is closer. Report: outputs/reports/consensus_vs_dk.md

    python scripts/consensus_vs_dk_study.py [--db db/nfl_odds.db]
"""
import argparse, os, re, sqlite3, sys
import numpy as np
import pandas as pd
from scipy.stats import norm as N

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--db", default=os.path.join(ROOT, "db", "nfl_odds.db"))
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "consensus_vs_dk.md"))
ap.add_argument("--min-books", type=int, default=4)
A = ap.parse_args()

MARKETS = {"player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds", "player_rush_yds": "rushing_yards",
           "player_reception_yds": "receiving_yards", "player_receptions": "receptions"}
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _out.append(s)
def devig(po, pu):
    """two American prices -> fair P(over)"""
    imp = lambda p: np.where(p > 0, 100 / (p + 100), -p / (-p + 100))
    a, b = imp(np.asarray(po, float)), imp(np.asarray(pu, float))
    return a / (a + b)

con = sqlite3.connect(A.db)
games = pd.read_sql("SELECT event_id, commence_time, season, week FROM games", con)
for c in ("season", "week"): games[c] = pd.to_numeric(games[c], errors="coerce")
games = games.dropna(subset=["season", "week"]).astype({"season": int, "week": int})
wk = pd.read_sql(f"SELECT player_display_name, season, week, {', '.join(sorted(set(MARKETS.values())))} FROM nflv_weekly", con)
wk["nm"] = wk.player_display_name.map(norm); wk = wk.drop_duplicates(["nm", "season", "week"])

say(f"# Consensus prop line vs DraftKings ({pd.Timestamp.now():%Y-%m-%d})\n")
say("Each book's last pre-kickoff line; rows with a DK line and at least "
    f"{A.min_books} books; error = |actual stat - line|. **cons - dk < 0 means the consensus is closer.**\n")
summary = []
for mkt, stat in MARKETS.items():
    pp = pd.read_sql(f"""SELECT event_id, bookmaker, player_name, outcome_type, price, point, snapshot_time
                         FROM player_props WHERE market='{mkt}' AND point IS NOT NULL""", con)
    pp = pp.merge(games, on="event_id")
    pp = pp[pp.snapshot_time <= pp.commence_time]
    pp["rk"] = pp.groupby(["event_id", "bookmaker", "player_name", "outcome_type"]).snapshot_time.rank(ascending=False, method="first")
    pp = pp[pp.rk == 1]
    ov = pp[pp.outcome_type == "Over"][["event_id", "bookmaker", "player_name", "season", "week", "point", "price"]]
    un = pp[pp.outcome_type == "Under"][["event_id", "bookmaker", "player_name", "point", "price"]].rename(columns={"price": "u_price"})
    b = ov.merge(un, on=["event_id", "bookmaker", "player_name", "point"], how="left")
    b["nm"] = b.player_name.map(norm)
    b = b.merge(wk[["nm", "season", "week", stat]], on=["nm", "season", "week"], how="inner").rename(columns={stat: "actual"})
    b = b.dropna(subset=["actual"])
    if b.empty or "draftkings" not in set(b.bookmaker):
        say(f"## {mkt}\n\nno DraftKings rows\n"); continue
    # the 10 books a single call could name: the most-quoted books in this market
    top10 = list(b.bookmaker.value_counts().index[:10])
    if "draftkings" not in top10: top10 = top10[:9] + ["draftkings"]
    b = b[b.bookmaker.isin(top10)]
    key = ["event_id", "nm"]
    b["n_books"] = b.groupby(key).bookmaker.transform("nunique")
    b = b[b.n_books >= A.min_books]
    sd = float((b.actual - b.point).std())                      # the market's residual spread, for the price shift
    p = devig(b.price, b.u_price.fillna(b.price))              # one-sided quote -> 50/50, i.e. no shift
    b["adj"] = b.point + sd * N.ppf(np.clip(p, 0.02, 0.98))
    g = b.groupby(key)
    rows = pd.DataFrame({"season": g.season.first(), "actual": g.actual.first(), "n_books": g.n_books.first(),
                         "cons": g.point.median(), "cons_adj": g.adj.median()})
    dk = b[b.bookmaker == "draftkings"].set_index(key)[["point", "adj"]].rename(columns={"point": "dk", "adj": "dk_adj"})
    r = rows.join(dk, how="inner")
    for c in ("dk", "cons", "dk_adj", "cons_adj"): r[f"e_{c}"] = (r.actual - r[c]).abs()
    r["d_line"] = r.e_cons - r.e_dk; r["d_adj"] = r.e_cons_adj - r.e_dk_adj; r["d_best"] = r.e_cons_adj - r.e_dk
    r["differs"] = (r.cons - r.dk).abs() > 1e-9
    say(f"## {mkt}\n")
    say(f"books used: {', '.join(top10)} · residual sd {sd:.2f} · rows {len(r):,} · median books per row {r.n_books.median():.0f} · "
        f"consensus line differs from DK's on {r.differs.mean():.0%}\n")
    say("| season | rows | MAE DK | MAE consensus | cons − DK (± se) | MAE DK price-adj | MAE cons price-adj | cons-adj − DK (± se) |")
    say("|---|---|---|---|---|---|---|---|")
    for s, x in list(r.groupby("season")) + [("all", r)]:
        se = lambda v: v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan
        say(f"| {s} | {len(x):,} | {x.e_dk.mean():.3f} | {x.e_cons.mean():.3f} | {x.d_line.mean():+.3f} ± {se(x.d_line):.3f} | "
            f"{x.e_dk_adj.mean():.3f} | {x.e_cons_adj.mean():.3f} | {x.d_best.mean():+.3f} ± {se(x.d_best):.3f} |")
    say(f"\nWhere the lines differ ({int(r.differs.sum()):,} rows): cons − DK {r[r.differs].d_line.mean():+.3f}.\n")
    summary.append((mkt, len(r), r.e_dk.mean(), r.d_line.mean(), r.d_line.std() / np.sqrt(len(r)), r.d_best.mean(), r.d_best.std() / np.sqrt(len(r))))
con.close()

say("## Verdict\n")
say("| market | rows | MAE DK | consensus line vs DK | consensus price-adj vs DK |"); say("|---|---|---|---|---|")
wins = 0
for m_, n, e, d, s, d2, s2 in summary:
    say(f"| {m_} | {n:,} | {e:.3f} | {d:+.3f} ({d / e:+.1%}) ± {s:.3f} | {d2:+.3f} ({d2 / e:+.1%}) ± {s2:.3f} |")
    wins += (d2 < -2 * s2)
say(f"\nThe price-adjusted consensus is clearly closer (more than 2 se) in {wins} of {len(summary)} markets. "
    "If it wins, pull the same 10 books in the send (bookmakers=... costs the same credits as DK alone) and blend the "
    "consensus instead of DK; if it is a wash, the extra books buy nothing and the send stays as it is.")
os.makedirs(os.path.dirname(A.out), exist_ok=True)
open(A.out, "w", encoding="utf-8").write("\n".join(_out) + "\n")
print(f"\nwrote {A.out}")
