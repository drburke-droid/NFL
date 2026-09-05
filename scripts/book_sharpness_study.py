"""
Which sportsbook posts the sharpest closing lines in our db, per year?

Sharpness = line placement error vs what actually happened, measured RELATIVE to the
per-game consensus so a book isn't penalized for covering a different mix of games:

  game lines (2020-2025, snapshot ~T-2h):
    spreads:  err = |home_margin + home_point|
    totals:   err = |total_pts - point|
    h2h:      Brier of devigged (2-way) home win prob
  props (2023-2025): err = |line - actual stat| for the 9 stat-mappable markets,
    actuals joined from player_stats on event_id + normalized player name.

  rel = err_book - mean(err of all books quoting the same game/player line).
  Negative rel = sharper than the market. Books shown need >=100 game-lines
  (>=500 prop lines) in a season.

Report: outputs/reports/book_sharpness.md
"""
import os, re, sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))

# 2023 has duplicate event rows: odds often attach to one row, scores to its twin.
# Inherit scores across matchup duplicates (same team pair within 2 days), flipping
# the margin when the duplicate lists home/away reversed.
allg = pd.read_sql("""SELECT event_id, season, home_team, away_team, home_score, away_score,
                             completed, commence_time FROM games""", con)
allg["date"] = pd.to_datetime(allg.commence_time).dt.tz_localize(None)
done = allg[(allg.completed == 1) & allg.home_score.notna()]
by_pair = {}
for r in done.itertuples():
    by_pair.setdefault(frozenset((r.home_team, r.away_team)), []).append(r)
rows = []
for r in allg.itertuples():
    if r.completed == 1 and pd.notna(r.home_score):
        rows.append((r.event_id, r.season, r.home_team, r.away_team,
                     r.home_score - r.away_score, r.home_score + r.away_score))
        continue
    for d in by_pair.get(frozenset((r.home_team, r.away_team)), []):
        if abs((d.date - r.date).days) <= 2:
            m = d.home_score - d.away_score
            if d.home_team != r.home_team:
                m = -m
            rows.append((r.event_id, r.season, r.home_team, r.away_team,
                         m, d.home_score + d.away_score))
            break
games = pd.DataFrame(rows, columns=["event_id", "season", "home_team", "away_team",
                                    "margin", "total"])

# ---------------- game lines ----------------
go = pd.read_sql("""SELECT event_id, bookmaker, market, outcome_name, price, point
                    FROM game_odds WHERE market IN ('spreads','totals','h2h')""", con)
go = go.merge(games, on="event_id", how="inner")

def rel_table(df, err_col, min_n):
    """per book-season mean err, rel-to-consensus (games quoted by >=4 books), n"""
    df = df.copy()
    grp = df.groupby(["event_id", "key"])
    df["n_books"] = grp.bookmaker.transform("nunique")
    df["cons"] = grp[err_col].transform("mean")
    df = df[df.n_books >= 4]
    df["rel"] = df[err_col] - df.cons
    agg = (df.groupby(["season", "bookmaker"])
             .agg(n=("rel", "size"), mae=(err_col, "mean"), rel=("rel", "mean"))
             .reset_index())
    return agg[agg.n >= min_n]

# spreads: home-team row
sp = go[(go.market == "spreads") & (go.outcome_name == go.home_team)].dropna(subset=["point"]).copy()
sp = sp.drop_duplicates(["event_id", "bookmaker"])
sp["err"] = (sp.margin + sp.point).abs()
sp["key"] = "spreads"

# totals: Over row
tt = go[(go.market == "totals") & (go.outcome_name == "Over")].dropna(subset=["point"]).copy()
tt = tt.drop_duplicates(["event_id", "bookmaker"])
tt["err"] = (tt.total - tt.point).abs()
tt["key"] = "totals"

# h2h: devig 2-way, Brier on home win (skip pushes/ties)
h2 = go[go.market == "h2h"].copy()
h2["imp"] = np.where(h2.price > 0, 100 / (h2.price + 100), -h2.price / (-h2.price + 100))
piv = (h2.pivot_table(index=["event_id", "bookmaker", "season", "margin"],
                      columns=h2.outcome_name.eq(h2.home_team).map({True: "h", False: "a"}),
                      values="imp", aggfunc="first").reset_index().dropna(subset=["h", "a"]))
piv = piv[piv.margin != 0]
piv["p_home"] = piv.h / (piv.h + piv.a)
piv["err"] = (piv.p_home - (piv.margin > 0)) ** 2
piv["key"] = "h2h"

print("=" * 78)
print("GAME LINES — sharpness per book-year (rel<0 ⇒ sharper than consensus)")
print("=" * 78)
for label, frame, mcol in (("SPREADS  (pts of error)", sp, "err"),
                           ("TOTALS   (pts of error)", tt, "err"),
                           ("MONEYLINE (Brier ×100)", piv.assign(err=piv.err * 100), "err")):
    agg = rel_table(frame, "err", 100 if label[0] != "M" else 100)
    print(f"\n{label} — top 3 sharpest per season:")
    for season in sorted(agg.season.unique()):
        s = agg[agg.season == season].sort_values("rel")
        top = "   ".join(f"{r.bookmaker} ({r.rel:+.2f})" for r in s.head(3).itertuples())
        worst = s.iloc[-1]
        print(f"  {season}: {top}   | dullest: {worst.bookmaker} ({worst.rel:+.2f})  [n≈{int(s.n.median())}]")

# ---------------- props ----------------
PROP_STAT = {
    "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds",
    "player_pass_attempts": "attempts", "player_pass_completions": "completions",
    "player_pass_interceptions": "interceptions",
    "player_rush_yds": "rushing_yards", "player_rush_attempts": "carries",
    "player_receptions": "receptions", "player_reception_yds": "receiving_yards",
}
pp = pd.read_sql(f"""SELECT event_id, bookmaker, market, player_name, point
                     FROM player_props
                     WHERE market IN {tuple(PROP_STAT)} AND outcome_type='Over'
                       AND point IS NOT NULL""", con)
ps = pd.read_sql(f"""SELECT event_id, player_display_name,
                            {",".join(set(PROP_STAT.values()))}
                     FROM player_stats""", con)
pp["nm"] = pp.player_name.map(norm)
ps["nm"] = ps.player_display_name.map(norm)
pp = pp.drop_duplicates(["event_id", "bookmaker", "market", "nm"])
m = pp.merge(ps.drop(columns=["player_display_name"]), on=["event_id", "nm"], how="inner")
m["actual"] = m.apply(lambda r: r[PROP_STAT[r.market]], axis=1)
m = m.dropna(subset=["actual"])
m = m.merge(games[["event_id", "season"]], on="event_id", how="inner")
m["err"] = (m.point - m.actual).abs()
# normalize error scale across prop types so yds don't drown receptions:
m["z_err"] = m.err / m.groupby("market").err.transform("mean")
m["key"] = m.market + "|" + m.nm

print("\n" + "=" * 78)
print("PLAYER PROPS — 9 stat markets pooled, scale-normalized error")
print("=" * 78)
print(f"matched prop lines: {len(m):,} ({m.event_id.nunique()} games, join rate "
      f"{len(m)/len(pp):.0%})")
agg = rel_table(m, "z_err", 500)
for season in sorted(agg.season.unique()):
    s = agg[agg.season == season].sort_values("rel")
    print(f"\n  {season} (rel z-err; negative = sharper):")
    for r in s.itertuples():
        bar = "#" * max(0, int((0.06 - r.rel) * 200))
        print(f"    {r.bookmaker:<18} rel {r.rel:+.4f}  n={r.n:>6}  {bar}")

# pinnacle check per prop market (the sharp-book benchmark)
print("\npinnacle rel z-err by market (all seasons pooled):")
mm = m.copy()
grp = mm.groupby(["event_id", "key"])
mm["n_books"] = grp.bookmaker.transform("nunique")
mm["rel"] = mm.z_err - grp.z_err.transform("mean")
mm = mm[mm.n_books >= 4]
pin = mm[mm.bookmaker == "pinnacle"].groupby("market").rel.agg(["mean", "size"])
for mk, r in pin.iterrows():
    print(f"    {mk:<28} rel {r['mean']:+.4f}  n={int(r['size'])}")

open(os.path.join(ROOT, "outputs", "reports", "book_sharpness.md"), "w", encoding="utf-8").write(
    "# Sharpest sportsbook per year — game lines & props\n\n"
    "Generated by `scripts/book_sharpness_study.py`. Sharpness = closing-line error vs\n"
    "actual result, relative to the consensus of all books quoting the same line\n"
    "(rel<0 = sharper than market). Game lines 2020-2025; props 2023-2025.\n\n"
    "```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/book_sharpness.md")
