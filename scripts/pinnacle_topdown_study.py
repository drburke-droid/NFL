"""
TOP-DOWN props strategy: Pinnacle's de-vigged close = fair probability; line-shop the 19
softer books and bet ONLY where a soft price is +EV vs fair. Sentiment (the replicated
bad-news signal) layers on as a filter. 2023-25, six stat markets, closing snapshots.

Bets are restricted to soft quotes AT PINNACLE'S LINE (no distribution model needed to
translate probabilities across different lines). DNP props dropped (books void), pushes
dropped. Caveat: historical snapshots can contain stale soft lines that were unbetable in
practice; treat absolute ROI as optimistic ceiling, season-consistency as the real test.
Appends to outputs/reports/sentiment_props.md.
"""
import os, re, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

MK = {"player_receptions": "receptions", "player_reception_yds": "receiving_yards",
      "player_rush_yds": "rushing_yards", "player_rush_attempts": "carries",
      "player_pass_completions": "completions", "player_pass_yds": "passing_yards"}

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
pp = pd.read_sql(f"""
    SELECT p.event_id, p.bookmaker, p.market, p.player_name, p.outcome_type, p.price,
           p.point, p.snapshot_time, g.commence_time, g.week, g.season
    FROM player_props p JOIN games g ON g.event_id=p.event_id
    WHERE g.season IN (2023,2024,2025) AND p.market IN ({",".join("'"+m+"'" for m in MK)})""", con)
pp = pp[pp.snapshot_time < pp.commence_time]
pp["price"] = np.where(pp.price < 0, 1 + 100 / pp.price.abs(), 1 + pp.price / 100)
pp = pp.sort_values("snapshot_time").groupby(
    ["event_id", "market", "player_name", "bookmaker", "outcome_type"], as_index=False).last()
ov = pp[pp.outcome_type == "Over"].rename(columns={"price": "over", "point": "pt_o"})
un = pp[pp.outcome_type == "Under"].rename(columns={"price": "under", "point": "pt_u"})
b = ov.merge(un[["event_id", "market", "player_name", "bookmaker", "under", "pt_u"]],
             on=["event_id", "market", "player_name", "bookmaker"])
b = b[b.pt_o == b.pt_u].rename(columns={"pt_o": "point"})

# fair value from Pinnacle's close
pin = b[b.bookmaker == "pinnacle"].copy()
pin["p_fair"] = (1 / pin.over) / (1 / pin.over + 1 / pin.under)
pin = pin[["event_id", "market", "player_name", "point", "p_fair", "season", "week", "commence_time"]]
soft = b[b.bookmaker != "pinnacle"].merge(
    pin.rename(columns={"point": "pin_point"}),
    on=["event_id", "market", "player_name"], suffixes=("", "_p"))
soft = soft[soft.point == soft.pin_point].copy()             # same line as Pinnacle only
print(f"pinnacle-anchored props: {pin.groupby('season').size().to_dict()} | "
      f"soft same-line quotes: {len(soft)}")

# outcomes
wkst = pd.read_sql("""SELECT player_display_name nm, season, week, receptions,
                             receiving_yards, rushing_yards, carries, completions, passing_yards
                      FROM nflv_weekly WHERE season>=2023 AND season_type='REG'""", con)
wkst["nm"] = wkst.nm.map(norm)
soft["nm"] = soft.player_name.map(norm)
soft["week"] = pd.to_numeric(soft.week, errors="coerce")
soft = soft.dropna(subset=["week"]); soft["week"] = soft.week.astype(int)
soft = soft[soft.week <= 18]
soft = soft.merge(wkst, left_on=["nm", "season_p", "week"], right_on=["nm", "season", "week"], how="left")
soft["actual"] = [r[MK[m]] for m, r in zip(soft.market, soft.to_dict("records"))]
soft = soft.dropna(subset=["actual"])
soft = soft[soft.actual != soft.point]
soft["won_over"] = (soft.actual > soft.point).astype(int)
soft["ev_over"] = soft.p_fair * soft.over - 1
soft["ev_under"] = (1 - soft.p_fair) * soft.under - 1

# sentiment (monthly, frozen)
gd = pd.read_sql("SELECT name nm, ym, articles, avg_tone FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)
soft["month"] = np.select([soft.week <= 4, soft.week <= 8, soft.week <= 13, soft.week <= 17], [9, 10, 11, 12], 1)
soft["sy"] = soft.season_p; soft.loc[soft.month == 1, "sy"] = soft.season_p + 1
pr = gd[["nm", "y", "m", "articles", "avg_tone"]].copy()
pr["month"] = pr.m + 1; pr.loc[pr.m == 12, "month"] = 1
pr["sy"] = pr.y; pr.loc[pr.m == 12, "sy"] += 1
soft = soft.merge(pr[["nm", "sy", "month", "articles", "avg_tone"]]
                  .rename(columns={"articles": "prev_n", "avg_tone": "prev_tone"}), on=["nm", "sy", "month"], how="left")
soft["bad"] = ((soft.prev_tone < -2) & (soft.prev_n >= 100)).fillna(False)

def run(sub, tmin, veto=False, only_aligned=False):
    o = sub[(sub.ev_over >= tmin)]
    u = sub[(sub.ev_under >= tmin)]
    if veto: o = o[~o.bad]
    if only_aligned: u = u[u.bad]; o = o.iloc[0:0]
    ret = np.concatenate([np.where(o.won_over == 1, o.over - 1, -1.0),
                          np.where(u.won_over == 0, u.under - 1, -1.0)])
    return (ret.mean() if len(ret) else 0.0), len(ret)

print(f"\nprops with outcomes: {len(soft)} soft quotes | bad-news cell quotes: {soft.bad.sum()}")
print("\ntop-down ROI (bet soft book when EV vs Pinnacle-fair >= t):")
print(f"{'':>30}{'t>=0%':>16}{'t>=2%':>16}{'t>=4%':>16}")
for lab, kw in (("baseline top-down", {}), ("+ sentiment VETO (overs)", {"veto": True}),
                ("sentiment-aligned unders only", {"only_aligned": True})):
    cells = [run(soft, t, **kw) for t in (0.0, 0.02, 0.04)]
    print(f"   {lab:<27}" + "".join(f"  n={n:>6} {m:+.1%}" for m, n in cells))
print("\nby season (t>=2%):")
for ssn in (2023, 2024, 2025):
    s = soft[soft.season_p == ssn]
    for lab, kw in (("baseline", {}), ("+veto", {"veto": True})):
        m, n = run(s, 0.02, **kw)
        print(f"   {ssn} {lab:<10} n={n:>6}  ROI {m:+.1%}")
print("\nsanity — ALL soft quotes both sides blind (juice floor):")
m0 = np.concatenate([np.where(soft.won_over == 1, soft.over - 1, -1.0),
                     np.where(soft.won_over == 0, soft.under - 1, -1.0)])
print(f"   n={len(m0)}  ROI {m0.mean():+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Pinnacle top-down + sentiment (`pinnacle_topdown_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
