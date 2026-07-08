"""
TRUE out-of-sample test of the fade-bad-news-unders rule on 2023 — a season neither the
rule nor the backtest frame ever saw. The rule is FROZEN from the 2024-25 study:
    bet UNDER (best closing price) when the player's PRIOR CALENDAR MONTH had
    avg_tone < -2 AND >= 100 articles.  (weekly variant: prior-2-GDELT-weeks, vol >= 25)

The 2023 frame is built from raw player_props (closing snapshot per book before kickoff,
consensus median line, de-vigged over prob, best prices) + nflv_weekly outcomes. DNP props
are dropped (books void them). Appends to outputs/reports/sentiment_props.md.
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
           p.point, p.snapshot_time, g.commence_time, g.week
    FROM player_props p JOIN games g ON g.event_id=p.event_id
    WHERE g.season=2023 AND p.market IN ({",".join("'"+m+"'" for m in MK)})""", con)
pp = pp[pp.snapshot_time < pp.commence_time]
# raw prices are AMERICAN odds; convert to decimal (the 2024-25 frame was pre-converted)
pp["price"] = np.where(pp.price < 0, 1 + 100 / pp.price.abs(), 1 + pp.price / 100)
pp["snap"] = pd.to_datetime(pp.snapshot_time)
# closing snapshot per (event, market, player, book, side)
pp = pp.sort_values("snap").groupby(
    ["event_id", "market", "player_name", "bookmaker", "outcome_type"], as_index=False).last()
ov = pp[pp.outcome_type == "Over"].rename(columns={"price": "over", "point": "pt_o"})
un = pp[pp.outcome_type == "Under"].rename(columns={"price": "under", "point": "pt_u"})
b = ov.merge(un[["event_id", "market", "player_name", "bookmaker", "under", "pt_u"]],
             on=["event_id", "market", "player_name", "bookmaker"])
b = b[b.pt_o == b.pt_u].rename(columns={"pt_o": "point"})
# consensus line = median across books; keep books AT the consensus line
med = b.groupby(["event_id", "market", "player_name"]).point.median().rename("mline").reset_index()
b = b.merge(med, on=["event_id", "market", "player_name"])
b = b[b.point == b.mline].copy()
b["pnv"] = (1 / b.over) / (1 / b.over + 1 / b.under)
f = b.groupby(["event_id", "market", "player_name"]).agg(
    mline=("mline", "first"), novig=("pnv", "mean"), best_over=("over", "max"),
    best_under=("under", "max"), books=("bookmaker", "nunique"),
    week=("week", "first"), commence=("commence_time", "first")).reset_index()
f = f[f.books >= 2]
print(f"2023 frame: {len(f)} props ({f.event_id.nunique()} games, median books/prop "
      f"{int(f.books.median())})")

# outcomes
wkst = pd.read_sql("""SELECT player_display_name nm, week, receptions, receiving_yards,
                             rushing_yards, carries, completions, passing_yards
                      FROM nflv_weekly WHERE season=2023 AND season_type='REG'""", con)
wkst["nm"] = wkst.nm.map(norm)
f["nm"] = f.player_name.map(norm)
f["week"] = pd.to_numeric(f.week, errors="coerce")
f = f.dropna(subset=["week"]); f["week"] = f.week.astype(int)
f = f[f.week <= 18]                                          # regular season only
f = f.merge(wkst, on=["nm", "week"], how="left")
f["actual"] = [r[MK[m]] for m, r in zip(f.market, f.to_dict("records"))]
f = f.dropna(subset=["actual"])                              # DNP -> prop voided
f = f[f.actual != f.mline]                                   # drop pushes
f["won_over"] = (f.actual > f.mline).astype(int)
print(f"with outcomes (DNP/push dropped): {len(f)}; overs hit {f.won_over.mean():.1%} "
      f"vs novig {f.novig.mean():.1%}")

# monthly prior-month sentiment (frozen mapping)
gd = pd.read_sql("SELECT name nm, ym, articles, avg_tone FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)
f["month"] = np.select([f.week <= 4, f.week <= 8, f.week <= 13, f.week <= 17], [9, 10, 11, 12], 1)
f["sy"] = 2023; f.loc[f.month == 1, "sy"] = 2024
pr = gd[["nm", "y", "m", "articles", "avg_tone"]].copy()
pr["month"] = pr.m + 1; pr.loc[pr.m == 12, "month"] = 1
pr["sy"] = pr.y; pr.loc[pr.m == 12, "sy"] += 1
f = f.merge(pr[["nm", "sy", "month", "articles", "avg_tone"]]
            .rename(columns={"articles": "prev_n", "avg_tone": "prev_tone"}), on=["nm", "sy", "month"], how="left")

def roi(sub, side):
    if not len(sub): return 0.0, 0
    ret = np.where(sub.won_over == (0 if side == "under" else 1),
                   (sub.best_under if side == "under" else sub.best_over) - 1, -1.0)
    return ret.mean(), len(sub)

ok = f.dropna(subset=["prev_tone"])
print(f"\nFROZEN MONTHLY RULE on unseen 2023 (n with sentiment {len(ok)}):")
for lab, sub, side in (
    ("ALL unders baseline", ok, "under"),
    ("ALL overs baseline", ok, "over"),
    ("UNDER prev_tone<-2 & n>=100  [THE RULE]", ok[(ok.prev_tone < -2) & (ok.prev_n >= 100)], "under"),
    ("OVER  same cell (sanity)", ok[(ok.prev_tone < -2) & (ok.prev_n >= 100)], "over"),
    ("UNDER prev_tone<-2 (no vol floor)", ok[ok.prev_tone < -2], "under"),
):
    m, n = roi(sub, side)
    print(f"   {lab:<42} n={n:>5}  ROI {m:+.1%}")
cell = ok[(ok.prev_tone < -2) & (ok.prev_n >= 100)]
print("   rule cell by market:")
for mk, sub in cell.groupby("market"):
    m, n = roi(sub, "under")
    print(f"     {mk:<28} n={n:>4}  ROI {m:+.1%}")

# weekly variant (frozen: last two GDELT weeks, tone<-2, vol>=25)
wk = pd.read_sql("SELECT name nm, wk, articles, avg_tone FROM nflv_gdelt_wk", con)
wk["wkd"] = pd.to_datetime(wk.wk)
f["gdate"] = pd.to_datetime(f.commence).dt.tz_localize(None).dt.normalize()
f = f.reset_index(drop=True); f["pi"] = f.index
m2 = f[["pi", "nm", "gdate"]].merge(wk, on="nm", how="left")
m2["lag"] = (m2.gdate - m2.wkd).dt.days
m2 = m2[m2.lag.between(3, 16)]
agg = m2.groupby("pi").apply(lambda x: pd.Series({
    "wk_n": x.articles.sum(),
    "wk_tone": np.average(x.avg_tone, weights=x.articles) if x.articles.sum() > 0 else np.nan}),
    include_groups=False).reset_index()
f = f.merge(agg, on="pi", how="left")
ok2 = f.dropna(subset=["wk_tone"])
print(f"\nFROZEN WEEKLY RULE on unseen 2023 (n {len(ok2)}):")
for lab, sub, side in (
    ("ALL unders baseline", ok2, "under"),
    ("UNDER wk_tone<-2 & wk_n>=25  [THE RULE]", ok2[(ok2.wk_tone < -2) & (ok2.wk_n >= 25)], "under"),
    ("OVER  same cell (sanity)", ok2[(ok2.wk_tone < -2) & (ok2.wk_n >= 25)], "over"),
):
    m, n = roi(sub, side)
    print(f"   {lab:<42} n={n:>5}  ROI {m:+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## TRUE out-of-sample: 2023 (`sentiment_props_2023_oos.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
