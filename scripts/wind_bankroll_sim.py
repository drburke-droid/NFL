"""
$500 bankroll sim of the wind-unders strategy, 2024-25 (the two seasons with archived
24h-prior forecasts). Deployable rules only:
  - bet the UNDER when the 24h-prior FORECAST for kickoff->+3h is 8-15 kn (nflv_game_wind_fc)
  - price = best under at the consensus median closing line among REGULATED books
  - one bet per game, chronological, compounding
Sizing: flat $25, quarter-Kelly, half-Kelly. Kelly p = 0.583 (the 2015-19 OUT-OF-SAMPLE
pooled win rate - estimated entirely before the bet window), stake capped at 10% of bank.
Appends to outputs/reports/sentiment_props.md.
"""
import os, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

REG = {"draftkings","fanduel","betmgm","espnbet","williamhill_us","betrivers","hardrockbet",
       "fanatics","ballybet","betparx","unibet_us","pointsbetus","barstool","windcreek","wynnbet"}

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
g = pd.read_sql("""SELECT event_id, season, week, home_team, away_team, commence_time,
                          home_score+away_score AS total FROM games WHERE completed=1""", con)
fc = pd.read_sql("SELECT * FROM nflv_game_wind_fc", con)
tot = pd.read_sql("""SELECT event_id, bookmaker, outcome_name, price, point, snapshot_time
                     FROM game_odds WHERE market='totals'""", con)
tot = tot.sort_values("snapshot_time").groupby(["event_id","bookmaker","outcome_name"], as_index=False).last()
ov = tot[tot.outcome_name=="Over"].rename(columns={"point":"line"})
un = tot[tot.outcome_name=="Under"].rename(columns={"price":"up","point":"l2"})
t2 = ov.merge(un[["event_id","bookmaker","up","l2"]], on=["event_id","bookmaker"])
t2 = t2[t2.line==t2.l2]
t2["up"] = np.where(t2.up<0, 1+100/t2.up.abs(), 1+t2.up/100)
med = t2.groupby("event_id").line.median().rename("ml").reset_index()
t2 = t2.merge(med, on="event_id")
t2 = t2[(t2.line==t2.ml) & t2.bookmaker.isin(REG)]
cons = t2.loc[t2.groupby("event_id").up.idxmax(),
              ["event_id","line","up","bookmaker"]].rename(columns={"bookmaker":"book"})

bets = g.merge(fc, on="event_id").merge(cons, on="event_id")
bets = bets[bets.wind_fc_kn.between(8, 15)].dropna(subset=["total","line"])
bets = bets.sort_values("commence_time").reset_index(drop=True)
bets["won"] = bets.total < bets.line
bets["push"] = bets.total == bets.line
print(f"bets: {len(bets)} forecast-8-15kn games with a regulated under at the consensus line "
      f"(2024: {(bets.season==2024).sum()}, 2025: {(bets.season==2025).sum()})")
w = bets[~bets.push]
print(f"record {w.won.sum()}-{(~w.won).sum()}-{bets.push.sum()} "
      f"({w.won.mean():.1%} ex-push, avg price {bets.up.mean():.3f}, "
      f"flat ROI {np.where(w.won, w.up-1, -1.0).mean():+.1%})")

P_KELLY = 0.583                                   # 2015-19 OOS pooled, pre-window
def sim(mode, frac=0.0):
    bank, peak, maxdd, path = 500.0, 500.0, 0.0, {}
    for _, b in bets.iterrows():
        if mode == "flat":
            stake = min(25.0, bank)
        else:
            fstar = (P_KELLY * b.up - 1) / (b.up - 1)
            stake = min(max(fstar, 0) * frac * bank, 0.10 * bank)
        if bank <= 0: break
        if not b.push:
            bank += stake * (b.up - 1) if b.won else -stake
        peak = max(peak, bank); maxdd = max(maxdd, 1 - bank / peak)
        path[b.season] = bank
    return bank, maxdd, path

print(f"\n$500 bankroll, chronological ({len(bets)} bets):")
print(f"   {'sizing':<22}{'end 2024':>10}{'end 2025':>10}{'max DD':>8}")
for lab, mode, fr in (("flat $25", "flat", 0), ("quarter-Kelly (~3%)", "k", 0.25),
                      ("half-Kelly (~6%)", "k", 0.50)):
    bank, dd, path = sim(mode, fr)
    print(f"   {lab:<22}{path.get(2024, 500):>10.0f}{path.get(2025, bank):>10.0f}{dd:>8.0%}")

for s in (2024, 2025):
    ss = bets[bets.season == s]; sw = ss[~ss.push]
    roi = np.where(sw.won, sw.up - 1, -1.0).mean()
    print(f"   {s}: {sw.won.sum()}-{(~sw.won).sum()}  flat ROI {roi:+.1%}")

print("\nsample bets (first 3 of each season):")
for _, b in pd.concat([bets[bets.season==2024].head(3), bets[bets.season==2025].head(3)]).iterrows():
    res = "PUSH" if b.push else ("WON" if b.won else "lost")
    print(f"   wk{int(b.week):>2} {b.season} {b.away_team} @ {b.home_team}: fc wind {b.wind_fc_kn:.0f}kn, "
          f"U{b.line} @{b.up:.2f} ({b.book}) - {b.total:.0f} pts, {res}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Wind-unders $500 bankroll sim 2024-25 (`wind_bankroll_sim.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
