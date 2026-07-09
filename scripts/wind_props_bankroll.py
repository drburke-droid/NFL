"""
$500 bankroll sim 2024-25, wind-unders COMBINED: game totals + QB pass-yards unders,
same deployable filter (24h-prior forecast 8-15kn for kickoff->+3h), regulated books,
consensus median closing line, best regulated under price at that line.
Pass-completions excluded (near-duplicate exposure to pass-yards on the same QB).
Correlation warning is real: one game can carry 1 total + 2 QB unders.
Kelly p: totals 0.583 (2015-19 OOS), props 0.55 (conservative; only 2023 actual-wind
history exists as prior). Stakes capped at 10% of bank. Appends to sentiment_props.md.
"""
import os, re, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

REG = {"draftkings","fanduel","betmgm","espnbet","williamhill_us","betrivers","hardrockbet",
       "fanatics","ballybet","betparx","unibet_us","pointsbetus","barstool","windcreek","wynnbet"}
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
g = pd.read_sql("""SELECT event_id, season, week, home_team, away_team, commence_time,
                          home_score+away_score AS total FROM games WHERE completed=1""", con)
fc = pd.read_sql("SELECT * FROM nflv_game_wind_fc", con)
g = g.merge(fc, on="event_id")
g = g[g.wind_fc_kn.between(8, 15)]

# ---- leg 1: game-total unders (same as wind_bankroll_sim) ----
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
cons = t2.loc[t2.groupby("event_id").up.idxmax(), ["event_id","line","up","bookmaker"]]
tb = g.merge(cons, on="event_id").dropna(subset=["total","line"])
tb["won"] = tb.total < tb.line; tb["push"] = tb.total == tb.line
tb["kind"] = "total"; tb["desc"] = "U" + tb.line.astype(str)

# ---- leg 2: QB pass-yards unders ----
pp = pd.read_sql("""SELECT event_id, bookmaker, player_name, outcome_type, price, point, snapshot_time
                    FROM player_props WHERE market='player_pass_yds'""", con)
pp = pp[pp.event_id.isin(g.event_id)]
pp["price"] = np.where(pp.price<0, 1+100/pp.price.abs(), 1+pp.price/100)
pp = pp.sort_values("snapshot_time").groupby(
    ["event_id","player_name","bookmaker","outcome_type"], as_index=False).last()
o = pp[pp.outcome_type=="Over"].rename(columns={"point":"pt"})
u = pp[pp.outcome_type=="Under"].rename(columns={"price":"upx","point":"pt2"})
m = o.merge(u[["event_id","player_name","bookmaker","upx","pt2"]], on=["event_id","player_name","bookmaker"])
m = m[m.pt==m.pt2]
pmed = m.groupby(["event_id","player_name"]).pt.median().rename("mpt").reset_index()
m = m.merge(pmed, on=["event_id","player_name"])
m = m[(m.pt==m.mpt) & m.bookmaker.isin(REG)]
pr = m.loc[m.groupby(["event_id","player_name"]).upx.idxmax(),
           ["event_id","player_name","pt","upx","bookmaker"]]
pr = pr.merge(g[["event_id","season","week","home_team","away_team","commence_time"]], on="event_id")
wk = pd.read_sql("""SELECT player_display_name nm, season, week, passing_yards
                    FROM nflv_weekly WHERE season>=2024 AND season_type='REG'""", con)
wk["nm"] = wk.nm.map(norm)
pr["nm"] = pr.player_name.map(norm)
pr["week"] = pd.to_numeric(pr.week, errors="coerce"); pr = pr.dropna(subset=["week"])
pr["week"] = pr.week.astype(int)
pr = pr.merge(wk, on=["nm","season","week"], how="left").dropna(subset=["passing_yards"])
pr["won"] = pr.passing_yards < pr.pt; pr["push"] = pr.passing_yards == pr.pt
pr = pr.rename(columns={"upx":"up"})
pr["kind"] = "prop"; pr["desc"] = pr.player_name + " U" + pr.pt.astype(str) + "yd"

cols = ["event_id","season","week","commence_time","kind","desc","up","won","push"]
bets = pd.concat([tb[cols], pr[cols]]).sort_values(["commence_time","kind"]).reset_index(drop=True)
for k, sub in bets.groupby("kind"):
    w = sub[~sub.push]
    print(f"{k}s: {len(sub)} bets, {w.won.sum()}-{(~w.won).sum()}-{sub.push.sum()} "
          f"({w.won.mean():.1%}), flat ROI {np.where(w.won, w.up-1, -1.0).mean():+.1%}")
w = bets[~bets.push]
print(f"COMBINED: {len(bets)} bets, {w.won.sum()}-{(~w.won).sum()}-{bets.push.sum()} "
      f"({w.won.mean():.1%}), flat ROI {np.where(w.won, w.up-1, -1.0).mean():+.1%}")
per_game = bets.groupby("event_id").size()
print(f"bets per qualifying game: mean {per_game.mean():.1f}, max {per_game.max()} (correlated exposure)")
for s in (2024, 2025):
    sw = bets[(bets.season==s) & ~bets.push]
    print(f"   {s}: {sw.won.sum()}-{(~sw.won).sum()}  flat ROI {np.where(sw.won, sw.up-1, -1.0).mean():+.1%}")

P = {"total": 0.583, "prop": 0.55}
def sim(mode, frac=0.0):
    bank, peak, maxdd, path = 500.0, 500.0, 0.0, {}
    for _, b in bets.iterrows():
        if bank <= 0: break
        if mode == "flat":
            stake = min(25.0, bank)
        else:
            fstar = (P[b.kind] * b.up - 1) / (b.up - 1)
            stake = min(max(fstar, 0) * frac * bank, 0.10 * bank)
        if not b.push:
            bank += stake * (b.up - 1) if b.won else -stake
        peak = max(peak, bank); maxdd = max(maxdd, 1 - bank / peak)
        path[b.season] = bank
    return bank, maxdd, path

print(f"\n$500 bankroll, chronological ({len(bets)} bets):")
print(f"   {'sizing':<22}{'end 2024':>10}{'end 2025':>10}{'max DD':>8}")
for lab, mode, fr in (("flat $25", "flat", 0), ("quarter-Kelly", "k", 0.25), ("half-Kelly", "k", 0.50)):
    bank, dd, path = sim(mode, fr)
    print(f"   {lab:<22}{path.get(2024, 500):>10.0f}{path.get(2025, bank):>10.0f}{dd:>8.0%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Wind unders incl. QB pass-yds props, $500 sim (`wind_props_bankroll.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
