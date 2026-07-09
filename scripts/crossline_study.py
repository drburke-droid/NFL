"""
CROSS-LINE shopping: price soft-book quotes at lines DIFFERENT from Pinnacle's.

Bridge: stat ~ Normal(mu, sigma_m), sigma per market fit on the PRIOR season as
1.2533 * mean|actual - pinnacle_line| (half-normal identity), mu solved from Pinnacle's
anchor: P(X > L_pin) = p_fair  =>  mu = L_pin + sigma * PhiInv(p_fair).
Fair prob at a soft line Ls: p(Ls) = 1 - Phi((Ls - mu)/sigma).

Discipline: sigma is season-out (fit t-1, bet t -> 2024 & 2025 bettable); reliability
check on the bridge BEFORE ROI; regulated books only; sentiment veto on overs; EV
thresholds 4%/8% (model risk premium); anchors restricted to 0.25<p_fair<0.75 and line
distance <= 1.5 sigma (no wild extrapolation). Appends to sentiment_props.md.
"""
import os, re, sqlite3
import numpy as np
import pandas as pd
from scipy.stats import norm as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                       re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

MK = {"player_receptions": "receptions", "player_reception_yds": "receiving_yards",
      "player_rush_yds": "rushing_yards", "player_rush_attempts": "carries",
      "player_pass_completions": "completions", "player_pass_yds": "passing_yards"}
REG = {"draftkings","fanduel","betmgm","espnbet","williamhill_us","betrivers","hardrockbet",
       "fanatics","ballybet","betparx","unibet_us","pointsbetus","barstool","windcreek","wynnbet"}

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

pin = b[b.bookmaker == "pinnacle"].copy()
pin["p_fair"] = (1 / pin.over) / (1 / pin.over + 1 / pin.under)
pin = pin[(pin.p_fair > 0.25) & (pin.p_fair < 0.75)]
pin = pin[["event_id", "market", "player_name", "point", "p_fair"]].rename(columns={"point": "Lpin"})
soft = b[b.bookmaker != "pinnacle"].merge(pin, on=["event_id", "market", "player_name"])

# outcomes
wkst = pd.read_sql("""SELECT player_display_name nm, season, week, receptions,
                             receiving_yards, rushing_yards, carries, completions, passing_yards
                      FROM nflv_weekly WHERE season>=2023 AND season_type='REG'""", con)
wkst["nm"] = wkst.nm.map(nrm)
soft["nm"] = soft.player_name.map(nrm)
soft["week"] = pd.to_numeric(soft.week, errors="coerce")
soft = soft.dropna(subset=["week"]); soft["week"] = soft.week.astype(int); soft = soft[soft.week <= 18]
soft = soft.merge(wkst, on=["nm", "season", "week"], how="left")
soft["actual"] = [r[MK[m]] for m, r in zip(soft.market, soft.to_dict("records"))]
soft = soft.dropna(subset=["actual"])
soft = soft[soft.actual != soft.point]

# sigma per market from PRIOR season: 1.2533 * mean |actual - Lpin| on pinnacle-anchored props
anch = soft.drop_duplicates(["event_id", "market", "player_name"])
SIG = {}
for s in (2023, 2024):
    for mk, sub in anch[anch.season == s].groupby("market"):
        SIG[(s + 1, mk)] = 1.2533 * (sub.actual - sub.Lpin).abs().mean()
print("sigma (fit on prior season):")
for (s, mk), v in sorted(SIG.items()): print(f"   {s} {mk:<26} {v:6.2f}")

soft["sig"] = [SIG.get((s, m), np.nan) for s, m in zip(soft.season, soft.market)]
soft = soft.dropna(subset=["sig"])                            # 2024 & 2025 only
soft["mu"] = soft.Lpin + soft.sig * N.ppf(soft.p_fair)
soft["ldist"] = (soft.point - soft.Lpin).abs() / soft.sig
soft = soft[soft.ldist <= 1.5]
soft["p_over"] = 1 - N.cdf((soft.point - soft.mu) / soft.sig)
soft["won_over"] = (soft.actual > soft.point).astype(int)
soft["cross"] = soft.point != soft.Lpin

# reliability of the bridge on CROSS-line quotes (the new territory)
print("\nbridge reliability on cross-line quotes (pred p_over decile vs actual):")
cx = soft[soft.cross]
cx = cx.assign(dec_=pd.qcut(cx.p_over, 8, duplicates="drop"))
for iv, sub in cx.groupby("dec_", observed=True):
    print(f"   pred {sub.p_over.mean():.2f}  actual {sub.won_over.mean():.2f}  n={len(sub)}")

# sentiment (monthly, frozen) for the veto
gd = pd.read_sql("SELECT name nm, ym, articles, avg_tone FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)
soft["month"] = np.select([soft.week <= 4, soft.week <= 8, soft.week <= 13, soft.week <= 17], [9, 10, 11, 12], 1)
soft["sy"] = soft.season; soft.loc[soft.month == 1, "sy"] = soft.season + 1
pr = gd[["nm", "y", "m", "articles", "avg_tone"]].copy()
pr["month"] = pr.m + 1; pr.loc[pr.m == 12, "month"] = 1
pr["sy"] = pr.y; pr.loc[pr.m == 12, "sy"] += 1
soft = soft.merge(pr[["nm", "sy", "month", "articles", "avg_tone"]]
                  .rename(columns={"articles": "prev_n", "avg_tone": "prev_tone"}), on=["nm", "sy", "month"], how="left")
soft["bad"] = ((soft.prev_tone < -2) & (soft.prev_n >= 100)).fillna(False)

soft = soft[soft.bookmaker.isin(REG)]                        # bettable books only
soft["ev_over"] = soft.p_over * soft.over - 1
soft["ev_under"] = (1 - soft.p_over) * soft.under - 1

def bets(sub, tmin):
    o = sub[(sub.ev_over >= tmin) & ~sub.bad].assign(side="O", dec=sub.over, win=sub.won_over == 1, ev=sub.ev_over)
    u = sub[sub.ev_under >= tmin].assign(side="U", dec=sub.under, win=sub.won_over == 0, ev=sub.ev_under)
    B = pd.concat([o, u]).sort_values("ev", ascending=False)
    return B.groupby(["event_id", "market", "player_name", "side"], as_index=False).first()

print("\nROI (regulated books, veto on overs; deduped best price per prop-side):")
print(f"{'':>34}{'EV>=4%':>18}{'EV>=8%':>18}")
for lab, mask in (("SAME-line (old strategy)", ~soft.cross), ("CROSS-line only (new)", soft.cross)):
    for ssn in (2024, 2025):
        sub = soft[mask & (soft.season == ssn)]
        row = f"   {lab:<28} {ssn}"
        for t in (0.04, 0.08):
            B = bets(sub, t)
            ret = np.where(B.win, B.dec - 1, -1.0)
            row += f"  n={len(B):>5} {ret.mean() if len(B) else 0:+.1%}"
        print(row)
print("\ncross-line by distance (EV>=4%, both seasons):")
cxb = bets(soft[soft.cross], 0.04)
cxb["dist"] = pd.cut(cxb.ldist, [0, 0.3, 0.7, 1.5], labels=["tiny (<0.3σ)", "0.3-0.7σ", "0.7-1.5σ"])
for dv, sub in cxb.groupby("dist", observed=True):
    ret = np.where(sub.win, sub.dec - 1, -1.0)
    print(f"   {dv:<14} n={len(sub):>5}  ROI {ret.mean():+.1%}")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## Cross-line shopping (`crossline_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
