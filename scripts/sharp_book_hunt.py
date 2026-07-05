"""1) Which book was SHARPEST in 2024? (Brier of each book's de-vigged closing P(over) vs outcomes,
   restricted to props where the book quotes the consensus line -> apples-to-apples.)
2) Use that book's 2025 probability as truth to hunt weak DK prices: bet DK ($10) when
   EV = p_sharp x DK_dec - 1 clears a threshold, same-line props only. Settle on actuals."""
import os, sqlite3, warnings, sys, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
STAT = {"player_reception_yds": "receiving_yards", "player_receptions": "receptions",
        "player_rush_yds": "rushing_yards", "player_pass_yds": "passing_yards",
        "player_rush_attempts": "carries", "player_pass_completions": "completions",
        "player_pass_attempts": "attempts", "player_pass_tds": "passing_tds",
        "player_pass_interceptions": "interceptions"}
mlist = ",".join("'" + m + "'" for m in STAT)
con = sqlite3.connect(DB)
cl = pd.read_sql(f"""WITH r AS (SELECT event_id,bookmaker,market,player_name,outcome_type,price,point,
   ROW_NUMBER() OVER (PARTITION BY event_id,bookmaker,market,player_name,outcome_type ORDER BY snapshot_time DESC) rn
   FROM player_props WHERE market IN ({mlist}))
SELECT event_id,bookmaker,market,player_name,outcome_type,price,point FROM r WHERE rn=1""", con)
g = pd.read_sql("select event_id,season from games", con)
ps = pd.read_sql(f"select event_id,player_display_name nm,{','.join(set(STAT.values()))} from player_stats", con)
con.close()
cl["dec"] = np.where(cl.price > 0, 1 + cl.price / 100.0, 1 + 100.0 / cl.price.abs())
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
cl["k"] = cl.player_name.map(nrm); ps["k"] = ps.nm.map(nrm)
cl = cl.merge(g, on="event_id")
med = cl.groupby(["event_id", "market", "k"]).point.median().rename("mline").reset_index()
cl = cl.merge(med, on=["event_id", "market", "k"])
cl = cl[cl.point == cl.mline]                               # consensus-line props only
piv = cl.pivot_table(index=["event_id", "season", "bookmaker", "market", "k", "mline"],
                     columns="outcome_type", values="dec", aggfunc="first").reset_index().dropna(subset=["Over", "Under"])
piv["novig"] = (1 / piv.Over) / (1 / piv.Over + 1 / piv.Under)
ps2 = ps.drop(columns=["nm"]).drop_duplicates(["event_id", "k"])
piv = piv.merge(ps2, on=["event_id", "k"], how="inner")
piv["y"] = [r[STAT[m]] for m, r in zip(piv.market, piv.to_dict("records"))]
piv = piv.dropna(subset=["y"])
piv = piv[piv.y != piv.mline]                                # drop pushes
piv["over_won"] = (piv.y > piv.mline).astype(int)
# ---------- 1) sharpness 2024 ----------
d24 = piv[piv.season == 2024]
rows = []
for b, d in d24.groupby("bookmaker"):
    if len(d) < 2000: continue
    rows.append({"book": b, "n": len(d), "brier": ((d.novig - d.over_won) ** 2).mean(),
                 "hold": (1 / d.Over + 1 / d.Under - 1).mean()})
R = pd.DataFrame(rows).sort_values("brier")
print("=== 2024 sharpness (Brier of de-vigged P(over), consensus-line props, n>=2000) ===")
print(R.round(4).to_string(index=False))
alive25 = set(piv[piv.season == 2025].bookmaker.value_counts().loc[lambda x: x >= 2000].index)
R = R[R.book.isin(alive25) & (R.book != "draftkings")]
SHARP = R.iloc[0].book
print(f"\nSHARPEST 2024: {SHARP}")
# ---------- 2) 2025 hunt: sharp truth vs DK prices (same line) ----------
d25 = piv[piv.season == 2025]
sh = d25[d25.bookmaker == SHARP][["event_id", "market", "k", "mline", "novig"]].rename(columns={"novig": "p_sharp"})
dk = d25[d25.bookmaker == "draftkings"][["event_id", "market", "k", "mline", "Over", "Under", "over_won", "y"]]
H = dk.merge(sh, on=["event_id", "market", "k", "mline"], how="inner")
print(f"\n2025 same-line DK x {SHARP} props: {len(H):,}")
H["ev_o"] = H.p_sharp * H.Over - 1
H["ev_u"] = (1 - H.p_sharp) * H.Under - 1
H["side"] = np.where(H.ev_o > H.ev_u, 1, 0)
H["ev"] = np.maximum(H.ev_o, H.ev_u)
print("=== bet DK $10 when sharp-implied EV > threshold ===")
for th in (0.0, 0.02, 0.04, 0.07):
    b = H[H.ev > th]
    if len(b) < 40: print(f"  EV>{th:.0%}: n={len(b)} too few"); continue
    win = (b.side == b.over_won).astype(int)
    dp = np.where(b.side == 1, b.Over, b.Under)
    pnl = np.where(win == 1, 10 * (dp - 1), -10.0)
    print(f"  EV>{th:.0%}: n={len(b):5d}  overs {b.side.mean():.0%}  hit {win.mean():.1%}  P&L ${pnl.sum():+,.0f}  ROI {pnl.mean()/10:+.1%}")
b = H[H.ev > 0.02]
if len(b) > 100:
    win = (b.side == b.over_won).astype(int)
    b = b.assign(pnl=np.where(win == 1, 10 * (np.where(b.side == 1, b.Over, b.Under) - 1), -10.0))
    print("\n  by market (EV>2%):")
    for m, d in b.groupby("market"):
        if len(d) >= 30: print(f"    {m:26s} n={len(d):4d}  ROI {d.pnl.mean()/10:+.1%}")
    # sanity: same bets but at the SHARP book's own price (should be ~0/negative if sharp is efficient)
    sp = d25[d25.bookmaker == SHARP][["event_id", "market", "k", "mline", "Over", "Under"]] \
         .rename(columns={"Over": "sO", "Under": "sU"})
    b2 = b.merge(sp, on=["event_id", "market", "k", "mline"])
    dp2 = np.where(b2.side == 1, b2.sO, b2.sU)
    win2 = (b2.side == b2.over_won).astype(int)
    pnl2 = np.where(win2 == 1, 10 * (dp2 - 1), -10.0)
    print(f"\n  sanity - same bets AT {SHARP}'s own prices: ROI {pnl2.mean()/10:+.1%} (should be ~<=0)")
