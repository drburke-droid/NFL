"""
2025 walk-through of the Pinnacle top-down + sentiment-veto strategy:
  - the actual bets it would have flagged (one bet per prop-side, best +EV soft price,
    EV vs Pinnacle's de-vigged close >= threshold, sentiment veto on overs)
  - a $500 bankroll simulated chronologically with Kelly sizing (full / half / quarter),
    Kelly fraction f* = (p*d - 1)/(d - 1) with p = Pinnacle fair prob, capped at 10%/bet.
Appends to outputs/reports/sentiment_props.md.
"""
import os, re, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

# rebuild the soft-quote frame (same construction as pinnacle_topdown_study)
g = {"__file__": os.path.abspath(os.path.join(ROOT, "scripts", "pinnacle_topdown_study.py"))}
src = open(os.path.join(ROOT, "scripts", "pinnacle_topdown_study.py"), encoding="utf-8").read()
src = src.split("def run(")[0].replace("_out = []; _print = print", "_o2=[]").replace(
    '''def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)''', "")
exec(src, g)
soft = g["soft"]

def bets_for(season, tmin, veto=True):
    s = soft[soft.season_p == season].copy()
    o = s[s.ev_over >= tmin].copy();  o["side"] = "OVER";  o["dec"] = o.over;  o["ev"] = o.ev_over
    u = s[s.ev_under >= tmin].copy(); u["side"] = "UNDER"; u["dec"] = u.under; u["ev"] = u.ev_under
    if veto: o = o[~o.bad]
    B = pd.concat([o, u])
    B["p_bet"] = np.where(B.side == "OVER", B.p_fair, 1 - B.p_fair)
    B["win"] = np.where(B.side == "OVER", B.won_over == 1, B.won_over == 0)
    # one bet per prop-side at the BEST price
    B = B.sort_values("ev", ascending=False).groupby(
        ["event_id", "market", "player_name", "side"], as_index=False).first()
    return B.sort_values("commence_time")

def kelly(B, bank0=500.0, frac=0.5, cap=0.10):
    bank = bank0; hist = []
    for _, r in B.iterrows():
        f = max(0.0, (r.p_bet * r.dec - 1) / (r.dec - 1)) * frac
        stake = min(f, cap) * bank
        bank += stake * (r.dec - 1) if r.win else -stake
        hist.append((r.week, bank))
        if bank < 5: break
    return bank, hist

for T in (0.04, 0.02):
    B = bets_for(2025, T)
    amer = lambda d: f"+{round((d-1)*100)}" if d >= 2 else f"-{round(100/(d-1))}"
    if T == 0.04:
        print(f"=== SAMPLE BETS, 2025 season (EV>={T:.0%} vs Pinnacle fair, sentiment veto) ===")
        print(f"{'wk':>3} {'player':<22} {'market':<16} {'side':<5} {'line':>5} {'book':<12} {'odds':>5} {'EV':>5} {'res':<4}")
        for _, r in B.head(18).iterrows():
            print(f"{r.week:>3} {r.player_name[:22]:<22} {r.market.replace('player_',''):<16} "
                  f"{r.side:<5} {r.point:>5} {r.bookmaker:<12} {amer(r.dec):>5} {r.ev:>4.0%} "
                  f"{'WIN' if r.win else 'loss'}")
        wins = B.win.mean()
        print(f"  ... {len(B)} bets total in 2025 | win rate {wins:.0%} | avg odds {B.dec.mean():.2f} "
              f"| flat-stake ROI {np.mean(np.where(B.win, B.dec-1, -1.0)):+.1%}")
    print(f"\n=== $500 bankroll, 2025, EV>={T:.0%} + veto ({len(B)} bets, chronological) ===")
    for frac, lab in ((1.0, "full Kelly"), (0.5, "half Kelly"), (0.25, "quarter Kelly")):
        bank, hist = kelly(B, 500, frac)
        peak = max(b for _, b in hist); trough = min(b for _, b in hist)
        print(f"   {lab:<14} end ${bank:,.0f}   peak ${peak:,.0f}   trough ${trough:,.0f}")
    bank, hist = kelly(B, 500, 0.5)
    wk = {}
    for w, b in hist: wk[w] = b
    print("   half-Kelly bankroll by week: " + " ".join(f"w{w}:${b:,.0f}" for w, b in sorted(wk.items())))

# context: same sim on 2023/2024
print("\n=== context: other seasons, EV>=4% + veto, half Kelly from $500 ===")
for ssn in (2023, 2024):
    B = bets_for(ssn, 0.04)
    bank, _ = kelly(B, 500, 0.5)
    print(f"   {ssn}: {len(B)} bets -> end ${bank:,.0f}  (flat ROI {np.mean(np.where(B.win, B.dec-1, -1.0)):+.1%})")

with open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## 2025 bet log + $500 Kelly sim (`kelly_2025_sim.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/sentiment_props.md")
