"""Opening-line study: can we beat the OPENER instead of the close?
(a) How much do prop lines move open->close?
(b) Does OUR model's edge vs the opener predict the direction of movement (CLV)?
(c) ROI betting at the opening price when the model disagrees with the opener.
(d) Benchmark: does the close simply sharpen the open (steam-following)?
Uses the walk-forward mu/sig saved by prop_ev_model.py (outputs/prop_ev_backtest.pkl)."""
import os, sqlite3, warnings, sys, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
A = pd.read_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
MK = {"player_reception_yds": 0, "player_receptions": 1, "player_rush_yds": 0,
      "player_pass_yds": 0, "player_rush_attempts": 1, "player_pass_completions": 1}
mlist = ",".join("'" + m + "'" for m in MK)
con = sqlite3.connect(DB)
op = pd.read_sql(f"""
WITH r AS (SELECT event_id,bookmaker,market,player_name,outcome_type,price,point,snapshot_time,
   ROW_NUMBER() OVER (PARTITION BY event_id,bookmaker,market,player_name,outcome_type
                      ORDER BY snapshot_time ASC) rn
   FROM player_props WHERE market IN ({mlist}))
SELECT event_id,bookmaker,market,player_name,outcome_type,price,point,snapshot_time
FROM r WHERE rn=1""", con)
con.close()
op["dec"] = np.where(op.price > 0, 1 + op.price / 100.0, 1 + 100.0 / op.price.abs())
omed = op.groupby(["event_id", "market", "player_name"]).point.median().rename("oline").reset_index()
op = op.merge(omed, on=["event_id", "market", "player_name"])
op = op[op.point == op.oline]
piv = op.pivot_table(index=["event_id", "market", "player_name", "oline"],
                     columns="outcome_type", values="dec", aggfunc="median").reset_index()
piv = piv.dropna(subset=["Over", "Under"])
piv["o_po"] = 1 / piv.Over; piv["o_pu"] = 1 / piv.Under
piv["onovig"] = piv.o_po / (piv.o_po + piv.o_pu)
piv = piv.rename(columns={"Over": "oOver", "Under": "oUnder"})
# best opening price across books
ob = op.pivot_table(index=["event_id", "market", "player_name", "oline"],
                    columns="outcome_type", values="dec", aggfunc="max").reset_index() \
       .rename(columns={"Over": "oBestOver", "Under": "oBestUnder"})
piv = piv.merge(ob, on=["event_id", "market", "player_name", "oline"])
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
piv["k"] = piv.player_name.map(nrm)
D = A.merge(piv[["event_id", "market", "k", "oline", "onovig", "oOver", "oUnder", "oBestOver", "oBestUnder"]],
            on=["event_id", "market", "k"], how="inner")
print(f"matched open+close+model: {len(D):,} markets")
# (a) line movement
D["dline"] = D.mline - D.oline
print("\n(a) line movement open->close:")
for m, d in D.groupby("market"):
    print(f"  {m:26s} |move|>0: {(d.dline!=0).mean():.0%}  mean move {d.dline.mean():+.2f}  sd {d.dline.std():.2f}")
# model P(over) at the OPEN line
D["adj"] = D.market.map(MK) * 0.5
D["p_open"] = 1 - norm.cdf((D.oline + D.adj - D.mu) / D.sig)
D["oedge"] = D.p_open - D.onovig          # + = we like the OVER vs the opener
# (b) does our open-edge predict line movement? (line moves UP when the market agrees with our over-lean)
r = np.corrcoef(D.oedge, D.dline)[0, 1]
r2 = np.corrcoef(D.oedge, D.novig - D.onovig)[0, 1]
print(f"\n(b) corr(model edge vs opener, line movement): {r:+.3f}   (prob movement: {r2:+.3f})")
q = pd.qcut(D.oedge, 5, labels=False, duplicates="drop")
print(D.groupby(q).agg(edge=("oedge", "mean"), dline=("dline", "mean"), dprob=("novig", "mean"), n=("dline", "size"))
       .assign(dprob=lambda x: x.dprob - D.groupby(q).onovig.mean()).round(3).to_string())
# CLV: for our open bets, did the market move OUR way by close?
for th in (0.06, 0.12):
    b = D[D.oedge.abs() >= th].copy()
    b["side"] = np.where(b.oedge > 0, 1, -1)
    b["clv_line"] = np.where(b.side == 1, b.dline > 0, b.dline < 0)   # line moved our way
    b["clv_prob"] = np.where(b.side == 1, b.novig - b.onovig, b.onovig - b.novig)  # prob moved our way
    print(f"\n(b2) open-edge>={th:.0%}: n={len(b):5d}  line moved our way {b.clv_line.mean():.1%} "
          f"(vs against {(np.where(b.side==1,b.dline<0,b.dline>0)).mean():.1%})  mean prob-CLV {b.clv_prob.mean():+.3f}")
# (c) ROI betting the OPENER
print("\n(c) ROI at OPENING prices (settled on actuals):")
for th in (0.06, 0.12):
    for pricecol_o, pricecol_u, lab in [("oOver", "oUnder", "median-book"), ("oBestOver", "oBestUnder", "best-book")]:
        b = D[D.oedge.abs() >= th].copy()
        b["side"] = np.where(b.oedge > 0, 1, 0)
        b["won_o"] = (b.y > b.oline).astype(int); b = b[b.y != b.oline]
        b["win"] = (b.side == b.won_o).astype(int)
        b["dp"] = np.where(b.side == 1, b[pricecol_o], b[pricecol_u])
        b["pnl"] = np.where(b.win == 1, b.dp - 1, -1.0)
        print(f"  edge>={th:.0%} @{lab:11s}: n={len(b):5d}  hit {b.win.mean():.1%}  ROI {b.pnl.mean():+.1%}")
        # split by side
        for s, nm2 in [(0, "under"), (1, "over")]:
            bb = b[b.side == s]
            if len(bb) > 100:
                print(f"      {nm2}: n={len(bb):5d} ROI {bb.pnl.mean():+.1%}")
# (d) steam benchmark: bet the side the line MOVED toward, at open price (impossible in practice, sanity only)
b = D[D.dline != 0].copy()
b["side"] = np.where(b.dline > 0, 1, 0)
b["won_o"] = (b.y > b.oline).astype(int); b = b[b.y != b.oline]
b["win"] = (b.side == b.won_o).astype(int)
b["dp"] = np.where(b.side == 1, b.oOver, b.oUnder)
b["pnl"] = np.where(b.win == 1, b.dp - 1, -1.0)
print(f"\n(d) steam-follow sanity (close knows better than open): n={len(b)}, hit {b.win.mean():.1%}, ROI {b.pnl.mean():+.1%}")
D.to_pickle(os.path.join(ROOT, "outputs", "prop_open_clv.pkl"))
