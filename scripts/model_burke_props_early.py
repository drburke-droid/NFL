"""Early-vs-close test with the snapshots we hold.

'Early' = each book's FIRST captured snapshot, only for events where that
snapshot leads kickoff by >= 5h (median lead in the kept set ~6-25h).
The model's baseline is the EARLY line (a close baseline would be lookahead
at early-bet time). Same calibrated P(over) layer as v2; bets priced and
de-vigged at the early prices. Benchmark: the v2 close-line sim restricted
to the same events.
"""
import os, re, sys, sqlite3, warnings
from functools import partial
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "model_burke", "pkg"))
from model_burke import pipeline, residual, point
# thinner frames: lower the walk-forward minimums proportionally
pipeline.fit_residual_model = partial(residual.fit_residual_model, min_train=800)
pipeline.fit_point_model = partial(point.fit_point_model, min_train=600, min_pos_train=100)

DB = r"C:\Users\drbur\Documents\GitHub\NFL\db\nfl_odds.db"
MARKETS = {"player_reception_yds": "receiving_yards",
           "player_receptions": "receptions",
           "player_rush_yds": "rushing_yards"}
LEAD_MIN_H = 5.0

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)

def amer_imp(o):
    o = float(o); return 100 / (o + 100) if o > 0 else -o / (-o + 100)
def amer_profit(o):
    o = float(o); return o / 100 if o > 0 else 100 / -o

con = sqlite3.connect(DB)
games = pd.read_sql("SELECT event_id, commence_time, season, week FROM games", con)
games["season"] = pd.to_numeric(games.season, errors="coerce")
games["week"] = pd.to_numeric(games.week, errors="coerce")
games = games.dropna(subset=["season", "week"]).astype({"season": int, "week": int})
wk = pd.read_sql("""SELECT * FROM nflv_weekly WHERE season BETWEEN 2022 AND 2025
                    AND position IN ('QB','RB','WR','TE')""", con)
wk = wk.drop_duplicates(["player_id", "season", "week"])
from model_burke.features import build_lagged_features
lag = build_lagged_features(wk)
wk["nname"] = wk.player_display_name.map(norm)

def consensus(pp, which):
    """which='first' or 'last' snapshot per (event, book, player, side)."""
    asc = which == "first"
    pp = pp.copy()
    pp["rk"] = pp.groupby(["event_id", "bookmaker", "player_name", "outcome_type"]).snapshot_time.rank(
        ascending=asc, method="first")
    pp = pp[pp.rk == 1]
    o = pp[pp.outcome_type == "Over"]; u = pp[pp.outcome_type == "Under"]
    g = ["event_id", "player_name", "season", "week"]
    return pd.concat([o.groupby(g).point.median().rename("line"),
                      o.groupby(g).price.median().rename("over_price"),
                      u.groupby(g).price.median().rename("under_price"),
                      o.groupby(g).snapshot_time.min().rename("snap")], axis=1).reset_index()

def run_variant(L, statcol, label):
    L = L.copy(); L["nname"] = L.player_name.map(norm)
    d = L.merge(wk[["nname", "season", "week", "player_id", "player_display_name",
                    "position", statcol]], on=["nname", "season", "week"], how="left")
    d = d.dropna(subset=["player_id"])
    d = d.rename(columns={statcol: "actual_ppr", "line": "baseline_proj",
                          "player_display_name": "player"})
    d = d.merge(lag, on=["player_id", "season", "week"], how="left")
    d = d.drop_duplicates(["player_id", "season", "week"])
    d["market_proj"] = np.nan
    ev, _ = pipeline.run(d, verbose=False, use_wind=False)
    ev = ev.dropna(subset=["Model_Burke_mean", "mb_p25", "mb_p75",
                           "actual_ppr", "baseline_proj"]).copy()
    ev["sd"] = ((ev.mb_p75 - ev.mb_p25) / 1.35).clip(lower=1e-3)
    ev["z"] = ((ev.Model_Burke_mean - ev.baseline_proj) / ev.sd).clip(-4, 4)
    ev["went_over"] = (ev.actual_ppr > ev.baseline_proj).astype(int)
    ev["is_push"] = ev.actual_ppr == ev.baseline_proj
    ev = ev.sort_values("t")
    ev["p_over2"] = np.nan
    for t in sorted(ev.t.unique()):
        tr = ev[(ev.t < t) & (~ev.is_push)]
        te = ev.index[ev.t == t]
        if len(tr) < 300 or not len(te): continue
        lr = LogisticRegression(C=1.0).fit(tr[["z", "baseline_proj"]].values, tr.went_over.values)
        ev.loc[te, "p_over2"] = lr.predict_proba(ev.loc[te, ["z", "baseline_proj"]].values)[:, 1]
    e = ev[(ev.season == 2025) & ev.p_over2.notna()].dropna(subset=["over_price", "under_price"])
    e = e[(e.over_price.abs() >= 100) & (e.under_price.abs() >= 100)].copy()
    io = e.over_price.map(amer_imp); iu = e.under_price.map(amer_imp)
    e["fair_over"] = io / (io + iu)
    e["edge"] = e.p_over2 - e.fair_over
    rows = []
    for thr in (0.03, 0.05, 0.08):
        b = e[e.edge.abs() >= thr].copy()
        if not len(b): continue
        b["side"] = np.where(b.edge > 0, "Over", "Under")
        b["win"] = np.where(b.side == "Over", b.actual_ppr > b.baseline_proj,
                            b.actual_ppr < b.baseline_proj)
        b["price"] = np.where(b.side == "Over", b.over_price, b.under_price)
        b["profit"] = np.where(b.is_push, 0.0,
                       np.where(b.win, b.price.map(amer_profit), -1.0))
        rows.append({"variant": label, "thr": thr, "bets": len(b),
                     "win%": b[~b.is_push].win.mean(),
                     "roi": b.profit.sum() / len(b), "units": b.profit.sum()})
    return rows

out = []
for mkt, statcol in MARKETS.items():
    pp = pd.read_sql(f"""SELECT event_id, bookmaker, player_name, outcome_type, price,
                         point, snapshot_time FROM player_props WHERE market='{mkt}'""", con)
    pp = pp.merge(games, on="event_id")
    cm = pd.read_sql("SELECT event_id, commence_time FROM games", con)
    pp = pp[pp.snapshot_time <= pp.commence_time]
    # events with a real early snapshot
    lead = pp.groupby("event_id").agg(first=("snapshot_time", "min"),
                                      commence=("commence_time", "first"))
    lead["h"] = (pd.to_datetime(lead.commence) - pd.to_datetime(lead["first"])).dt.total_seconds() / 3600
    early_ev = set(lead.index[lead.h >= LEAD_MIN_H])
    pe = pp[pp.event_id.isin(early_ev)]
    print(f"\n### {mkt}: {len(early_ev)} early events "
          f"(median lead {lead.h[lead.h>=LEAD_MIN_H].median():.1f}h)")
    for which, label in (("first", "EARLY"), ("last", "CLOSE")):
        L = consensus(pe, which)
        out += run_variant(L, statcol, f"{mkt[7:]}|{label}")

R = pd.DataFrame(out)
print("\n========== EARLY vs CLOSE, same events, 2025 ==========")
print(R.round(3).to_string(index=False))
con.close()
