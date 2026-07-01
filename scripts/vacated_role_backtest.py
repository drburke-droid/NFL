"""
Backtest: does the presumptive inheritor of a vacated WR role beat projection the next year?

Actionable (ex-ante) test — no look-ahead: after year Y, a team loses a WR with target share >= 12%
(traded/FA/etc, not on the team in Y+1). The presumptive inheritor = the top RETURNING WR on that
team by year-Y target share, in a WR2/3 range (6-18%, i.e. room to ascend). Do these players beat a
naive (last-year) projection in Y+1, MORE than a role-matched control (same ts_Y range, on teams
that did NOT lose a WR)? Matching on prior role strips out plain mean-reversion.

Also reported: the look-ahead "actual biggest gainer" (upper bound), and whether inheritors' target
share actually rose (mechanism check). PPR, nfl_data_py weekly 2018-2024.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2018, 2025))
w = nfl.import_weekly_data(YEARS); w = w[w["season_type"] == "REG"].copy()
for c in ["targets", "receptions", "receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds"]:
    w[c] = w[c].fillna(0)
w["fp"] = w["receptions"] + w["receiving_yards"] * 0.1 + w["receiving_tds"] * 6 + w["rushing_yards"] * 0.1 + w["rushing_tds"] * 6
team_tgt = w.groupby(["recent_team", "season"])["targets"].sum().rename("team_tgt")
pj = w.groupby(["player_id", "position", "season"]).agg(
    tgt=("targets", "sum"), fp=("fp", "sum"), g=("fp", "size"),
    team=("recent_team", lambda s: s.mode().iat[0])).reset_index().merge(
    team_tgt, left_on=["team", "season"], right_on=["recent_team", "season"])
pj["ts"] = pj["tgt"] / pj["team_tgt"]; pj["ppg"] = pj["fp"] / pj["g"]
wr = pj[(pj["position"] == "WR") & (pj["g"] >= 6)]
bykey = {(r.player_id, r.season): r for r in wr.itertuples()}
team_wr = {}
for r in wr.itertuples(): team_wr.setdefault((r.team, r.season), []).append(r)
LO, HI = 0.06, 0.18                                       # WR2/3 role band (room to ascend)

inh, actual, ctrl = [], [], []
for (A, Y), roster in team_wr.items():
    if Y + 1 > 2024: continue
    hold = [p for p in roster if bykey.get((p.player_id, Y + 1)) and bykey[(p.player_id, Y + 1)].team == A]
    departed = [p for p in roster if p.ts >= 0.12 and (bykey.get((p.player_id, Y + 1)) is None or bykey[(p.player_id, Y + 1)].team != A)]
    band = [p for p in hold if LO <= p.ts <= HI]
    if departed:                                          # team lost a real WR role
        if band:                                          # presumptive inheritor = top-ts returning WR in band
            m = max(band, key=lambda p: p.ts); n = bykey[(m.player_id, Y + 1)]
            inh.append({"ppgY": m.ppg, "ppgY1": n.ppg, "dts": n.ts - m.ts})
        if hold:                                          # look-ahead: actual biggest ts gainer (upper bound)
            g = max(hold, key=lambda p: bykey[(p.player_id, Y + 1)].ts - p.ts); n = bykey[(g.player_id, Y + 1)]
            if LO <= g.ts <= HI: actual.append({"ppgY": g.ppg, "ppgY1": n.ppg, "dts": n.ts - g.ts})
    else:                                                 # control: role-matched holdovers, no departure
        for p in band:
            n = bykey[(p.player_id, Y + 1)]; ctrl.append({"ppgY": p.ppg, "ppgY1": n.ppg, "dts": n.ts - p.ts})

I, AC, C = pd.DataFrame(inh), pd.DataFrame(actual), pd.DataFrame(ctrl)
def line(lbl, d):
    print(f"  {lbl:34s} n={len(d):<4} ts {d.dts.mean()*100:+.1f}pp  PPG {d.ppgY.mean():.1f}->{d.ppgY1.mean():.1f} "
          f"({d.ppgY1.mean()-d.ppgY.mean():+.1f})  beat-line {(d.ppgY1>d.ppgY).mean()*100:.0f}%")
print(f"Vacated-role backtest — WR2/3 ({int(LO*100)}-{int(HI*100)}% target share) next-year outcome\n")
line("PRESUMPTIVE inheritor (ex-ante)", I)
line("actual biggest gainer (look-ahead)", AC)
line("CONTROL (role-matched, no vacancy)", C)
edge = (I.ppgY1.mean()-I.ppgY.mean()) - (C.ppgY1.mean()-C.ppgY.mean())
print(f"\n  Actionable edge (presumptive inheritor rebound - control): {edge:+.2f} PPG")
print(f"  MAE predicting inheritor Y+1 from last-year PPG: {(I.ppgY1-I.ppgY).abs().mean():.2f}   "
      f"from last-year +{max(edge,0):.1f} uplift: {(I.ppgY1-(I.ppgY+max(edge,0))).abs().mean():.2f}")
