"""
When a WR changes teams, what happens to usage — his, his new teammates', his replacement's, his
old teammates'? Tests whether target share is "conserved" and how it redistributes.

Metric: target share = a player's targets / his team's total targets that season (usage, not luck).
Movers = WR with a real role (target share >= 8%, >=6 games) who changed primary team from year Y
to Y+1 (2018->2024). For each we look at:
  Q1 mover:            his target share & PPR PPG on the OLD team (Y) vs the NEW team (Y+1).
  Q2 new teammates:    WRs who were on the NEW team both years — do they dip in Y+1?
  Q3 replacement:      on the OLD team in Y+1, the biggest target-share gainer — how much of the
                       mover's vacated share does one player absorb (concentrated vs spread)?
  Q4 old teammates:    WRs who stayed on the OLD team both years — do they get a boost?
Control = "stayers" (same team both years) for the mover's own YoY change. PPR, nfl_data_py weekly.
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
pj = w.groupby(["player_id", "player_display_name", "position", "season"]).agg(
    tgt=("targets", "sum"), fp=("fp", "sum"), g=("fp", "size"),
    team=("recent_team", lambda s: s.mode().iat[0])).reset_index()
pj = pj.merge(team_tgt, left_on=["team", "season"], right_on=["recent_team", "season"])
pj["ts"] = pj["tgt"] / pj["team_tgt"]; pj["ppg"] = pj["fp"] / pj["g"]
wr = pj[(pj["position"] == "WR") & (pj["g"] >= 6)].copy()
bykey = {(r.player_id, r.season): r for r in wr.itertuples()}
team_wr = {}                                        # (team, season) -> list of WR rows
for r in wr.itertuples(): team_wr.setdefault((r.team, r.season), []).append(r)

mover, stayer = [], []
for r in wr[wr["ts"] >= 0.08].itertuples():
    nx = bykey.get((r.player_id, r.season + 1))
    if nx is None: continue
    if nx.team == r.team:                           # stayer (control for Q1)
        stayer.append({"dts": nx.ts - r.ts, "dppg": nx.ppg - r.ppg}); continue
    Y, A, B = r.season, r.team, nx.team
    # Q2: new-team WRs present both years (holdovers), excluding the mover
    newhold = [(p, bykey.get((p.player_id, Y + 1))) for p in team_wr.get((B, Y), []) if p.player_id != r.player_id]
    q2 = sum((n.ts - p.ts) for p, n in newhold if n and n.team == B) if newhold else np.nan
    # Q4: old-team WRs present both years (holdovers), excluding the mover
    oldhold = [(p, bykey.get((p.player_id, Y + 1))) for p in team_wr.get((A, Y), []) if p.player_id != r.player_id]
    q4 = sum((n.ts - p.ts) for p, n in oldhold if n and n.team == A) if oldhold else np.nan
    # Q3: on the OLD team in Y+1, biggest target-share gainer vs Y (new WRs count as ts_Y=0)
    prevA = {p.player_id: p.ts for p in team_wr.get((A, Y), [])}
    gains = [(p.ts - prevA.get(p.player_id, 0.0)) for p in team_wr.get((A, Y + 1), []) if p.player_id != r.player_id]
    q3 = (max(gains) / r.ts) if gains and r.ts > 0 else np.nan     # fraction of mover's share one player absorbed
    mover.append({"mts_Y": r.ts, "mts_Y1": nx.ts, "mppg_Y": r.ppg, "mppg_Y1": nx.ppg,
                  "dts": nx.ts - r.ts, "dppg": nx.ppg - r.ppg, "q2": q2, "q4": q4, "q3": q3})

M, S = pd.DataFrame(mover), pd.DataFrame(stayer)
pct = lambda x: f"{x*100:+.1f}pp"
print(f"WR movers analyzed: {len(M)}   |   stayer controls: {len(S)}\n")
print("Q1 — THE MOVER (old team Y -> new team Y+1):")
print(f"   target share {M.mts_Y.mean()*100:.1f}% -> {M.mts_Y1.mean()*100:.1f}%  ({pct(M.dts.mean())})   "
      f"PPG {M.mppg_Y.mean():.1f} -> {M.mppg_Y1.mean():.1f} ({M.dppg.mean():+.1f})")
print(f"   control stayers (same team): target share {pct(S.dts.mean())},  PPG {S.dppg.mean():+.1f}")
print(f"\nQ2 — NEW TEAM's incumbent WRs (holdovers): combined target share {pct(M.q2.mean())}  "
      f"(negative = they dip as the mover takes targets)")
print(f"Q4 — OLD TEAM's remaining WRs (holdovers): combined target share {pct(M.q4.mean())}  "
      f"(positive = they get a boost)")
print(f"\nQ3 — REPLACEMENT on the old team: the single biggest gainer absorbs "
      f"{np.nanmean(M.q3)*100:.0f}% of the mover's vacated share on average")
print(f"   (100% = one player fully inherits the role; ~50% = split between two; lower = spread around)")
print(f"   share of moves where one player absorbs >=60% of the role: {(M.q3>=0.6).mean()*100:.0f}%")
