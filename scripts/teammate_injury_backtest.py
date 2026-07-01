"""
Backtest: do pass-catchers whose starting QB missed an extended stretch REBOUND above their
depressed line the next year (when the QB is typically healthy again)?

For flag years Y in 2018-2023 (outcome Y+1 in 2019-2024, weekly endpoint works pre-2025):
  FLAGGED = pass-catchers whose team's primary QB (most attempts) missed >= 3 games, who were real
            contributors WITH the QB (in-PPG >= 8) and dropped materially without him (out/in<=0.78).
            Record blended_Y (their full-season PPG) and inPPG_Y (their with-QB rate).
  CONTROL = pass-catchers (full-season PPG >= 8) whose primary QB was stable (played >= 15 games).
Compare each group's next-year (Y+1) PPG. If the concept holds, flagged players beat their blended
line by MORE than the control regresses, and their with-QB rate predicts Y+1 better than blended.
(PPR scoring; WR/RB/TE; >=6 games to count a season. Survivorship affects both groups equally.)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2018, 2025))
w = nfl.import_weekly_data(YEARS)
w = w[w["season_type"] == "REG"].copy()
w["fp"] = (w["receptions"].fillna(0) + w["receiving_yards"].fillna(0) * 0.1 + w["receiving_tds"].fillna(0) * 6
           + w["rushing_yards"].fillna(0) * 0.1 + w["rushing_tds"].fillna(0) * 6)
TEAM = "recent_team"

# season PPG per player-year (>=6 games)
seas = w.groupby(["player_id", "player_display_name", "position", "season"]).agg(
    fp=("fp", "sum"), g=("fp", "size"), team=(TEAM, lambda s: s.mode().iat[0])).reset_index()
seas = seas[seas["g"] >= 6]; seas["ppg"] = seas["fp"] / seas["g"]
ppg_next = {(r.player_id, r.season): r.ppg for r in seas.itertuples()}   # (pid, year) -> ppg

# primary QB per team-year + weeks he was out
qb = w[(w["position"] == "QB") & (w["attempts"].fillna(0) > 0)]
prim = qb.groupby([TEAM, "season", "player_id"])["attempts"].sum().reset_index()
prim = prim.sort_values("attempts").groupby([TEAM, "season"]).tail(1).set_index([TEAM, "season"])["player_id"]
qb_wk = qb.groupby([TEAM, "season", "week", "player_id"])["attempts"].sum()
team_weeks = w.groupby([TEAM, "season"])["week"].apply(lambda s: set(s.unique()))

def qb_split(team, yr):
    """return (out_count, in_weeks, out_weeks) for the primary QB of team-year, else None."""
    key = (team, yr)
    if key not in prim.index: return None
    qid = prim.loc[key]; wks = team_weeks.get(key, set())
    inw = {ww for ww in wks if qb_wk.get((team, yr, qid, ), None) is not None and qb_wk.get((team, yr, ww, qid) if False else (team, yr, ww, qid), 0)}
    inw = {ww for ww in wks if qb_wk.get((team, yr, ww, qid), 0) >= 10}
    return len(wks - inw), inw, (wks - inw)

flag, ctrl = [], []
for yr in range(2018, 2024):
    yw = w[w["season"] == yr]
    for r in seas[(seas["season"] == yr) & (seas["position"].isin(["WR", "RB", "TE"]))].itertuples():
        nxt = ppg_next.get((r.player_id, yr + 1))
        if nxt is None: continue                                  # must have played Y+1
        sp = qb_split(r.team, yr)
        if sp is None: continue
        outc, inw, outw = sp
        sub = yw[yw["player_id"] == r.player_id]
        ins = sub[sub["week"].isin(inw)]["fp"]; outs = sub[sub["week"].isin(outw)]["fp"]
        if outc >= 3 and len(ins) >= 5 and len(outs) >= 2 and ins.mean() >= 8 and (outs.mean() / ins.mean()) <= 0.78:
            flag.append({"blended": r.ppg, "inppg": ins.mean(), "next": nxt})
        elif outc <= 2 and r.ppg >= 8:                            # stable-QB control
            ctrl.append({"blended": r.ppg, "next": nxt})

F, C = pd.DataFrame(flag), pd.DataFrame(ctrl)
print(f"Flagged (QB-injury-depressed) player-seasons: {len(F)}   |   Control (stable QB): {len(C)}\n")
print("Group             Yr blended  with-QB   next-yr   next-blended   (rebound)")
print(f"  FLAGGED  n={len(F):<4}   {F.blended.mean():6.1f}   {F.inppg.mean():6.1f}   {F.next.mean():6.1f}     {F.next.mean()-F.blended.mean():+6.1f}")
print(f"  CONTROL  n={len(C):<4}   {C.blended.mean():6.1f}      —      {C.next.mean():6.1f}     {C.next.mean()-C.blended.mean():+6.1f}")
print(f"\n  Concept edge (flagged rebound - control change): {(F.next.mean()-F.blended.mean())-(C.next.mean()-C.blended.mean()):+.1f} PPG")
print(f"  % of flagged who beat their (depressed) blended line next year: {(F.next>F.blended).mean()*100:.0f}%  "
      f"(control: {(C.next>C.blended).mean()*100:.0f}%)")
print(f"\n  Predicting next-yr PPG for flagged players — MAE:")
print(f"     from blended (depressed) line : {(F.next-F.blended).abs().mean():.2f}")
print(f"     from with-QB rate             : {(F.next-F.inppg).abs().mean():.2f}   <- lower = the adjustment helps")
mid = (F.blended + F.inppg) / 2
print(f"     from 50/50 blend of the two   : {(F.next-mid).abs().mean():.2f}")
