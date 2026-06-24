"""
Half-season role/production TREND features, all seasons (2011-2025).

For each (player, played-season S) we summarize how their role moved from the
first half (wk 1-9) to the second half (wk 10-18) of S, plus their END-of-season
role level. Keyed by season = S+1 so it merges straight onto season_dataset
(whose `season` is the TARGET year and prior_* come from S) with NO leakage —
every value is known before the predicted season starts.

Features (prefix ht_):
  ht_d_ppg     H2 minus H1 fantasy PPG (your league scoring)
  ht_h2_ppg    second-half PPG (recent form / end-of-year role)
  ht_d_snap    snap-share change (pp, 2013+)
  ht_h2_snap   second-half snap share (pp)
  ht_d_tch     touches/g change (carries+targets)
  ht_d_tgtsh   target-share change (pp)
  ht_h2_tgtsh  second-half target share (pp)
  ht_slope     per-week OLS slope of weekly points (momentum)

Writes nflv_half_trend.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
CUT = 9


def fpts(d):
    return (d.passing_yards.fillna(0)*0.04 + d.passing_tds.fillna(0)*6 - d.passing_interceptions.fillna(0)
            + d.rushing_yards.fillna(0)*0.1 + d.rushing_tds.fillna(0)*6 - d.rushing_fumbles.fillna(0)*2
            + d.receiving_yards.fillna(0)*0.1 + d.receiving_tds.fillna(0)*6 + d.receptions.fillna(0))


def main():
    con = sqlite3.connect(DB)
    w = pd.read_sql("""SELECT * FROM nflv_weekly
                       WHERE season_type='REG' AND week<=18
                       AND position IN ('QB','RB','WR','TE')""", con)
    sn = pd.read_sql("""SELECT season,week,player,team,offense_pct FROM nflv_snaps
                        WHERE game_type='REG' AND week<=18""", con)
    con.close()
    nm = lambda s: s.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    w["_k"] = nm(w.player_display_name) + "|" + w.team + "|" + w.season.astype(str)
    sn["_k"] = nm(sn.player) + "|" + sn.team + "|" + sn.season.astype(str)
    snap = sn.groupby(["_k", "week"])["offense_pct"].max().reset_index()
    w = w.merge(snap, on=["_k", "week"], how="left")

    w["fp"] = fpts(w)
    w["touches"] = w.carries.fillna(0) + w.targets.fillna(0)
    w["half"] = np.where(w.week <= CUT, "H1", "H2")

    def agg_half(d):
        return pd.Series({"g": d.week.nunique(), "ppg": d.fp.mean(), "tch": d.touches.mean(),
                          "tgtsh": d.target_share.mean(), "snap": d.offense_pct.mean()})
    g = w.groupby(["player_id", "position", "season", "half"]).apply(agg_half).reset_index()
    piv = g.pivot_table(index=["player_id", "position", "season"], columns="half",
                        values=["g", "ppg", "tch", "tgtsh", "snap"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    # per-week slope of points (momentum), needs >=4 weeks
    def slope(d):
        d = d.dropna(subset=["fp"])
        if d.week.nunique() < 4: return np.nan
        x = d.week.values.astype(float); y = d.fp.values
        return float(np.polyfit(x, y, 1)[0])
    sl = w.groupby(["player_id", "season"]).apply(slope).reset_index(name="ht_slope")
    piv = piv.merge(sl, on=["player_id", "season"], how="left")

    for c in ["snap_H1", "snap_H2"]:
        piv[c] = piv[c] * 100
    out = pd.DataFrame({
        "player_id": piv.player_id, "season": piv.season + 1,            # -> target season
        "ht_d_ppg": (piv.ppg_H2.fillna(0) - piv.ppg_H1.fillna(0)),
        "ht_h2_ppg": piv.ppg_H2,
        "ht_d_snap": (piv.snap_H2 - piv.snap_H1),
        "ht_h2_snap": piv.snap_H2,
        "ht_d_tch": (piv.tch_H2.fillna(0) - piv.tch_H1.fillna(0)),
        "ht_d_tgtsh": (piv.tgtsh_H2 - piv.tgtsh_H1) * 100,
        "ht_h2_tgtsh": piv.tgtsh_H2 * 100,
        "ht_slope": piv.ht_slope,
    })
    # require a real second half so the trend means something
    out = out[(piv.g_H2.fillna(0) >= 2).values].copy()
    out = out.round(3)

    con = sqlite3.connect(DB)
    out.to_sql("nflv_half_trend", con, if_exists="replace", index=False)
    con.close()
    print(f"nflv_half_trend: {len(out):,} player-seasons ({int(out.season.min())}-{int(out.season.max())})")
    print(out[["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
