"""
Props-based weekly fantasy projection (2023-2025) for the realistic league sim.

Converts the betting player-prop lines into an expected PPR fantasy projection per
player-week — the same number an informed manager would use for start/sit and
waivers. Median line across books/snapshots per (game, player, market).

Markets -> PPR points:
  pass_yds*.04, pass_tds*4, pass_interceptions*-2, rush_yds*.1, reception_yds*.1,
  receptions*1, anytime_td (Yes prob from price)*6 [covers rush+rec TDs],
  kicking_points (K).
Names mapped to gsis player_id via nflv_weekly that season. Writes weekly_proj_props.
"""
import os, sqlite3, warnings, sys, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
norm = lambda s: re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("'", "")).strip()


def amer_prob(price):
    try:
        p = float(price)
    except Exception:
        return np.nan
    return 100.0 / (p + 100.0) if p > 0 else (-p) / ((-p) + 100.0)


def main():
    con = sqlite3.connect(DB)
    g = pd.read_sql("SELECT event_id, season, week FROM games WHERE season IS NOT NULL", con)
    pp = pd.read_sql("""SELECT event_id, market, player_name, outcome_type, price, point
                        FROM player_props""", con)
    wk = pd.read_sql("SELECT DISTINCT season, player_id, player_display_name nm FROM nflv_weekly", con)
    con.close()
    pp = pp.merge(g, on="event_id", how="inner")
    pp["k"] = pp.market.str.replace("player_", "", regex=False)

    # yardage/count markets: median Over line point per (season,week,player,market)
    cnt = pp[pp.outcome_type.isin(["Over", "Under"]) | pp.point.notna()]
    agg = cnt.groupby(["season", "week", "player_name", "k"])["point"].median().reset_index()
    wide = agg.pivot_table(index=["season", "week", "player_name"], columns="k", values="point").reset_index()
    # anytime TD probability from Yes price
    td = pp[pp.k == "anytime_td"].copy(); td["prob"] = td.price.map(amer_prob)
    tda = td.groupby(["season", "week", "player_name"])["prob"].median().reset_index().rename(columns={"prob": "td_prob"})
    w = wide.merge(tda, on=["season", "week", "player_name"], how="left")

    def col(c): return w[c] if c in w.columns else 0.0
    proj = (col("pass_yds").fillna(0)*0.04 + col("pass_tds").fillna(0)*4 - col("pass_interceptions").fillna(0)*2
            + col("rush_yds").fillna(0)*0.1 + col("reception_yds").fillna(0)*0.1 + col("receptions").fillna(0)*1
            + w.get("td_prob", pd.Series(0, index=w.index)).fillna(0)*6)
    # kickers: kicking_points line is ~the projection directly
    if "kicking_points" in w.columns:
        proj = np.where(w["kicking_points"].notna(), w["kicking_points"].fillna(0), proj)
    w["proj_pts"] = np.clip(proj, 0, None)
    w["nmn"] = w.player_name.map(norm)

    # map name -> player_id within season
    wk["nmn"] = wk.nm.map(norm)
    namemap = {(r.season, r.nmn): r.player_id for r in wk.itertuples()}
    w["player_id"] = [namemap.get((s, n)) for s, n in zip(w.season, w.nmn)]
    out = w[w.player_id.notna()][["season", "week", "player_id", "proj_pts"]].copy()
    out["proj_pts"] = out.proj_pts.round(2)

    con = sqlite3.connect(DB); out.to_sql("weekly_proj_props", con, if_exists="replace", index=False); con.close()
    matched = w.player_id.notna().mean()
    print(f"weekly_proj_props: {len(out):,} player-weeks ({int(out.season.min())}-{int(out.season.max())}), name match {matched:.0%}")
    s = out[out.season == 2024]
    top = s.groupby("player_id").proj_pts.mean().sort_values(ascending=False).head(8)
    nm = wk[wk.season == 2024].set_index("player_id").nm.to_dict()
    print("2024 highest avg weekly projection:")
    for pid, v in top.items(): print(f"   {v:5.1f}  {nm.get(pid, pid)}")


if __name__ == "__main__":
    main()
