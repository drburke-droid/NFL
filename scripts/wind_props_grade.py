"""
Season-to-date grade of the Wind Props rule: UNDER on DK pass yds / completions / attempts / receptions /
rec yds when the kickoff->+3h FORECAST wind at an outdoor stadium is >= 10 kn.

Lines   data/props_frames/snapshots/props_<S>.csv -- the last DK two-sided line pulled before kickoff
Results nflverse stats_player_week_<S>.parquet (no box-score row = did not play = void; pushes void)
Wind    open-meteo historical-forecast archive (the forecast runs as they stood, the same model the tab's
        Check forecasts calls), mean of kickoff hour .. +3 h, knots -- what you would have seen pregame.

    python scripts/wind_props_grade.py [--season 2026] [--snapshots f] [--stats f] [--games f]
"""
import argparse, json, os, re, time, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAD = {"BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
        "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DEN": (39.7439, -105.0201), "GB": (44.5013, -88.0622),
        "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839), "MIA": (25.9580, -80.2389), "NE": (42.0909, -71.2643),
        "NYG": (40.8128, -74.0742), "NYJ": (40.8128, -74.0742), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
        "SF": (37.4032, -121.9698), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713),
        "WAS": (38.9077, -76.8645)}
STAT = {"player_pass_yds": ["passing_yards"], "player_pass_completions": ["completions"],
        "player_pass_attempts": ["attempts"], "player_receptions": ["receptions"],
        "player_reception_yds": ["receiving_yards"]}
LABEL = {"player_pass_yds": "pass yds", "player_pass_completions": "completions", "player_pass_attempts": "pass att",
         "player_receptions": "receptions", "player_reception_yds": "rec yds"}


def norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("'", ""))
    return re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)).strip()


def payout(p):
    p = np.asarray(p, float); return np.where(p > 0, p / 100, 100 / -p)


def forecast_wind(games, cache_path):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = {}
    for r in games.itertuples():
        if r.game_id in cache: out[r.game_id] = cache[r.game_id]; continue
        la, lo = STAD[r.home_team]
        day = r.ko.strftime("%Y-%m-%d"); day2 = (r.ko + pd.Timedelta(hours=4)).strftime("%Y-%m-%d")
        url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}"
               f"&start_date={day}&end_date={day2}&hourly=wind_speed_10m,wind_gusts_10m&windspeed_unit=kn&timezone=UTC")
        for i in range(3):
            try:
                H = json.load(urllib.request.urlopen(url, timeout=30))["hourly"]; break
            except Exception:
                time.sleep(2 * (i + 1)); H = None
        if not H: continue
        idx = {t: i for i, t in enumerate(H["time"])}
        k = r.ko.floor("h")
        w = [H["wind_speed_10m"][idx[(k + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:00")]]
             for h in range(4) if (k + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") in idx]
        w = [x for x in w if x is not None]
        if w: out[r.game_id] = cache[r.game_id] = round(float(np.mean(w)), 1)
        time.sleep(0.15)
    json.dump(cache, open(cache_path, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026); ap.add_argument("--snapshots"); ap.add_argument("--stats")
    ap.add_argument("--games"); ap.add_argument("--min-wind", type=float, default=10.0)
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "props_frames", "wind_fc_cache.json"))
    A = ap.parse_args()
    S = A.season
    p = pd.read_csv(A.snapshots or os.path.join(ROOT, "data", "props_frames", "snapshots", f"props_{S}.csv"))
    st = pd.read_parquet(A.stats or os.path.join(ROOT, "data", "nflverse_cache", f"stats_player_week_{S}.parquet"))
    g = pd.read_csv(A.games or "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
    g = g[(g.season == S) & g.home_score.notna()].copy()
    p = p[p.market.isin(STAT) & (p.price.abs() >= 100)].copy()
    p["ko"] = pd.to_datetime(p.commence, utc=True); p["snap"] = pd.to_datetime(p.snapshot_utc, utc=True)
    p = p[p.snap <= p.ko]
    p = p.sort_values("snap").groupby(["event", "market", "player", "side"]).tail(1)
    u = p[p.side == "Under"]; o = p[p.side == "Over"]
    k = ["event", "market", "player"]
    d = u.merge(o[k + ["point", "price"]], on=k, suffixes=("", "_o")); d = d[d.point == d.point_o]
    # results: match the player's box-score row on the kickoff's Eastern date
    st = st[st.season_type == "REG"].copy(); st["nm"] = st.player_display_name.map(norm)
    g["ko"] = [pd.Timestamp(f"{a} {b}", tz="America/New_York").tz_convert("UTC") for a, b in zip(g.gameday, g.gametime)]
    st = st.merge(g[["game_id", "ko", "home_team", "roof"]], on="game_id")
    d["nm"] = d.player.map(norm); d["day"] = (d.ko - pd.Timedelta(hours=5)).dt.strftime("%Y-%m-%d")
    st["day"] = (st.ko - pd.Timedelta(hours=5)).dt.strftime("%Y-%m-%d")
    m = d.merge(st, on=["nm", "day"], how="inner", suffixes=("", "_st"))
    m = m.drop_duplicates(k, keep=False)
    m["actual"] = [sum(row[c] or 0 for c in STAT[mk]) for mk, row in zip(m.market, m[sum(STAT.values(), [])].to_dict("records"))]
    m = m[m.actual != m.point]
    m["under_won"] = (m.actual < m.point).astype(int)
    m["pnl"] = np.where(m.under_won == 1, payout(m.price), -1.0)
    out_g = g[g.home_team.isin(STAD) & g.roof.isin(["outdoors", "open"])]
    fc = forecast_wind(out_g, A.cache)
    m["wind_fc"] = m.game_id.map(fc)
    m["outdoor"] = m.game_id.isin(out_g.game_id)
    m["band"] = pd.cut(m.wind_fc, [-1, 4, 8, 10, 13, 15, 60], labels=["0-4", "4-8", "8-10", "10-13", "13-15", "15+"])
    print(f"{S} season to date: {len(m):,} graded DK props in these markets, weeks {m.week.min()}-{m.week.max()}, "
          f"{m.game_id.nunique()} games; forecasts for {len(fc)} outdoor games")
    def line(x, name):
        if not len(x): return f"  {name:28s} n=0"
        se = x.pnl.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        return f"  {name:28s} n={len(x):4d}  unders {x.under_won.mean():5.1%}  ROI {x.pnl.mean():+6.1%}  (±{se:.1%})  units {x.pnl.sum():+6.1f}"
    rule = m[m.outdoor & (m.wind_fc >= A.min_wind)]
    print(f"\nTHE RULE: unders, outdoor, forecast >= {A.min_wind:g} kn")
    print(line(rule, "all markets"))
    for mk, x in rule.groupby("market"): print(line(x, LABEL[mk]))
    for wk, x in rule.groupby("week"): print(line(x, f"week {wk}"))
    print("\nunders by forecast band (outdoor) and everything else:")
    for b, x in m[m.outdoor].groupby("band", observed=False): print(line(x, f"{b} kn"))
    print(line(m[~m.outdoor], "domes / closed roofs"))
    print(line(m, "ALL unders (reference)"))
    print("\nqualifying games:")
    gg = rule.groupby("game_id").agg(wind=("wind_fc", "first"), n=("pnl", "size"), hit=("under_won", "mean"), units=("pnl", "sum"))
    print(gg.sort_values("wind", ascending=False).round(2).to_string())
    return m


if __name__ == "__main__":
    main()
