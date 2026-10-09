"""
Pattern hunt on DraftKings player props: is there ANY pregame rule -- obvious or esoteric -- that beats DK's
closing lines, and does it survive out of sample?

Frame (outputs/props_mine_frame.parquet, rebuilt with --build):
  DK main-line props 2023-25 (db/nfl_odds.db, last two-sided DK snapshot before kickoff, alternates excluded)
  graded against nflverse box scores (no box-score row = player did not play = void, dropped; pushes dropped).
Features -- all knowable before kickoff:
  last_ppg_z     last game's PPR points vs his season-to-date mean before it, in his own SDs (>= 3 prior games)
  last_stat_z    the same for the prop's own stat
  last_res       last game's result on the same market vs DK's line that week (+1 over / -1 under), streak_over
  line_vs_avg    (line - season-to-date mean of the stat) / season SD;  line_vs_l3 vs last-3 mean
  gd_z / gd_tone GDELT articles about him in the last COMPLETED Tue-Mon week before kickoff, z vs his own
                 trailing weeks; tone of that coverage (the week containing the game is excluded: it leaks)
  game           closing total and spread (nflverse), home, divisional, primetime (kick >= 19:00 ET or
                 Thu/Mon/Sat), short week, wind (mph), dome, week of season
  price          over juice (DK over price), line rank within market x week (star lines)
  injury         Wednesday-Friday practice status and game status from the injury report (nflv_injuries)
  history        player's own season-to-date over rate on this market (>= 3 graded props), rookie

Mining protocol (--mine): every rule is a binary cell x side (over/under), evaluated at DK's actual price.
  DISCOVER on 2023-24, VALIDATE on 2025, then 2026 wk1-4 snapshots (--check-2026) for survivors only.
  Survivor = discovery ROI > 0 with one-sided p < 0.05 AND validation ROI > 0 with p < 0.10. The number of
  rules tried is printed: with ~150 tries, ~7 pass discovery and ~0.7 pass both by luck alone.

    python scripts/props_pattern_mine.py --build
    python scripts/props_pattern_mine.py --mine
"""
import argparse, os, re, sqlite3, sys
import numpy as np, pandas as pd
from scipy.stats import norm as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
FRAME = os.path.join(ROOT, "outputs", "props_mine_frame.parquet")
OUTMD = os.path.join(ROOT, "outputs", "reports", "props_pattern_mine.md")

MARKETS = {"player_pass_yds": ["passing_yards"], "player_pass_tds": ["passing_tds"],
           "player_pass_attempts": ["attempts"], "player_pass_completions": ["completions"],
           "player_pass_interceptions": ["passing_interceptions"], "player_rush_yds": ["rushing_yards"],
           "player_rush_attempts": ["carries"], "player_receptions": ["receptions"],
           "player_reception_yds": ["receiving_yards"],
           "player_rush_reception_yds": ["rushing_yards", "receiving_yards"],
           "player_pass_rush_yds": ["passing_yards", "rushing_yards"]}
GROUP = {"player_pass_yds": "qb", "player_pass_tds": "qb", "player_pass_attempts": "qb",
         "player_pass_completions": "qb", "player_pass_interceptions": "qb", "player_pass_rush_yds": "qb",
         "player_rush_yds": "rush", "player_rush_attempts": "rush", "player_receptions": "rec",
         "player_reception_yds": "rec", "player_rush_reception_yds": "rec"}
TEAM = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
        "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


def norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("'", ""))
    return re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)).strip()


def payout(price):
    """Profit per 1 unit staked at an American price."""
    p = np.asarray(price, float)
    return np.where(p > 0, p / 100, 100 / -p)


# ------------------------------------------------------------------ build
def build():
    con = sqlite3.connect(DB)
    print("reading DK props ...", flush=True)
    pp = pd.read_sql(f"""SELECT p.event_id, p.market, p.player_name, p.outcome_type, p.price, p.point, p.snapshot_time,
                                g.commence_time, g.home_team, g.away_team, g.season, g.week
                         FROM player_props p JOIN games g USING(event_id)
                         WHERE p.bookmaker='draftkings' AND p.market IN ({','.join("'" + m + "'" for m in MARKETS)})
                           AND p.snapshot_time <= g.commence_time""", con)
    pp = pp.sort_values("snapshot_time").groupby(["event_id", "market", "player_name", "outcome_type"]).tail(1)
    o = pp[pp.outcome_type == "Over"]; u = pp[pp.outcome_type == "Under"]
    k = ["event_id", "market", "player_name"]
    d = o.merge(u[k + ["price", "point"]], on=k, suffixes=("_o", "_u"))
    d = d[d.point_o == d.point_u].rename(columns={"point_o": "line"}).drop(columns=["point_u", "outcome_type"])
    d = d[(d.price_o.abs() >= 100) & (d.price_u.abs() >= 100)]
    d["week"] = pd.to_numeric(d.week, errors="coerce")
    d = d[d.week.notna()]; d["week"] = d.week.astype(int)
    d["home"], d["away"] = d.home_team.map(TEAM), d.away_team.map(TEAM)
    d["nm"] = d.player_name.map(norm)
    print(f"  {len(d):,} two-sided DK props", flush=True)

    # box scores
    st = pd.concat([pd.read_parquet(os.path.join(CACHE, f"stats_player_week_{y}.parquet")) for y in (2023, 2024, 2025)])
    st["nm"] = st.player_display_name.map(norm)
    st["ppr"] = st.fantasy_points_ppr
    st = st.sort_values(["season", "week"])
    # pregame history features per player-week (season to date, before this week)
    g = st.groupby(["player_id", "season"])
    feats = {}
    cols = sorted({c for v in MARKETS.values() for c in v} | {"ppr"})
    for c in cols:
        st[c] = st[c].fillna(0).astype(float)
        sh = g[c].shift(1)
        st[f"{c}_n"] = g[c].cumcount()
        st[f"{c}_mean_prev"] = sh.groupby([st.player_id, st.season]).transform(lambda x: x.expanding().mean())
        st[f"{c}_sd_prev"] = sh.groupby([st.player_id, st.season]).transform(lambda x: x.expanding().std())
        st[f"{c}_last"] = sh
        st[f"{c}_l3"] = sh.groupby([st.player_id, st.season]).transform(lambda x: x.rolling(3, min_periods=1).mean())
        # the LAST game vs the mean BEFORE that game: z of last game
        sh2 = g[c].shift(2)
        st[f"{c}_mean_prev2"] = sh2.groupby([st.player_id, st.season]).transform(lambda x: x.expanding().mean())
        st[f"{c}_sd_prev2"] = sh2.groupby([st.player_id, st.season]).transform(lambda x: x.expanding().std())
        st[f"{c}_n2"] = sh2.groupby([st.player_id, st.season]).transform(lambda x: x.expanding().count())
    keep = ["player_id", "nm", "season", "week", "team", "opponent_team", "position", "game_id"] + \
           [c for c in st.columns if any(c.startswith(x + "_") for x in cols)] + cols
    st = st[keep]
    m = d.merge(st, on=["season", "week", "nm"], how="inner")
    m = m[(m.team == m.home) | (m.team == m.away)]
    m = m.drop_duplicates(["event_id", "market", "player_name"], keep=False)
    print(f"  {len(m):,} matched to a box score (rest = DNP / void / name miss)", flush=True)
    # actual stat + results
    m["actual"] = sum(m[c] for c in MARKETS[next(iter(MARKETS))]) * 0
    for mk, cs in MARKETS.items():
        sel = m.market == mk
        m.loc[sel, "actual"] = sum(m.loc[sel, c] for c in cs)
        for suf in ("mean_prev", "sd_prev", "last", "l3", "mean_prev2", "sd_prev2", "n2", "n"):
            if len(cs) == 1:
                m.loc[sel, "s_" + suf] = m.loc[sel, f"{cs[0]}_{suf}"]
            else:
                # combined markets: means add; sds approximated by the root-sum-square
                if "sd" in suf:
                    m.loc[sel, "s_" + suf] = np.sqrt(sum(m.loc[sel, f"{c}_{suf}"] ** 2 for c in cs))
                elif suf in ("n2", "n"):
                    m.loc[sel, "s_" + suf] = m.loc[sel, f"{cs[0]}_{suf}"]
                else:
                    m.loc[sel, "s_" + suf] = sum(m.loc[sel, f"{c}_{suf}"] for c in cs)
    m = m[m.actual != m.line]                                                   # pushes
    m["over"] = (m.actual > m.line).astype(int)
    m["pay_o"], m["pay_u"] = payout(m.price_o), payout(m.price_u)
    m["group"] = m.market.map(GROUP)
    # ---- features
    ok2 = m.s_n2 >= 3
    m["last_stat_z"] = np.where(ok2, (m.s_last - m.s_mean_prev2) / m.s_sd_prev2.replace(0, np.nan), np.nan)
    okp = m.ppr_n2 >= 3
    m["last_ppg_z"] = np.where(okp, (m.ppr_last - m.ppr_mean_prev2) / m.ppr_sd_prev2.replace(0, np.nan), np.nan)
    ok = m.s_n >= 3
    m["line_vs_avg"] = np.where(ok, (m.line - m.s_mean_prev) / m.s_sd_prev.replace(0, np.nan), np.nan)
    m["line_vs_l3"] = np.where(ok, (m.line - m.s_l3) / m.s_sd_prev.replace(0, np.nan), np.nan)
    m["line_rank"] = m.groupby(["season", "week", "market"]).line.rank(ascending=False, pct=True)
    # prior result on the same market (needs the previous week's DK line for him)
    m = m.sort_values(["player_id", "market", "season", "week"])
    gg = m.groupby(["player_id", "market", "season"])
    m["last_res"] = gg.over.shift(1).map({1: 1, 0: -1})
    m.loc[m.groupby(["player_id", "market", "season"]).week.diff() > 2, "last_res"] = np.nan   # skipped weeks: not "last"
    m["prior_props"] = gg.cumcount()
    m["over_rate_prev"] = gg.over.transform(lambda x: x.shift(1).expanding().mean())
    def streak(x):
        out, s = [], 0
        for v in x:
            out.append(s); s = s + 1 if v == 1 else 0
        return pd.Series(out, index=x.index)
    m["streak_over"] = gg.over.transform(streak)
    # game context from nflverse schedule
    gm = pd.read_csv(os.path.join(CACHE, "games.csv"))
    gm = gm[gm.season.between(2023, 2025)][["game_id", "gameday", "weekday", "gametime", "spread_line", "total_line",
                                             "div_game", "roof", "wind", "temp", "home_team", "away_team",
                                             "home_rest", "away_rest"]]
    m = m.drop(columns=["home_team", "away_team"]).merge(gm, on="game_id", how="left")
    m["is_home"] = (m.team == m.home_team).astype(int)
    m["team_spread"] = np.where(m.is_home == 1, -m.spread_line, m.spread_line)    # negative = favourite
    m["rest"] = np.where(m.is_home == 1, m.home_rest, m.away_rest)
    m["prime"] = ((m.gametime >= "19:00") | m.weekday.isin(["Thursday", "Monday", "Saturday"])).astype(int)
    m["dome"] = m.roof.isin(["dome", "closed"]).astype(int)
    # GDELT: last completed Tue-Mon week (start 8-14 days before a Sunday kickoff; >= 7 days generally)
    gd = pd.read_sql("SELECT name, wk, articles, avg_tone FROM nflv_gdelt_wk", con)
    gd["nm"] = gd.name.map(norm); gd["wkd"] = pd.to_datetime(gd.wk)
    gd = gd.sort_values(["nm", "wkd"])
    gd["art_mu"] = gd.groupby("nm").articles.transform(lambda x: x.shift(1).rolling(8, min_periods=3).mean())
    gd["art_sd"] = gd.groupby("nm").articles.transform(lambda x: x.shift(1).rolling(8, min_periods=3).std())
    m["kick"] = pd.to_datetime(m.commence_time).dt.tz_localize(None)
    m["kick_day"] = m.kick.dt.normalize()
    # the Tuesday that starts the week containing kickoff, minus 7 days = last completed week
    m["cur_tue"] = m.kick_day - pd.to_timedelta((m.kick_day.dt.weekday - 1) % 7, unit="D")
    m["prev_tue"] = m.cur_tue - pd.Timedelta(days=7)
    m = m.merge(gd[["nm", "wkd", "articles", "avg_tone", "art_mu", "art_sd"]].rename(columns={"wkd": "prev_tue"}),
                on=["nm", "prev_tue"], how="left")
    m["gd_articles"] = m.articles.fillna(0)
    m["gd_z"] = (m.articles - m.art_mu) / m.art_sd.replace(0, np.nan)
    m["gd_tone"] = m.avg_tone
    # injury report for that week
    inj = pd.read_sql("SELECT season, week, gsis_id player_id, report_status, practice_status FROM nflv_injuries "
                      "WHERE season >= 2023 AND season_type='REG'", con)
    inj["season"] = inj.season.astype(int); inj["week"] = inj.week.astype(int)
    inj = inj.drop_duplicates(["season", "week", "player_id"], keep="last")
    m = m.merge(inj, on=["season", "week", "player_id"], how="left")
    ros = pd.concat([pd.read_parquet(os.path.join(CACHE, f"roster_{y}.parquet"))[["season", "gsis_id", "rookie_year"]]
                     for y in (2023, 2024, 2025)]).rename(columns={"gsis_id": "player_id"}).drop_duplicates(["season", "player_id"])
    m = m.merge(ros, on=["season", "player_id"], how="left")
    m["rookie"] = (m.rookie_year == m.season).astype(int)
    drop = [c for c in m.columns if re.match(r"^(passing|rushing|receiving|attempts|completions|carries|receptions|ppr)_", c)
            and c not in ("ppr_last",)]
    m = m.drop(columns=drop + ["articles", "avg_tone"])
    os.makedirs(os.path.dirname(FRAME), exist_ok=True)
    m.to_parquet(FRAME, index=False)
    print(f"-> {os.path.relpath(FRAME, ROOT)}: {len(m):,} graded props; overs hit {m.over.mean():.3f}")
    print(m.groupby("season").agg(n=("over", "size"), over=("over", "mean")).to_string())


# ------------------------------------------------------------------ mine
def cells(m):
    """Binary rule cells: name -> boolean mask (over the whole frame)."""
    C = {}
    def add(name, mask): C[name] = mask.fillna(False).astype(bool)
    add("ALL", pd.Series(True, index=m.index))
    for t in (1.0, 1.5, 2.0):
        add(f"last_ppg_z>{t}", m.last_ppg_z > t); add(f"last_stat_z>{t}", m.last_stat_z > t)
    for t in (-1.0, -1.5):
        add(f"last_ppg_z<{t}", m.last_ppg_z < t); add(f"last_stat_z<{t}", m.last_stat_z < t)
    add("last_res=over", m.last_res == 1); add("last_res=under", m.last_res == -1)
    add("streak_over>=2", m.streak_over >= 2); add("streak_over>=3", m.streak_over >= 3)
    add("over_rate_prev>=0.7 (n>=4)", (m.over_rate_prev >= 0.7) & (m.prior_props >= 4))
    add("over_rate_prev<=0.3 (n>=4)", (m.over_rate_prev <= 0.3) & (m.prior_props >= 4))
    for t in (0.5, 1.0):
        add(f"line_vs_avg>{t}", m.line_vs_avg > t); add(f"line_vs_avg<-{t}", m.line_vs_avg < -t)
        add(f"line_vs_l3>{t}", m.line_vs_l3 > t); add(f"line_vs_l3<-{t}", m.line_vs_l3 < -t)
    add("gd_z>2", m.gd_z > 2); add("gd_z>1", m.gd_z > 1); add("gd_z<-1", m.gd_z < -1)
    add("gd_articles top10%", m.gd_articles >= m.gd_articles.quantile(0.9))
    add("gd_tone<-2", m.gd_tone < -2); add("gd_tone>0", m.gd_tone > 0)
    add("gd_z>1 & tone<-2", (m.gd_z > 1) & (m.gd_tone < -2))
    add("total>=48", m.total_line >= 48); add("total<=40", m.total_line <= 40)
    add("big fav (<=-7)", m.team_spread <= -7); add("big dog (>=7)", m.team_spread >= 7)
    add("home", m.is_home == 1); add("away", m.is_home == 0)
    add("divisional", m.div_game == 1); add("primetime", m.prime == 1)
    add("short rest (<=5)", m.rest <= 5); add("long rest (>=10)", m.rest >= 10)
    add("wind>=15", m.wind >= 15); add("dome", m.dome == 1); add("cold (<=35F)", m.temp <= 35)
    add("weeks 1-3", m.week <= 3); add("weeks 15+", m.week >= 15); add("playoffs (wk>=19)", m.week >= 19)
    add("over juiced (<=-130)", m.price_o <= -130); add("under juiced (<=-130)", m.price_u <= -130)
    add("plus-money over", m.price_o > 0)
    add("star line (top 10%)", m.line_rank <= 0.10); add("low line (bottom 25%)", m.line_rank >= 0.75)
    add("Questionable", m.report_status == "Questionable")
    add("limited practice", m.practice_status.fillna("").str.contains("Limited"))
    add("DNP practice, playing", m.practice_status.fillna("").str.contains("Did Not"))
    add("rookie", m.rookie == 1)
    add("half-point line .5 only", (m.line % 1) == 0.5)
    return C


def evaluate(m, mask, side):
    x = m[mask]
    if len(x) == 0: return dict(n=0)
    win = x.over.values == (1 if side == "over" else 0)
    pay = (x.pay_o if side == "over" else x.pay_u).values
    pnl = np.where(win, pay, -1.0)
    roi = pnl.mean(); se = pnl.std(ddof=1) / np.sqrt(len(pnl)) if len(pnl) > 1 else np.nan
    return dict(n=len(x), hit=win.mean(), roi=roi, z=roi / se if se else np.nan,
                p=1 - N.cdf(roi / se) if se else np.nan)


def mine():
    m = pd.read_parquet(FRAME)
    C = cells(m)
    disc, val = m.season.isin([2023, 2024]), m.season == 2025
    rows = []
    for scope in ["all", "qb", "rush", "rec"]:
        sc = pd.Series(True, index=m.index) if scope == "all" else (m.group == scope)
        for name, mask in C.items():
            for side in ("over", "under"):
                a = evaluate(m, mask & sc & disc, side); b = evaluate(m, mask & sc & val, side)
                if a["n"] < 150 or b["n"] < 60: continue
                rows.append(dict(scope=scope, rule=name, side=side, d_n=a["n"], d_hit=a["hit"], d_roi=a["roi"],
                                 d_p=a["p"], v_n=b["n"], v_hit=b["hit"], v_roi=b["roi"], v_p=b["p"]))
    r = pd.DataFrame(rows)
    n_tests = len(r)
    surv = r[(r.d_roi > 0) & (r.d_p < 0.05) & (r.v_roi > 0) & (r.v_p < 0.10)].sort_values("v_p")
    disc_pass = r[(r.d_roi > 0) & (r.d_p < 0.05)]
    pd.set_option("display.width", 220)
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P(f"# DK props pattern hunt\n\nframe: {len(m):,} graded DK closing props 2023-25; overs hit {m.over.mean():.3f}")
    P("```")
    base = r[r.rule == "ALL"][["scope", "side", "d_n", "d_hit", "d_roi", "v_n", "v_hit", "v_roi"]]
    P("baseline (every prop, one side):"); P(base.round(3).to_string(index=False))
    P(f"\nrules tested: {n_tests}  (cell x side x scope, n >= 150 discovery / 60 validation)")
    P(f"passed discovery (2023-24, ROI > 0, p < .05): {len(disc_pass)}   expected by luck: {0.05 * n_tests:.1f}")
    P(f"survived validation too (2025, ROI > 0, p < .10): {len(surv)}   expected by luck: {0.05 * n_tests * 0.10:.1f}\n")
    P("discovery winners and what 2025 did with them:")
    P(disc_pass.sort_values("d_p")[["scope", "rule", "side", "d_n", "d_hit", "d_roi", "d_p", "v_n", "v_hit", "v_roi", "v_p"]]
      .round(3).to_string(index=False))
    P("\nSURVIVORS:"); P(surv.round(3).to_string(index=False) if len(surv) else "  none")
    P("```")
    r.to_csv(os.path.join(ROOT, "outputs", "props_mine_rules.csv"), index=False)
    open(OUTMD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"-> {os.path.relpath(OUTMD, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); ap.add_argument("--mine", action="store_true")
    A = ap.parse_args()
    if A.build: build()
    if A.mine: mine()
