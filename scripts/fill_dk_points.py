"""
Fill the blank dk_points in data/dk_salaries/ (the DailyFantasyFuel seasons, 2022 on) from nflverse.

Players are rescored with dk_scoring.actual_frame from the weekly box score and matched on
season/week/team/normalised name. D/ST is scored here under DraftKings' rules:

    sack +1, interception +2, fumble recovery +2, safety +2, blocked kick +2,
    defensive / return TD +6, two-point / extra-point return +2,
    points allowed  0:+10  1-6:+7  7-13:+4  14-20:+1  21-27:0  28-34:-1  35+:-4

DraftKings' points allowed leaves out points the offense gave up (pick-sixes, fumble-return TDs),
so the opponent's defensive TDs (fumble returns included) come off its final score. Return TDs stay in: special teams are
part of the D/ST. Checked against RotoGuru's own DK points, which cover 2014-21:

    python scripts/fill_dk_points.py --check 2014-2021     # agreement with RotoGuru, writes nothing
    python scripts/fill_dk_points.py                       # fill blanks in every season file
    python scripts/fill_dk_points.py --season 2026         # one season (weekly, after MNF)

Rows that cannot be matched to a box score keep a blank dk_points and are listed.
"""
import argparse, glob, html, os, re, sys, unicodedata
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dk_scoring import actual_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAL = os.path.join(ROOT, "data", "dk_salaries")
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
# DFF / RotoGuru team codes -> nflverse
TEAM = {"LAR": "LA", "ari": "ARI", "atl": "ATL", "bal": "BAL", "buf": "BUF", "car": "CAR", "chi": "CHI",
        "cin": "CIN", "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gnb": "GB", "hou": "HOU",
        "ind": "IND", "jac": "JAX", "kan": "KC", "lac": "LAC", "lar": "LA", "lvr": "LV", "oak": "OAK",
        "mia": "MIA", "min": "MIN", "nor": "NO", "nwe": "NE", "nyg": "NYG", "nyj": "NYJ", "phi": "PHI",
        "pit": "PIT", "sea": "SEA", "sfo": "SF", "tam": "TB", "ten": "TEN", "was": "WAS", "sdg": "SD",
        "stl": "STL"}
DST_POS = {"DST", "Def"}
OFF_POS = {"QB", "RB", "WR", "TE", "FB", "HB"}


def norm(name):
    s = unicodedata.normalize("NFKD", html.unescape(str(name))).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s)
    return re.sub(r"[^a-z]", "", s)


def initial_last(name):
    """'Robby Anderson' / 'R.Anderson' -> 'randerson'."""
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", str(name).lower().replace(".", ". ")).split()
    return norm(s[0][0] + s[-1]) if len(s) >= 2 else norm(name)


def team_code(t, season):
    t = TEAM.get(t, t).upper()
    if t == "OAK" and season >= 2020: t = "LV"
    if t == "LV" and season < 2020: t = "OAK"
    if t == "SD" and season >= 2017: t = "LAC"
    if t == "STL" and season >= 2016: t = "LA"
    return t


def games():
    path = os.path.join(CACHE, "games.csv")
    if not os.path.exists(path) or os.path.getmtime(path) < pd.Timestamp.now().timestamp() - 86400:
        pd.read_csv(GAMES_URL).to_csv(path, index=False)
    return pd.read_csv(path)


def pa_points(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10, 7, 4, 1, 0, -1], -4)


def dst_frame(st, g, season):
    """DK D/ST points per team-week from player box scores + final scores."""
    z = lambda c: st[c].fillna(0).astype(float)
    st = st.assign(_sk=z("def_sacks"), _int=z("def_interceptions"), _fr=z("fumble_recovery_opp"),
                   _sf=z("def_safeties"), _blk=z("def_punt_blocks") + z("def_fg_blocks") + z("def_pat_blocks"),
                   # a defender's fumble-return TD lands in fumble_recovery_tds, not def_tds
                   _dtd=z("def_tds") + np.where(st.position.isin(OFF_POS), 0, z("fumble_recovery_tds")),
                   _sttd=z("special_teams_tds"), _x2=z("def_2pt_made"))
    t = st.groupby(["season", "week", "team"])[["_sk", "_int", "_fr", "_sf", "_blk", "_dtd", "_sttd", "_x2"]].sum().reset_index()
    g = g[(g.season == season) & g.home_score.notna()]
    sides = pd.concat([
        g.assign(team=g.home_team, opp=g.away_team, opp_score=g.away_score),
        g.assign(team=g.away_team, opp=g.home_team, opp_score=g.home_score)])[["season", "week", "team", "opp", "opp_score"]]
    t = sides.merge(t, on=["season", "week", "team"], how="left").fillna(0)
    opp_dtd = t[["season", "week", "team", "_dtd"]].rename(columns={"team": "opp", "_dtd": "_opp_dtd"})
    t = t.merge(opp_dtd, on=["season", "week", "opp"], how="left").fillna({"_opp_dtd": 0})
    pa = (t.opp_score - 6 * t._opp_dtd).clip(lower=0)
    t["pts"] = (t._sk + 2 * (t._int + t._fr + t._sf + t._blk + t._x2) + 6 * (t._dtd + t._sttd) + pa_points(pa))
    return t[["season", "week", "team", "pts"]]


def score_season(season, g):
    path = os.path.join(CACHE, f"stats_player_week_{season}.parquet")
    st = pd.read_parquet(path)
    st = st[st.season_type == "REG"].copy()
    st["pts"] = actual_frame(st)
    off = st[st.position.isin(OFF_POS)]                               # keeps Michael Carter (RB) apart from Carter II (DB)
    players = off.assign(key=off.player_display_name.map(norm))[["season", "week", "team", "key", "pts"]]
    # renamed players (Robby Anderson -> Robbie Chosen, Gabriel -> Gabe Davis): first initial + surname on the team
    alt = off.assign(key=off.player_name.map(initial_last))[["season", "week", "team", "key", "pts"]]
    return players, alt, dst_frame(st, g, season)


def attach(d, season, g):
    players, alt, dst = score_season(season, g)
    d = d.copy()
    d["_team"] = [team_code(t, season) for t in d.team]
    d["_key"] = d.player.map(norm)
    isd = d.pos.isin(DST_POS)
    # players: exact display name on team-week, then the abbreviated name, then name-only within the week (trades)
    dup = lambda f: f.drop_duplicates(["season", "week", "team", "key"], keep=False)
    m = d.merge(dup(players), left_on=["season", "week", "_team", "_key"], right_on=["season", "week", "team", "key"],
                how="left", suffixes=("", "_n"))["pts"].values
    d["_ikey"] = d.player.map(initial_last)
    m2 = d.merge(dup(alt), left_on=["season", "week", "_team", "_ikey"], right_on=["season", "week", "team", "key"],
                 how="left", suffixes=("", "_n"))["pts"].values
    wk = players.drop_duplicates(["season", "week", "key"], keep=False)
    m3 = d.merge(wk, left_on=["season", "week", "_key"], right_on=["season", "week", "key"], how="left",
                 suffixes=("", "_n"))["pts"].values
    calc = np.where(~np.isnan(m), m, np.where(~np.isnan(m2), m2, m3))
    md = d.merge(dst, left_on=["season", "week", "_team"], right_on=["season", "week", "team"], how="left",
                 suffixes=("", "_n"))["pts"].values
    calc = np.where(isd, md, calc)
    # weeks the box-score file does not have yet (nflverse lags a few days) stay blank, D/ST included
    have = d.week.isin(set(players.week))
    calc = np.where(have, calc, np.nan)
    # a skill player on the slate with no box-score row did not play: 0, as DraftKings scores him
    played_week = {k for k in zip(dst.season, dst.week, dst.team) if k[1] in set(players.week)}
    team_played = np.array([(s, w, t) in played_week for s, w, t in zip(d.season, d.week, d._team)])
    missing = np.isnan(calc) & ~isd & team_played
    unmatched = d.loc[missing, ["season", "week", "player", "pos", "team"]]
    return np.round(calc, 2), unmatched


def check(seasons, g):
    for y in seasons:
        d = pd.read_csv(os.path.join(SAL, f"dk_salaries_{y}.csv"))
        calc, _ = attach(d, y, g)
        ok = d.dk_points.notna() & ~np.isnan(calc)
        diff = (calc - d.dk_points)[ok].abs()
        isd = d.pos.isin(DST_POS)[ok]
        for lab, msk in (("players", ~isd), ("D/ST", isd)):
            x = diff[msk]
            print(f"{y} {lab:7s} n={len(x):5d}  exact(<=0.05) {np.mean(x <= 0.05):6.1%}  within 1 {np.mean(x <= 1):6.1%}"
                  f"  MAE {x.mean():.3f}   unmatched {int((d.dk_points.notna() & np.isnan(calc) & (d.pos.isin(DST_POS) == (lab == 'D/ST'))).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--check", help="seasons to compare with RotoGuru's DK points, e.g. 2014-2021")
    A = ap.parse_args()
    g = games()
    if A.check:
        a, b = (A.check.split("-") + [A.check])[:2]
        return check(range(int(a), int(b) + 1), g)
    files = sorted(glob.glob(os.path.join(SAL, "dk_salaries_*.csv")))
    for f in files:
        y = int(re.search(r"(\d{4})", os.path.basename(f)).group(1))
        if A.season and y != A.season: continue
        d = pd.read_csv(f, dtype={"gid": str})
        blank = d.dk_points.isna()
        if not blank.any(): print(f"{y}: no blanks"); continue
        calc, unmatched = attach(d, y, g)
        fill = blank & ~np.isnan(calc)
        d.loc[fill, "dk_points"] = calc[fill]
        # on the slate, team played, no box-score row under any name -> did not record a stat: 0
        zero = blank & ~fill & d.index.isin(unmatched.index)
        d.loc[zero, "dk_points"] = 0.0
        d.to_csv(f, index=False)
        left = int(d.dk_points.isna().sum())
        print(f"{y}: filled {int(fill.sum()):,} from box scores, {int(zero.sum()):,} as 0 (no box-score row), {left} still blank")
        if len(unmatched):
            u = unmatched[unmatched.pos.isin(["QB"])]
            if len(u): print("   QBs set to 0 (check names):", u.player.unique()[:15].tolist())


if __name__ == "__main__":
    main()
