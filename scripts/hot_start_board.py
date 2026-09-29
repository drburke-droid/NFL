"""Who to sell, who to buy, after three weeks of the current season.

Applies the rule measured in scripts/hot_start_study.py: a player's early points split into the part
his VOLUME earns (targets, carries, attempts, which carry into the rest of the season) and the part
he got on top of it (efficiency and touchdowns, which do not). The model is fit on 2016-2025 and
applied to the season in progress, so nothing here is fit on the year it is judging.

It then marks each player against the league's own rosters: on my team (sell candidates), on someone
else's (trade targets), or free (waiver adds).

Usage: python scripts/hot_start_board.py [--season 2026] [--start-week 4] [--team-id 5]
Writes outputs/reports/hot_start_board_<season>.md
"""
import os, re, json, argparse, unicodedata
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nflverse_cache")
LH = r"C:\Users\drbur\Documents\GitHub\league-history"
SKILL = ["QB", "RB", "WR", "TE"]

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--start-week", type=int, default=0, help="first week of the rest-of-season window; 0 = the first week nflverse has not finished")
ap.add_argument("--team-id", type=int, default=5, help="my team id in the league-history rosters")
ap.add_argument("--min-games", type=int, default=2, help="early games needed to be ranked")
A = ap.parse_args()
if not A.start_week:      # match the model to the weeks that are actually in the box scores
    _fp = os.path.join(ROOT, "data", "nflverse_cache", f"stats_player_week_{A.season}.parquet")
    _d = pd.read_parquet(_fp)
    A.start_week = int(_d[_d.season_type == "REG"].week.max()) + 1
EARLY = A.start_week - 1
# ESPN writes "Travis Etienne Jr." and "James Cook III" where nflverse writes the bare name, so a
# plain letters-only key silently makes owned players look free. Suffixes come off, and where both
# sides carry an id we join on that instead (nflverse's roster file maps gsis_id to espn_id).
_SUF = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?\s*$", re.I)
def norm(s):
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    t = _SUF.sub("", t.replace(".", " ").strip())
    return re.sub(r"[^a-z]", "", t.lower())

import importlib.util
spec = importlib.util.spec_from_file_location("hss", os.path.join(ROOT, "scripts", "hot_start_study.py"))


def league_points(d):
    z = lambda c: d[c].fillna(0).astype(float) if c in d.columns else 0.0
    fl = z("fumbles_lost_total") if "fumbles_lost_total" in d.columns else \
        z("sack_fumbles_lost") + z("rushing_fumbles_lost") + z("receiving_fumbles_lost")
    return (0.04 * z("passing_yards") + 6 * z("passing_tds") - z("passing_interceptions")
            + 0.1 * z("rushing_yards") + 6 * z("rushing_tds")
            + z("receptions") + 0.1 * z("receiving_yards") + 6 * z("receiving_tds")
            + 2 * (z("passing_2pt_conversions") + z("rushing_2pt_conversions") + z("receiving_2pt_conversions")) - fl)


def season_frame(s, upto):
    fp = os.path.join(CACHE, f"stats_player_week_{s}.parquet")
    if not os.path.exists(fp): return None
    d = pd.read_parquet(fp)
    d = d[(d.season_type == "REG") & d.position.isin(SKILL) & (d.week <= upto)].copy()
    d["pts"] = league_points(d)
    return d


def early_block(d, upto):
    e = d[d.week <= upto]
    g = e.groupby(["player_id"]).agg(
        player=("player_display_name", "last"), pos=("position", "last"), team=("team", "last"),
        g=("week", "count"), pts=("pts", "sum"), tgt=("targets", "sum"), car=("carries", "sum"),
        att=("attempts", "sum"), rec_td=("receiving_tds", "sum"), rush_td=("rushing_tds", "sum"),
        pass_td=("passing_tds", "sum"), tgt_share=("target_share", "mean")).reset_index()
    g["early_ppg"] = g.pts / g.g
    g["tgt_g"], g["car_g"], g["att_g"] = g.tgt / g.g, g.car / g.g, g.att / g.g
    g["touch_g"] = (g.tgt + g.car) / g.g
    g["td_g"] = (g.rec_td + g.rush_td + g.pass_td) / g.g
    g["tgt_share"] = g.tgt_share.fillna(0)
    return g


def main():
    # ---- history: every completed season, weeks 1..EARLY predicting the rest
    hist = []
    for s in range(2016, A.season):
        d = season_frame(s, 17)
        if d is None: continue
        e = early_block(d, EARLY)
        r = d[d.week >= A.start_week].groupby("player_id").agg(ros_g=("week", "count"), ros_pts=("pts", "sum")).reset_index()
        m = e.merge(r, on="player_id", how="inner")
        m = m[(m.g >= A.min_games) & (m.ros_g >= 6)]
        m["ros_ppg"] = m.ros_pts / m.ros_g
        m["season"] = s
        hist.append(m)
    H = pd.concat(hist, ignore_index=True)
    # last season's ppg, the prior the market had
    prev_rows = []
    for s in range(2016, A.season + 1):
        d = season_frame(s, 17)
        if d is None: continue
        p = d.groupby("player_id").agg(pg=("week", "count"), pp=("pts", "sum")).reset_index()
        p["prev_ppg"] = p.pp / p.pg; p["season"] = s + 1
        prev_rows.append(p[["player_id", "season", "prev_ppg", "pg"]])
    P = pd.concat(prev_rows, ignore_index=True).rename(columns={"pg": "prev_g"})
    H = H.merge(P, on=["player_id", "season"], how="left")
    H["prev_ppg"] = H.prev_ppg.where(H.prev_g >= 6).fillna(0.0)
    H["has_prev"] = (H.prev_g.fillna(0) >= 6).astype(int)
    H["gap"] = H.ros_ppg - H.early_ppg
    print(f"history: {len(H):,} player-seasons {H.season.min()}-{H.season.max()}")

    # ---- the volume rates, per position, from history only
    rates = {}
    for pos, grp in H.groupby("pos"):
        X = np.column_stack([grp[["tgt_g", "car_g", "att_g"]].fillna(0).values, np.ones(len(grp))])
        beta, *_ = np.linalg.lstsq(X, grp.early_ppg.values, rcond=None)
        rates[pos] = beta
    def luck(frame):
        out = np.full(len(frame), np.nan)
        for pos, beta in rates.items():
            m = (frame.pos == pos).values
            if not m.any(): continue
            X = np.column_stack([frame.loc[m, ["tgt_g", "car_g", "att_g"]].fillna(0).values, np.ones(m.sum())])
            out[m] = frame.loc[m, "early_ppg"].values - X @ beta
        return out
    H["luck"] = luck(H)

    # ---- the model, fit on history only
    FEATS = ["early_ppg", "luck", "tgt_share", "touch_g", "prev_ppg", "has_prev"]
    X = np.column_stack([H[FEATS].fillna(0).values, np.ones(len(H))])
    ok = np.isfinite(X).all(1) & np.isfinite(H.gap.values)
    beta, *_ = np.linalg.lstsq(X[ok], H.gap.values[ok], rcond=None)
    print("model fit on", int(ok.sum()), "rows;", ", ".join(f"{f} {b:+.3f}" for f, b in zip(FEATS, beta)))

    # ---- this season
    cur = season_frame(A.season, EARLY)
    if cur is None: raise SystemExit(f"no weekly file for {A.season}")
    C = early_block(cur, EARLY)
    C = C[C.g >= A.min_games].copy()
    C = C.merge(P[P.season == A.season][["player_id", "prev_ppg", "prev_g"]], on="player_id", how="left")
    C["prev_ppg"] = C.prev_ppg.where(C.prev_g >= 6).fillna(0.0)
    C["has_prev"] = (C.prev_g.fillna(0) >= 6).astype(int)
    C["luck"] = luck(C)
    Xc = np.column_stack([C[FEATS].fillna(0).values, np.ones(len(C))])
    C["pred_gap"] = Xc @ beta
    C["pred_ros"] = C.early_ppg + C.pred_gap
    print(f"{A.season}: {len(C)} players with {A.min_games}+ games through week {EARLY}")

    # ---- who owns them in my league
    C["n"] = C.player.map(norm)
    tag = pd.Series("free agent", index=C.index)
    owner = pd.Series("", index=C.index)
    # the live league: outputs/espn_league.json is this season's pull (My Team tab / gm build use it).
    # league-history's rosters.json only runs to the last completed season, so it is the fallback.
    live = os.path.join(ROOT, "outputs", "espn_league.json")
    inj = pd.Series("", index=C.index)
    if os.path.exists(live):
        L = json.load(open(live, encoding="utf-8"))
        me_id = int(L.get("myTeamId", A.team_id))
        # gsis_id <-> espn_id, so ownership joins on an id rather than a spelling
        e2g = {}
        rp = os.path.join(CACHE, f"roster_{A.season}.csv")
        if os.path.exists(rp):
            rr = pd.read_csv(rp, dtype=str)
            e2g = {str(x.espn_id).split(".")[0]: x.gsis_id for x in rr.itertuples()
                   if isinstance(getattr(x, "espn_id", None), str) and isinstance(getattr(x, "gsis_id", None), str)}
        by_id, by_name = {}, {}
        for t in L.get("teams", []):
            lab = "MY TEAM" if int(t.get("id", -1)) == me_id else "another team"
            team = "me" if lab == "MY TEAM" else t.get("name", "")
            for p in t.get("roster", []):
                v = (lab, team, p.get("inj") or "")
                gid = e2g.get(str(p.get("espn_id", "")).split(".")[0])
                if gid: by_id[gid] = v
                by_name[norm(p.get("name"))] = v
        look = lambda pid, nm: by_id.get(pid) or by_name.get(nm) or ("free agent", "", "")
        got = [look(pid, nm) for pid, nm in zip(C.player_id, C.n)]
        tag = pd.Series([x[0] for x in got], index=C.index)
        owner = pd.Series([x[1] for x in got], index=C.index)
        inj = pd.Series([x[2] for x in got], index=C.index)
        matched_by_id = sum(1 for pid in C.player_id if pid in by_id)
        print(f"live league ({L.get('name','?')}, season {L.get('season')}): {len(by_name)} rostered, "
              f"{sum(1 for v in by_name.values() if v[0]=='MY TEAM')} on my team; "
              f"{matched_by_id} of {len(C)} ranked players joined by id")
    else:
        print("no outputs/espn_league.json; everyone shows as free agent")
    C["own"] = tag        # not "where": that shadows DataFrame.where and breaks attribute access
    C["owner"] = owner
    C["inj"] = inj
    C["out"] = C.inj.isin(["INJURY_RESERVE", "OUT"])

    # ---- the board
    # a player on IR or ruled out is not a buy at any price, however good his usage looked
    real = C[(C.early_ppg >= 4) & (~C.out)].copy()
    if int(C.out.sum()):
        gone = C[C.out].sort_values("pred_gap", ascending=False).head(8)
        say = ", ".join(f"{r.player} ({str(r.inj).title().replace('_', ' ')})" for r in gone.itertuples())
        print("\nexcluded (IR / out): " + say)
    sells = real.sort_values("pred_gap").head(25)
    buys = real.sort_values("pred_gap", ascending=False).head(25)
    cols = ["player", "pos", "team", "own", "owner", "inj", "g", "early_ppg", "touch_g", "tgt_share", "td_g", "luck", "pred_ros", "pred_gap"]
    fmt = lambda d: d[cols].round(2).to_string(index=False)
    lines = [f"{A.season} weeks 1-{EARLY} -> rest of season, league scoring. "
             f"'luck' = points per game above what this player's volume alone earns. "
             f"pred_ros = what the model expects him to average from week {A.start_week} on.", "",
             "SELL HIGH: the model says these come down the most", fmt(sells), "",
             "BUY LOW: the model says these go up the most", fmt(buys)]
    print("\n" + "\n".join(lines[2:4])); print("\n" + "\n".join(lines[5:7]))
    for w in ("MY TEAM", "another team", "free agent"):
        sub_s = sells[sells.own == w]; sub_b = buys[buys.own == w]
        if len(sub_s): lines += ["", f"SELL candidates on {w}:", fmt(sub_s)]
        if len(sub_b): lines += ["", f"BUY candidates on {w}:", fmt(sub_b)]
    os.makedirs(os.path.join(ROOT, "outputs", "reports"), exist_ok=True)
    fp = os.path.join(ROOT, "outputs", "reports", f"hot_start_board_{A.season}.md")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write(f"# Sell high / buy low after week {EARLY}, {A.season}\n\n```\n" + "\n".join(lines) + "\n```\n")
    C.to_csv(os.path.join(ROOT, "outputs", "reports", f"hot_start_board_{A.season}.csv"), index=False)
    print(f"\nwrote {os.path.relpath(fp, ROOT)}")


if __name__ == "__main__":
    main()
