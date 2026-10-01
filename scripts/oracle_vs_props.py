"""THE ORACLE (and the Oracle + fan picks) against DraftKings player props, one week, after the fact.

For every DraftKings player prop captured before kickoff (data/props_frames/snapshots), compare:
    line    DK's number (the market's median)
    oracle  THE ORACLE's projection from the last Fan Picks bake made before that game kicked off
    crowd   the Oracle times the average fan adjustment (adjusted / baseline) of the fans who touched
            that player-stat, using each fan's last submission before kickoff
against the box score. Two questions:
    accuracy  is the projection closer to what happened than the line was?
    betting   betting the side the projection points to, at DK's own pre-kick price, what is the ROI?
Anytime TD is scored separately: P(TD) = 1 - exp(-(rush_tds + rec_tds)) against DK's Yes price.

A player with no box-score row is treated as not having played and the prop as void (DK voids it).
The projections are means and the line is a median; for skewed stats (receiving yards) a mean sits
above the median, so "bet the side the mean points to" leans over -- the report prints that lean.

Usage: python scripts/oracle_vs_props.py [--season 2026] [--week 3]
Writes outputs/reports/oracle_vs_props_<season>_wk<week>.md (+ .csv of every graded prop)
"""
import argparse, glob, json, os, re, unicodedata
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKETS = {"player_reception_yds": "rec_yds", "player_receptions": "rec", "player_rush_yds": "rush_yds",
           "player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds"}
BOX = {"rec_yds": "receiving_yards", "rec": "receptions", "rush_yds": "rushing_yards",
       "pass_yds": "passing_yards", "pass_tds": "passing_tds"}
TEAM_FIX = {"LA": "LAR", "LAR": "LA"}


def norm(s):
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", t)
    return re.sub(r"[^a-z]", "", t)


def payout(price):
    """Profit per 1 unit staked at American odds."""
    return price / 100 if price > 0 else 100 / -price


def implied(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def kickoffs(season, week):
    s = pd.read_csv(os.path.join(ROOT, "data", f"schedule_{season}.csv"))
    s = s[(s.game_type == "REG") & (s.week == week)]
    t = pd.to_datetime(s.gameday + " " + s.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    out = {}
    for (a, h), k in zip(zip(s.away_team, s.home_team), t):
        out[a] = out[h] = k
    return out


def oracle_pre_kick(season, week, kick):
    """{player_id: (bake_id, row)} from the last bake before the player's game kicked off."""
    bakes = []
    for f in glob.glob(os.path.join(ROOT, "docs", "fan", f"proj_{season}_wk{week}_*.json")):
        j = json.load(open(f, encoding="utf-8"))
        bakes.append((pd.Timestamp(j["baked"]).tz_convert("UTC") if pd.Timestamp(j["baked"]).tzinfo
                      else pd.Timestamp(j["baked"], tz="UTC"), j))
    bakes.sort(key=lambda x: x[0])
    best = {}
    for t, j in bakes:
        for p in j["players"]:
            k = kick.get(p["team"]) or kick.get(TEAM_FIX.get(p["team"], ""))
            if k is not None and t < k:
                best[p["id"]] = (j["bake_id"], p, k)       # later bakes overwrite earlier ones
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    a = ap.parse_args()
    S, W = a.season, a.week
    kick = kickoffs(S, W)
    orc = oracle_pre_kick(S, W, kick)
    by_name = {norm(p["name"]): pid for pid, (_, p, _) in orc.items()}

    # box scores
    box = pd.read_parquet(os.path.join(ROOT, "data", "nflverse_cache", f"stats_player_week_{S}.parquet"))
    box = box[(box.week == W) & (box.season_type == "REG")].set_index("player_id")

    # fans: last submission per fan before kickoff, as a ratio to the baseline that fan saw
    fan = pd.read_csv(os.path.join(ROOT, "data", "fan_adjustments", "fan_adjustments_long.csv"))
    fan = fan[(fan.season == S) & (fan.week == W) & (fan.baseline > 0)].copy()
    fan["sub"] = pd.to_datetime(fan.submitted_at, utc=True)
    fan["kick"] = fan.player_id.map(lambda g: orc[g][2] if g in orc else pd.NaT)
    fan = fan[fan["sub"] < fan["kick"]]
    fan = fan.sort_values("sub").drop_duplicates(["fan", "player_id", "stat"], keep="last")
    fan["ratio"] = fan.adjusted / fan.baseline
    fan_ratio = fan.groupby(["player_id", "stat"]).ratio.mean()
    fan_n = fan.groupby(["player_id", "stat"]).fan.nunique()
    per_fan = fan.set_index(["fan", "player_id", "stat"]).ratio

    # DK: the last snapshot before kickoff of each player-market-side
    snap = pd.read_csv(os.path.join(ROOT, "data", "props_frames", "snapshots", f"props_{S}.csv"), low_memory=False)
    snap = snap[(snap.week == W) & snap.market.isin(list(MARKETS) + ["player_anytime_td"])].copy()
    snap["t"] = pd.to_datetime(snap.snapshot_utc, utc=True)
    snap["pid"] = snap.player.map(lambda n: by_name.get(norm(n)))
    snap = snap[snap.pid.notna()]
    snap["kick"] = snap.pid.map(lambda g: orc[g][2])
    snap = snap[snap.t < snap.kick].sort_values("t")

    rows = []
    ou = snap[snap.market.isin(MARKETS)]
    last = ou.drop_duplicates(["pid", "market", "side"], keep="last")
    for (pid, mk), g in last.groupby(["pid", "market"]):
        sides = {r.side: r for r in g.itertuples()}
        if "Over" not in sides or "Under" not in sides or sides["Over"].point != sides["Under"].point:
            continue
        stat = MARKETS[mk]
        _, p, _ = orc[pid]
        proj = float(p["stats"].get(stat, 0.0))
        if pid not in box.index:
            continue                                         # did not play: void
        actual = float(box.loc[pid, BOX[stat]] or 0)
        line = float(sides["Over"].point)
        r = {"player": p["name"], "team": p["team"], "pos": p["pos"], "market": stat, "line": line,
             "over_price": sides["Over"].price, "under_price": sides["Under"].price,
             "oracle": proj, "actual": actual,
             "crowd": proj * float(fan_ratio.get((pid, stat), 1.0)), "fans": int(fan_n.get((pid, stat), 0)), "pid": pid}
        rows.append(r)
    D = pd.DataFrame(rows)
    if D.empty:
        raise SystemExit("no graded props")
    D = D[D.actual != D.line]                                # pushes

    def bets(frame, col, min_edge=0.0):
        e = (frame[col] - frame.line) / frame.line.clip(lower=0.5)
        f = frame[e.abs() > min_edge]
        over = f[col] > f.line
        won = np.where(over, f.actual > f.line, f.actual < f.line)
        price = np.where(over, f.over_price, f.under_price)
        prof = np.where(won, [payout(x) for x in price], -1.0)
        return {"n": len(f), "overs": int(over.sum()), "won": int(won.sum()),
                "hit": won.mean() if len(f) else np.nan, "roi": prof.mean() if len(f) else np.nan}

    def acc(frame, col):
        return {"mae": (frame[col] - frame.actual).abs().mean(), "line_mae": (frame.line - frame.actual).abs().mean(),
                "closer": ((frame[col] - frame.actual).abs() < (frame.line - frame.actual).abs()).mean(),
                "bias_vs_line": (frame[col] - frame.line).mean()}

    out = [f"# THE ORACLE vs DraftKings props — {S} week {W}", "",
           f"{len(D)} two-sided DK props (last price before each kickoff) with a box score; projections from the last "
           f"Fan Picks bake before each game. Fans: {fan.fan.nunique()} ({', '.join(sorted(fan.fan.unique()))}).", ""]
    out += ["## Accuracy (lower MAE is better; 'closer' = share of props where the projection beat the line)", "",
            "| Market | n | Line MAE | Oracle MAE | Oracle closer | Oracle − line | Crowd MAE | Crowd closer |",
            "|---|---|---|---|---|---|---|---|"]
    for mk, f in [("ALL", D)] + list(D.groupby("market")):
        o, c = acc(f, "oracle"), acc(f, "crowd")
        out.append(f"| {mk} | {len(f)} | {o['line_mae']:.2f} | {o['mae']:.2f} | {o['closer']:.0%} | {o['bias_vs_line']:+.2f} | "
                   f"{c['mae']:.2f} | {c['closer']:.0%} |")
    out += ["", "## Betting the side the projection points to, at DK's pre-kick price", "",
            "| Projection | Min edge | Bets | Overs | Won | Hit % | ROI |", "|---|---|---|---|---|---|---|"]
    for col in ("oracle", "crowd"):
        for me in (0.0, 0.10, 0.20):
            b = bets(D, col, me)
            out.append(f"| {col} | {me:.0%} | {b['n']} | {b['overs']} | {b['won']} | {b['hit']:.1%} | {b['roi']:+.1%} |")
    out += ["", "Break-even at -110 is 52.4%.", "", "### By market (Oracle, every bet)", "",
            "| Market | Bets | Overs | Hit % | ROI | Crowd hit % | Crowd ROI |", "|---|---|---|---|---|---|---|"]
    for mk, f in D.groupby("market"):
        b, c = bets(f, "oracle"), bets(f, "crowd")
        out.append(f"| {mk} | {b['n']} | {b['overs']} | {b['hit']:.1%} | {b['roi']:+.1%} | {c['hit']:.1%} | {c['roi']:+.1%} |")

    # where the fans moved the number: did the move help, and did it flip the bet?
    T = D[D.fans > 0].copy()
    if len(T):
        T["moved_right"] = np.sign(T.crowd - T.oracle) == np.sign(T.actual - T.oracle)
        T["err_removed"] = (T.oracle - T.actual).abs() - (T.crowd - T.actual).abs()
        flip = T[np.sign(T.oracle - T.line) != np.sign(T.crowd - T.line)]
        fb, ob = bets(flip, "crowd"), bets(flip, "oracle")
        out += ["", "## Where fans touched the projection", "",
                f"{len(T)} props had at least one fan adjustment. The fans moved the number toward the result "
                f"{T.moved_right.mean():.0%} of the time; error removed {T.err_removed.sum():+.1f} units in total "
                f"({T.err_removed.mean():+.2f} per prop).",
                f"The adjustment flipped the side of the bet on {len(flip)} props: "
                + (f"crowd side won {fb['won']}/{fb['n']} (ROI {fb['roi']:+.1%}) where the Oracle's side won "
                   f"{ob['won']}/{ob['n']} ({ob['roi']:+.1%})." if len(flip) else "none.")]
        out += ["", "### Each fan + Oracle (props that fan touched)", "",
                "| Fan | Props | Toward result | Error removed | Flipped bets | Flipped won | Bets (fan+Oracle) | Hit % | ROI |",
                "|---|---|---|---|---|---|---|---|---|"]
        for fn in sorted(fan.fan.unique()):
            r = [(i, per_fan.get((fn, x.pid, x.market))) for i, x in D.iterrows()]
            idx = [i for i, v in r if v is not None and not pd.isna(v)]
            if not idx:
                continue
            F = D.loc[idx].copy()
            F["mine"] = F.oracle * [float(per_fan[(fn, x.pid, x.market)]) for x in F.itertuples()]
            tr = (np.sign(F.mine - F.oracle) == np.sign(F.actual - F.oracle)).mean()
            er = ((F.oracle - F.actual).abs() - (F.mine - F.actual).abs()).sum()
            fl = F[np.sign(F.oracle - F.line) != np.sign(F.mine - F.line)]
            fw = bets(fl, "mine")
            b = bets(F, "mine")
            out.append(f"| {fn} | {len(F)} | {tr:.0%} | {er:+.1f} | {len(fl)} | {fw['won']}/{fw['n']} | {b['n']} | {b['hit']:.1%} | {b['roi']:+.1%} |")

        # The fan's own opinion, without the Oracle: an up arrow is an over bet, a down arrow an under,
        # whatever the line. A boost on a TD stat is a Yes on DK's anytime-TD price (DK offers no No).
        tdp = snap[snap.market == "player_anytime_td"].drop_duplicates(["pid"], keep="last").set_index("pid").price
        out += ["", "### Each fan's own calls: bet the direction of every arrow", "",
                "Up arrow = over, down arrow = under, on every DK prop the fan touched, at DK's pre-kick price; "
                "TD boosts = anytime-TD Yes. 1 unit per bet.", "",
                "| Fan | O/U bets | Overs | Won | Hit % | Units | ROI | TD Yes bets | TD won | TD units | All units | All ROI |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows_fan = []
        for fn in sorted(fan.fan.unique()):
            res = []
            for x in D.itertuples():
                v = per_fan.get((fn, x.pid, x.market))
                if v is None or pd.isna(v) or v == 1:
                    continue
                over = v > 1
                won = x.actual > x.line if over else x.actual < x.line
                res.append((over, won, payout(x.over_price if over else x.under_price) if won else -1.0))
            tds = []
            for pid in {g for (f_, g, s_) in per_fan.index if f_ == fn and s_ in ("rush_tds", "rec_tds")}:
                if pid not in tdp.index or pid not in box.index:
                    continue
                ups = [per_fan.get((fn, pid, s)) for s in ("rush_tds", "rec_tds")]
                ups = [u for u in ups if u is not None and not pd.isna(u)]
                if not ups or max(ups) <= 1:
                    continue
                scored = float((box.loc[pid, "rushing_tds"] or 0) + (box.loc[pid, "receiving_tds"] or 0)) > 0
                tds.append(payout(tdp[pid]) if scored else -1.0)
            if not res and not tds:
                continue
            u = sum(r[2] for r in res); tu = sum(tds); n = len(res)
            rows_fan.append((fn, n, sum(r[0] for r in res), sum(r[1] for r in res), u, len(tds), sum(t > 0 for t in tds), tu))
        for fn, n, ov, w, u, nt, wt, tu in sorted(rows_fan, key=lambda r: -(r[4] + r[7])):
            out.append(f"| {fn} | {n} | {ov} | {w} | {w / n if n else float('nan'):.1%} | {u:+.1f} | {u / n if n else float('nan'):+.1%} | "
                       f"{nt} | {wt} | {tu:+.1f} | {u + tu:+.1f} | {(u + tu) / max(1, n + nt):+.1%} |")

    # anytime TD
    td = snap[snap.market == "player_anytime_td"].drop_duplicates(["pid"], keep="last")
    trows = []
    for r in td.itertuples():
        if r.pid not in box.index:
            continue
        _, p, _ = orc[r.pid]
        lam = float(p["stats"].get("rush_tds", 0) + p["stats"].get("rec_tds", 0))
        fr = np.mean([fan_ratio.get((r.pid, s), 1.0) for s in ("rush_tds", "rec_tds")])
        scored = float((box.loc[r.pid, "rushing_tds"] or 0) + (box.loc[r.pid, "receiving_tds"] or 0)) > 0
        trows.append({"player": p["name"], "price": r.price, "imp": implied(r.price), "p_oracle": 1 - np.exp(-lam),
                      "p_crowd": 1 - np.exp(-lam * fr), "scored": scored})
    if trows:
        TD = pd.DataFrame(trows)
        out += ["", "## Anytime TD (one-sided Yes prices; DK's implied probability includes the vig)", "",
                f"{len(TD)} players; {TD.scored.sum()} scored. Mean P: DK implied {TD.imp.mean():.1%}, Oracle {TD.p_oracle.mean():.1%}, "
                f"crowd {TD.p_crowd.mean():.1%}; actual rate {TD.scored.mean():.1%}.", "",
                "| Probability | Brier | Bets (P > DK implied) | Won | ROI |", "|---|---|---|---|---|"]
        for col, lab in (("imp", "DK implied"), ("p_oracle", "Oracle"), ("p_crowd", "Crowd")):
            brier = ((TD[col] - TD.scored) ** 2).mean()
            if col == "imp":
                out.append(f"| {lab} | {brier:.4f} | — | — | — |"); continue
            B = TD[TD[col] > TD.imp]
            prof = np.where(B.scored, [payout(x) for x in B.price], -1.0)
            out.append(f"| {lab} | {brier:.4f} | {len(B)} | {int(B.scored.sum())} | {prof.mean() if len(B) else float('nan'):+.1%} |")

    out += ["", "_One week is a small sample: at ~200 bets the standard error on a hit rate is about ±3.5 points._"]
    rp = os.path.join(ROOT, "outputs", "reports", f"oracle_vs_props_{S}_wk{W}")
    open(rp + ".md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    D.to_csv(rp + ".csv", index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
