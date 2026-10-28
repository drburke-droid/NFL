"""Bake the league Values tab: docs/values_2026.js (global VALUES_DATA), read by docs/values.html.

Three questions, one file:

  best VALUES   what a player is worth to whoever holds him over every keeper year he has left,
                net of what keeping him costs. Omar Cooper Jr. is the case this exists for: a $1
                waiver add, three keeper years left, 22 years old, hurt after one catch in week 1 --
                his season total is low and every keeper year is nearly free.
  best PLAYERS  the healthy rest-of-season level, the fitted ridge gm/ uses for trades.
  BUYERS and SELLERS at the deadline, from the simulated playoff odds and the 3-keeper cap: a
                contender's 4th-best keeper is worth nothing to him and something to a seller, and
                a seller's rental is worth nothing to him and something to a contender.

The healthy level. gm/players.py fits its ridge on every game a player had, so the game he left
after five snaps counts as a full game at 1 point. Here a game counts only if he played a normal
share of his team's snaps: nflverse snap counts, a game below 40% of the player's own median
offensive share (and under 35% outright) is a partial game and is dropped from this season's
features before the same ridge is applied. A backup who always plays 20% keeps every game; a
starter carted off in the first quarter loses that one. Both the raw and the healthy level are
carried, so the page can show what the filter changed.

Keeper value, every remaining year. A player has max_times_kept - times_kept keeper years left
(Cooper: 3 - 0 = 3). Year k costs his basis ($1 for any waiver add, per league rule; his auction
price otherwise) plus the owner's bump: the expected bump from his playoff odds the first year
(gm/keepers.expected_bump), the league's +$5 each year after. Year k's worth is the league's
auction dollars (gm/keepers.price_curve: what this league has bid, by position) at his level after k years of the measured year-over-
year drift for his position and age, with the spread widening by sqrt(k). The owner keeps him only
in the years he is worth his price, so each year contributes E[max(dollars - cost, 0)], and later
years count less: x0.85 per year, for everything that can change in a year that the drift table
does not know (role, team, the league's own rules). That is a choice, stated on the page.

Free agents are on the board too, at the waiver basis: whoever adds one this week gets the same $1
keeper Cooper is.

Usage: python scripts/league_values_bake.py [--sims 5000] [--seed 0]
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm import config, keepers as kp, players as gp, simulate as sim  # noqa: E402

LEAGUE = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")
OUT = os.path.join(ROOT, "docs", "values_{season}.js")
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet"
SKILL = gp.SKILL
PARTIAL_REL, PARTIAL_ABS, PARTIAL_MED_MIN = 0.40, 0.35, 0.25   # partial game: < 40% of own median AND < 35%
DISCOUNT = 0.85            # each later keeper year counts this much less than the one before
LAST_WEEK = 17             # fantasy season, playoffs included
FA_PER_POS = 25            # free agents carried per position (by healthy level)
CHIP_MIN = 8.0             # a 4th+ keeper worth at least this many $ (league prices) is a trade chip
RENTAL_MIN_PAR = 2.0       # a rental: outside the owner's top 3 keepers, 2+ points a week over the waiver wire
MATCH_MIN_GAIN = 1.0       # a match: the buyer's lineup gains 1+ point a week ...
MATCH_MIN_KEEP = 3.0       # ... and the seller's best-three keeper total gains $3+
BUYER_P, SELLER_P = 0.50, 0.25
# A free agent is on the board only with an active role: 2+ games this season, and either his last
# three weeks or next week's FFA projection at or above this. The ridge was validated on rostered
# players; on a backup with one game it falls back toward the position's intercept, which put
# third-string quarterbacks at 20 points a week and dragged the QB waiver line up with them.
FA_ROLE = {"QB": 12.0, "RB": 6.0, "WR": 6.0, "TE": 4.0}


def age_at(birth, season):
    """Age on 1 September of `season` (the start of that season), or None."""
    try:
        b = pd.Timestamp(birth)
    except Exception:
        return None
    if pd.isna(b):
        return None
    d = pd.Timestamp(f"{season}-09-01")
    return int(d.year - b.year - ((d.month, d.day) < (b.month, b.day)))


def partial_games(weekly, season, snaps, ids):
    """Set of (gsis_id, week) this season in which the player played a partial game."""
    if snaps is None or snaps.empty:
        return set()
    s = snaps[(snaps.season == season) & (snaps.game_type == "REG")][["pfr_player_id", "week", "offense_pct"]]
    s = s.merge(ids[["pfr_id", "gsis_id"]].dropna(), left_on="pfr_player_id", right_on="pfr_id")
    cur = weekly[weekly.season == season][["player_id", "week"]]
    s = s.merge(cur, left_on=["gsis_id", "week"], right_on=["player_id", "week"])   # games he has a stat line in
    med = s.groupby("gsis_id").offense_pct.transform("median")
    bad = s[(med >= PARTIAL_MED_MIN) & (s.offense_pct < PARTIAL_REL * med) & (s.offense_pct < PARTIAL_ABS)]
    return set(zip(bad.gsis_id, bad.week))


def level_dist_value(curve, pos, level, ages, cost, years, mult=1.0, cap=None):
    """[(year k, expected surplus $, expected price $)] for k = 1..years.

    surplus = E[max(dollars(level_k) - cost_k, 0)] over the outcomes; price = dollars at his expected
    level that year, what the league would charge for him at that year's auction. `mult` is his own price premium (gm/keepers.price_premium)."""
    z = np.linspace(-2.5, 2.5, 41); w = np.exp(-0.5 * z ** 2); w /= w.sum()
    out, mu, var = [], float(level), 0.0
    for k in range(1, years + 1):
        a = ages[k - 1]
        drift, sd = kp.drift_vs_field(pos, a)          # the whole field ages; prices are by rank
        mu += drift; var += sd ** 2
        grid = mu + np.sqrt(var) * z
        dollars = mult * np.array([kp.dollars_at(curve, pos, g) for g in grid])
        if cap is not None:
            dollars = np.minimum(dollars, cap)            # never above what the room has ever paid
        # the price is the room's: what it pays for his EXPECTED level that year. The surplus is the
        # keeper's: averaged over the outcomes, because he keeps him only in the years he is worth it.
        # (Averaging the price too pulled the best players down -- the top of a position can only
        # fall -- and left one player at $60+ where this room pays that three or four times a year.)
        price = min(mult * kp.dollars_at(curve, pos, mu), cap if cap is not None else np.inf)
        out.append((k, float(np.sum(w * np.maximum(dollars - cost[k - 1], 0.0))), float(price)))
    return out


def lineup(cfg, levels):
    """Expected starting lineup from {pos: [levels]}: returns {slot: [levels started]}."""
    st = dict(cfg.starters); flex = st.pop("FLEX", 0)
    by = {p: sorted(levels.get(p, []), reverse=True) for p in SKILL}
    used = {p: by[p][:st.get(p, 0)] for p in SKILL}
    rest = sorted([v for p in ("RB", "WR", "TE") for v in by[p][st.get(p, 0):]], reverse=True)
    used["FLEX"] = rest[:flex]
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default=LEAGUE)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = config.load(a.league)
    season = int(cfg.raw["season"])
    kcfg = cfg.raw.get("keepers", {})
    max_kept = int(kcfg.get("max_times_kept", 3)); max_keepers = int(kcfg.get("max_per_team", 3))
    base_bump = float((kcfg.get("inflation") or {}).get("base", kp.NEXT_BUMP))

    # ---------- playoff odds (the gm/ simulation, as gm_report runs it) ----------
    est = gp.ros_estimates(cfg)
    need, _ = sim.weeks_needed(cfg)
    scores = sim.player_team_scores(cfg, est, a.sims, need, rng=np.random.default_rng(a.seed),
                                    common_seed=a.seed, weeks=sim.week_columns(cfg))
    odds = sim.run(cfg, scores)["teams"]
    bump = {tid: kp.expected_bump(cfg, odds[tid]["p_playoffs"]) for tid in odds}

    # ---------- weekly scores, partial games, the healthy level ----------
    weekly = gp._weekly_all(cfg, season)
    week = int(weekly[weekly.season == season].week.max())
    ids = pd.read_parquet(ROSTER_URL.format(season=season))
    ids["week"] = pd.to_numeric(ids.week, errors="coerce")
    ids = ids.sort_values("week").drop_duplicates("gsis_id", keep="last")
    try:
        snaps = pd.read_parquet(SNAP_URL.format(season=season))
    except Exception as ex:
        print(f"  snap counts unavailable ({ex}); no partial games dropped"); snaps = None
    partial = partial_games(weekly, season, snaps, ids)
    keep = [(pid, w) not in partial or s != season for pid, w, s in zip(weekly.player_id, weekly.week, weekly.season)]
    weekly_full = weekly[keep]
    ffa, ffa_stale = gp.ffa_next_week(cfg, season, week)
    model = gp.load_model()
    f_full = gp.history_features(weekly_full, season, week, ffa)
    f_raw = gp.history_features(weekly, season, week, ffa)
    lvl_full = gp.ridge_predict(model, f_full).clip(lower=0)
    lvl_raw = gp.ridge_predict(model, f_raw).clip(lower=0)
    cur = weekly[weekly.season == season]
    ppg = cur.groupby("player_id").pts.agg(["mean", "size"])
    cur_full = weekly_full[weekly_full.season == season]
    ppg_full = cur_full.groupby("player_id").pts.agg(["mean", "size"])
    last3 = cur[cur.week > week - 3].groupby("player_id").pts.mean()
    part_wk = {}
    for pid, w in partial:
        part_wk.setdefault(pid, []).append(int(w))

    byes = gp.bye_weeks(season)
    weeks_left = list(range(week + 1, LAST_WEEK + 1))
    espn2gsis = {str(e): g for e, g in zip(ids.espn_id, ids.gsis_id) if pd.notna(e)}
    birth = dict(zip(ids.gsis_id, ids.birth_date))
    team_of = dict(zip(ids.gsis_id, ids.team))
    key2gsis = {}
    for pid, r in f_full.iterrows():
        key2gsis.setdefault((r["key"], r["position"]), pid)
    # what THIS league pays for production at each position (its own auction bids); the model's curve
    # only for a position without enough bids
    curve = kp.price_curve(cfg)                  # the same prices the trade calculator values keepers on
    prem = kp.premiums(cfg)
    cap_of = kp.caps(cfg)
    rank_points = kp.league_price_curve.rank_points
    bids_all = kp.league_bids(cfg)

    # ---------- the player universe: every rostered skill player, plus the best free agents ----------
    teams = {str(t["team_id"]): t for t in cfg.teams}
    rows, taken = [], set()
    for tid, t in teams.items():
        for e in t["roster"]:
            if e["pos"] not in SKILL:
                continue
            g = espn2gsis.get(str(e["player_id"])) or key2gsis.get((gp.norm(e["name"]), e["pos"]))
            taken.add(g)
            rows.append({"tid": tid, "e": e, "g": g})
    fa = f_full.assign(lvl=lvl_full, l3=last3.reindex(f_full.index))
    role = np.maximum(fa.l3.fillna(0), fa.ffa_next.fillna(0)) >= fa.position.map(FA_ROLE).fillna(99)
    fa = fa[(fa.ytd_games.fillna(0) >= 2) & fa.position.isin(SKILL) & ~fa.index.isin(taken) & role]
    for p in SKILL:
        for g, r in fa[fa.position == p].sort_values("lvl", ascending=False).head(FA_PER_POS).iterrows():
            rows.append({"tid": "FA", "e": {"name": r["player_display_name"], "pos": p, "acquired": "ADD",
                                            "keeper_cost": {"price": 0}, "times_kept": 0, "injury": "",
                                            "nfl_team": None}, "g": g})
    # the waiver line each position is measured against: the mean of its five best free agents
    repl = {p: float(fa[fa.position == p].lvl.sort_values(ascending=False).head(5).mean()) for p in SKILL}

    avg_bump = float(np.mean(list(bump.values())))
    players = []
    for r in rows:
        e, g, tid = r["e"], r["g"], r["tid"]
        pos = e["pos"]
        lv = float(lvl_full.get(g, np.nan)) if g is not None else np.nan
        lr = float(lvl_raw.get(g, np.nan)) if g is not None else np.nan
        if not np.isfinite(lv):
            lv = repl[pos] * 0.5                          # no NFL history at all: a guess below the wire
        nfl = gp.TEAM_FIX.get(e.get("nfl_team"), e.get("nfl_team"))
        if not nfl and g in team_of:
            nfl = gp.TEAM_FIX.get(team_of[g], team_of[g])
        bye = byes.get(nfl)
        games_left = len([w for w in weeks_left if w != bye])
        price = float((e.get("keeper_cost") or {}).get("price") or 0)
        waiver = str(e.get("acquired") or "").upper() == "ADD"
        basis = 1.0 if (waiver or price <= 0) else price
        tk = int(e.get("times_kept") or 0)
        yrs = max(max_kept - tk, 0)
        age0 = age_at(birth.get(g), season) if g is not None else None
        ages = [(age0 + k) if age0 is not None else None for k in range(1, max(yrs, 1) + 1)]
        b1 = bump.get(tid, avg_bump)
        cost = [basis + b1 + base_bump * (k - 1) for k in range(1, max(yrs, 1) + 1)]
        mult = prem.get((kp._norm(e["name"]), pos), 1.0)
        # next year's auction price is priced for everyone -- a player at the keeper cap (Puka Nacua,
        # Jaxon Smith-Njigba) goes back into the auction, and that price is what he will cost there
        nxt_year = level_dist_value(curve, pos, lv, ages, cost, max(yrs, 1), mult, cap_of.get(pos))
        by_year = nxt_year[:yrs]
        kval = sum(DISCOUNT ** (k - 1) * v for k, v, _ in by_year)
        par = max(lv - repl[pos], 0.0)
        pf = f_full.loc[g] if g is not None and g in f_full.index else None
        nxt = float(pf["ffa_next"]) if pf is not None and pd.notna(pf.get("ffa_next")) else None
        if nxt is not None and bye == week + 1:
            nxt = 0.0
        players.append({
            "n": e["name"], "p": pos, "t": nfl or "", "o": tid, "inj": e.get("injury") or "",
            "sal": price, "basis": basis, "acq": e.get("acquired") or "", "tk": tk, "yrs": yrs,
            "age": age0, "lvl": round(lv, 2), "lvl_raw": round(lr, 2) if np.isfinite(lr) else None,
            "ppg": round(float(ppg["mean"].get(g)), 2) if g in ppg.index else None,
            "gp": int(ppg["size"].get(g)) if g in ppg.index else 0,
            "ppg_full": round(float(ppg_full["mean"].get(g)), 2) if g in ppg_full.index else None,
            "gp_full": int(ppg_full["size"].get(g)) if g in ppg_full.index else 0,
            "partial": sorted(part_wk.get(g, [])), "l3": round(float(last3.get(g)), 2) if g in last3.index else None,
            "next": round(nxt, 2) if nxt is not None else None, "bye": bye,
            "par": round(par, 2), "ros_par": round(par * games_left, 1), "gl": games_left,
            "kval": round(kval, 1), "kyears": [round(v, 1) for _, v, _ in by_year],
            "cost1": round(cost[0], 2) if yrs else None,
            "price1": round(nxt_year[0][2], 1), "prem": round(mult, 2),
            "margin1": round(nxt_year[0][2] - cost[0], 1) if yrs else None})

    # ---------- teams: keepers locked in, chips, rentals, the lineup, the verdict ----------
    def total(levels):
        return sum(sum(v) for v in lineup(cfg, levels).values())

    def by_pos(ps):
        return {pos: [p["lvl"] for p in ps if p["p"] == pos] for pos in SKILL}

    def top_keep(ps):
        return sum(sorted((p["kval"] for p in ps), reverse=True)[:max_keepers])

    roster_of = {tid: [p for p in players if p["o"] == tid] for tid in teams}
    tout = []
    for tid, t in teams.items():
        mine = roster_of[tid]
        ranked = sorted(mine, key=lambda p: -p["kval"])
        for i, p in enumerate(ranked):
            p["keep_rank"] = i + 1
        locked = [p["n"] for p in ranked[:max_keepers] if p["kval"] > 0]
        third = ranked[max_keepers - 1]["kval"] if len(ranked) >= max_keepers else 0.0
        # a chip: a keeper he cannot keep (4th or later on his own list) that is still worth real money
        chips = [p["n"] for p in ranked[max_keepers:] if p["kval"] >= CHIP_MIN]
        # a rental: a useful player outside his top three keepers -- worth this season and nothing after
        rentals = [p["n"] for p in sorted(mine, key=lambda p: -p["par"])
                   if p["keep_rank"] > max_keepers and p["par"] >= RENTAL_MIN_PAR]
        lu = lineup(cfg, by_pos(mine))
        weakest = {slot: round(min(v), 2) for slot, v in lu.items() if v}
        o = odds.get(tid, {})
        pp, pt = float(o.get("p_playoffs", 0)), float(o.get("p_title", 0))
        role = "buyer" if pp >= BUYER_P else "seller" if pp < SELLER_P else "bubble"
        tout.append({"id": tid, "name": t["name"], "owner": t.get("owner", ""), "w": t["record"]["w"],
                     "l": t["record"]["l"], "pf": t.get("points_for"), "p_playoffs": round(pp, 3),
                     "p_title": round(pt, 3), "seed": round(float(o.get("exp_seed", 0)), 2), "role": role,
                     "bump": round(bump.get(tid, avg_bump), 2), "locked": locked, "third_keeper": round(third, 1),
                     "keep_total": round(top_keep(mine), 1), "chips": chips, "rentals": rentals, "weakest": weakest,
                     "lineup_pts": round(sum(sum(v) for v in lu.values()), 1)})

    # ---------- deadline matches: a seller's rental for a buyer's chip, priced on both sides ----------
    # The buyer's gain is his whole lineup after the swap (the chip may be starting for him now), and the
    # seller's is his best-three keeper total after it: the chip has to break into his top three to matter.
    by_name = {(p["o"], p["n"]): p for p in players}
    matches = []
    for s_ in tout:
        if s_["role"] != "seller":
            continue
        s_ros = roster_of[s_["id"]]; s_keep = top_keep(s_ros)
        for rn in s_["rentals"]:
            r = by_name[(s_["id"], rn)]
            for b in tout:
                if b["role"] != "buyer":
                    continue
                b_ros = roster_of[b["id"]]; b_tot = total(by_pos(b_ros))
                best = None
                for cn in b["chips"]:
                    c = by_name[(b["id"], cn)]
                    gain = total(by_pos([p for p in b_ros if p is not c] + [r])) - b_tot
                    kgain = top_keep([p for p in s_ros if p is not r] + [c]) - s_keep
                    if gain < MATCH_MIN_GAIN or kgain < MATCH_MIN_KEEP:
                        continue
                    # the cheapest chip that still does it: the least keeper value the buyer has to give up
                    if best is None or c["kval"] < best[2]["kval"]:
                        best = (gain, kgain, c)
                if best:
                    gain, kgain, c = best
                    matches.append({"seller": s_["id"], "buyer": b["id"], "player": rn, "pos": r["p"],
                                    "chip": c["n"], "chip_pos": c["p"], "gain": round(gain, 2),
                                    "ros_gain": round(gain * r["gl"], 1), "keep_gain": round(kgain, 1)})
    matches.sort(key=lambda m: -m["ros_gain"])

    out = {"season": season, "week": week, "next_week": week + 1, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "league": cfg.raw["name"], "my_team": str(cfg.raw.get("my_team_id")), "sims": a.sims,
           "ffa_stale": bool(ffa_stale), "partial_games": len(partial), "repl": {p: round(v, 2) for p, v in repl.items()},
           "rules": {"max_keepers": max_keepers, "max_times_kept": max_kept, "base_bump": base_bump,
                     "discount": DISCOUNT, "deadline_week": (cfg.raw.get("acquisitions") or {}).get("trade_deadline_week"),
                     "buyer_p": BUYER_P, "seller_p": SELLER_P, "chip_min": CHIP_MIN,
                     "rental_min_par": RENTAL_MIN_PAR, "match_min_gain": MATCH_MIN_GAIN, "match_min_keep": MATCH_MIN_KEEP,
                     "partial": {"rel": PARTIAL_REL, "abs": PARTIAL_ABS}},
           "calibration": {"bids_60_by_auction": {str(s_): sum(1 for _, _, b, ss, _ in bids_all if ss == s_ and b >= 60)
                                                  for s_ in sorted({x[3] for x in bids_all})},
                           "top_bids": sorted([{"n": n, "p": p_, "bid": b, "s": ss} for p_, _, b, ss, n in bids_all if b >= 60],
                                              key=lambda x: (-x["s"], -x["bid"]))},
           "price_curve": {p: {"source": "league bids" if p in rank_points else "model",
                               "rank": [[float(r), round(float(v), 1)] for r, v in zip(*rank_points[p])] if p in rank_points else []}
                           for p in SKILL},
           "teams": tout, "players": players, "matches": matches[:40]}
    path = OUT.format(season=season)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("// baked by scripts/league_values_bake.py -- do not edit\nwindow.VALUES_DATA=")
        json.dump(out, fh, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")
    print(f"wrote {os.path.relpath(path, ROOT)}: {len(players)} players ({sum(p['o'] == 'FA' for p in players)} free agents), "
          f"{len(tout)} teams, {len(matches)} deadline matches, {len(partial)} partial games dropped, through week {week}")


if __name__ == "__main__":
    main()
