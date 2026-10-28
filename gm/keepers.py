"""What a roster is worth NEXT season, so a trade can be judged on more than this year's title.

The search so far asks only whether a trade raises P(title) now. In a keeper league that is the
wrong question for half the teams: a side sitting on 1% title odds should be selling this season
for next, and every such trade looks like a loss to a win-now objective. This supplies the other
term.

It does not recompute keeper economics -- scripts/predict_keepers.py already does that, against
the league's real rules, several of which are not guessable:

  * cost = last year's corrected basis + the CURRENT OWNER's inflation bump, and that bump depends
    on where the owner finished (+$5, +$3 if they missed the playoffs, +$0 if they also drafted the
    champion). So a player's keeper cost CHANGES when he is traded, because he inherits the new
    owner's bump. A contender pays more to keep the same player than a cellar team does.
  * acquisitionType decides the basis. TRADE carries the drafted salary to the new owner; ADD --
    any mid-season waiver pickup, including a drafted player who was dropped and re-added --
    resets it to $1. That means a roster cut is not merely a lost player: whoever claims him next
    gets a $1 keeper.
  * at most three keepers per team, so the fourth-best surplus on a roster is worth nothing, in
    exactly the way a fourth startable receiver is worth nothing.

That last rule is why this cannot be a per-player number added up. A roster's keeper value is the
sum of its best three surpluses, so what a player adds depends entirely on what is already there.
"""
import json
import os
import re

import numpy as np

MAX_KEEPERS = 3
NEXT_BUMP = 5          # the owner's inflation bump next year is not known until the standings settle
YOUNG = 24             # age at which the year-over-year drift flips sign (below: rises; above: fades)
# Year-over-year change in a player's league-scored ppg, players with 6+ games both seasons, 2012-25
# nflverse: (mean drift, sd) by position and whether he was 24 or younger. Young players drift up
# by half a point and everyone else down by the same, and the spread is four points at RB/WR,
# three at TE, five at QB. That spread is what gives a cheap young player option value: the dollar
# curve is convex, so the expected dollars of an uncertain player exceed the dollars of his
# expected level.
YOY = {("QB", True): (0.69, 5.27), ("QB", False): (-0.46, 5.15),
       ("RB", True): (0.30, 4.45), ("RB", False): (-0.87, 4.01),
       ("WR", True): (0.58, 3.58), ("WR", False): (-0.88, 3.61),
       ("TE", True): (0.46, 3.00), ("TE", False): (-0.45, 2.81)}
# Everyone ages, so against a price set by RANK a player moves only by how his drift differs from the
# position's: the mean of the young and the older drift (equal weight -- an assumption; the study above
# measured the two groups, not their mix). Without it every veteran slid down a field that stood
# still, and the 2027 board had two players at $60+ where this room pays that three or four times a year.
POP_DRIFT = {pos: (YOY[(pos, True)][0] + YOY[(pos, False)][0]) / 2.0 for pos in ("QB", "RB", "WR", "TE")}


def drift_vs_field(pos, age):
    """(drift relative to the position's field, sd) for one year at `age` (None: the older group)."""
    drift, sd = YOY.get((pos, bool(age is not None and age <= YOUNG)), (0.0, 4.0))
    return drift - POP_DRIFT.get(pos, 0.0), sd


def load_board(path=None, root=None):
    """The keeper board predict_keepers.py emits, as {team_id: {"bump": int, "players": {...}}}."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.path.join(root, "outputs", "draft_tool", "keepers_2026.js")
    s = open(path).read()
    d = json.loads(s[s.index("{"):s.rindex(";")])
    out = {}
    for t in d["teams"]:
        out[str(t["id"])] = {
            "bump": t["bump2026"],
            "players": {c["name"]: c for c in t["candidates"]},
        }
    return out


def surplus_under(entry, bump):
    """A player's keeper surplus if the owner's inflation bump were `bump`.

    cost = basis (or $1 for a waiver pickup) + the owner's bump, so moving a player to a team with
    a different finish changes what keeping him costs without changing what he is worth.
    """
    base = 1 if entry.get("waiver") else entry.get("basis", 0)
    return entry["value"] - (base + bump)


def team_keeper_value(board, team_id, names, bump=None, max_keepers=MAX_KEEPERS):
    """Best `max_keepers` surpluses on a roster. Players off the board contribute nothing."""
    t = board.get(str(team_id))
    if not t:
        return 0.0
    bump = t["bump"] if bump is None else bump
    vals = []
    for n in names:
        e = t["players"].get(n)
        if e is None:
            continue
        vals.append(surplus_under(e, bump))
    return float(sum(sorted(vals, reverse=True)[:max_keepers]))


def board_lookup(board, name):
    """Find a player's entry on whichever roster currently holds him."""
    for tid, t in board.items():
        if name in t["players"]:
            return tid, t["players"][name]
    return None, None


def keeper_delta(board, rosters_before, rosters_after, teams):
    """{team_id: change in keeper value} for a trade, honouring the new owner's bump.

    rosters_* are {team_id: [player names]}. A player who appears in neither after-roster was cut,
    and his keeper value is simply gone -- for the team that held him.
    """
    out = {}
    for t in teams:
        t = str(t)
        bump = board.get(t, {}).get("bump", 0)
        before = team_keeper_value(board, t, rosters_before.get(t, []))
        # a player arriving from elsewhere brings his basis but pays THIS owner's bump
        vals = []
        for n in rosters_after.get(t, []):
            src, e = board_lookup(board, n)
            if e is None:
                continue
            vals.append(surplus_under(e, bump))
        after = float(sum(sorted(vals, reverse=True)[:MAX_KEEPERS]))
        out[t] = after - before
    return out


# ----------------------------------------------------------------- next season, from this one
def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def draft_tool_players(root=None):
    """The draft tool's player file (docs/data.js): preseason projection and age by name."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = open(os.path.join(root, "docs", "data.js"), encoding="utf-8").read()
    P = json.loads(s.split("const PLAYERS = ")[1].rsplit(";", 1)[0])
    return {(_norm(p["name"]), p["position"]): p for p in P}


def _isotonic(x, y):
    """Pool-adjacent-violators: the closest non-decreasing fit of y on sorted x."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks = [[ys[i], 1] for i in range(len(ys))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m, n = blocks[i], blocks[i + 1]
            merged = [(m[0] * m[1] + n[0] * n[1]) / (m[1] + n[1]), m[1] + n[1]]
            blocks[i:i + 2] = [merged]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.concatenate([[b[0]] * b[1] for b in blocks])
    return xs, fitted


def dollar_curve(board=None, players=None):
    """{pos: (ppg grid, dollars)}: what this league pays at auction for a level of weekly scoring.

    Fitted from the keeper board's calibrated values against the draft tool's projected ppg for
    the same players, made monotone. The shape is the point: RB 8 ppg is $1, 11 is $6, 14 is $29.
    """
    board = board or load_board()
    players = players or draft_tool_players()
    pairs = {}
    for t in board.values():
        for name, e in t["players"].items():
            p = players.get((_norm(name), e["pos"]))
            if p and p.get("proj_ppg") is not None:
                pairs.setdefault(e["pos"], []).append((float(p["proj_ppg"]), float(e["value"])))
    return {pos: _isotonic(*zip(*v)) for pos, v in pairs.items() if len(v) >= 8}


# The league's auction prices, by order: the room's k-th most expensive player at a position, averaged
# over its auctions with the recent ones counting more (prices have climbed: the top running back went
# for $62, $44, $68, $68 in 2023-26), is the price of the k-th best player at that position.
RECENCY = 1.0              # an auction n seasons back counts 1 / (1 + RECENCY * n)
# A premium only ever raises a price. A star the room keeps paying up for is a brand; a $1 pick at a
# rank that usually costs $15 is mostly when he was nominated (budgets run dry late), and next year's
# room bids on what he has done since. Floored at 1, so Bucky Irving and Quinshon Judkins ($1 in 2026)
# are priced by their rank alone.
PREMIUM_SHRINK, PREMIUM_CAP = 0.5, (1.0, 2.0)
PREMIUM_MIN_BASE = 10.0    # below this rank price a ratio is noise ($10 on a $3 rank is not a brand)


def _keepers_final(root, season):
    """Normalised names of the players kept INTO `season` (docs/keepers_final_<season>.js)."""
    p = os.path.join(root, "docs", f"keepers_final_{season}.js")
    if not os.path.exists(p):
        return set()
    s = open(p, encoding="utf-8").read()
    return {_norm(k.split("|")[0]) for k in json.loads(s[s.index("{"):s.rindex("}") + 1])["keepers"]}


def league_bids(cfg, drafts_path=None, min_games=6):
    """Every non-keeper auction bid this league has made, with the rank the room was bidding on.

    Past auctions (outputs/espn_drafts.json, 2023-25): the player's rank at his position by the
    previous season's ppg in league scoring (6+ games) -- the production the room had seen. This
    season's auction: the price on each rostered player acquired by DRAFT who was not a keeper, at
    his rank by the preseason projection (docs/data.js) -- the room's expectation, the better proxy.
    Returns [(pos, rank, bid, season, name)].
    """
    from gm import players as gp            # local: players imports nothing from here, keep it that way
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    season = int(cfg.raw["season"])
    out = []
    path = drafts_path or os.path.join(root, "outputs", "espn_drafts.json")
    bids = []
    if os.path.exists(path):
        bids = [r for r in json.load(open(path, encoding="utf-8"))
                if str(r.get("keeper")).lower() not in ("true", "1") and r.get("pos") in gp.SKILL
                and r.get("bid") and int(r["season"]) < season]
    if bids:
        seasons = sorted({int(r["season"]) for r in bids})
        w = gp.weekly_in_league_scoring(cfg, range(seasons[0] - 1, seasons[-1]), seasons[-1])
        ss = w.groupby(["key", "position", "season"]).pts.agg(["mean", "size"]).reset_index()
        ss = ss[ss["size"] >= min_games]
        ss["rank"] = ss.groupby(["position", "season"])["mean"].rank(ascending=False, method="first")
        rank = {(r.key, r.position, int(r.season)): float(r["rank"]) for _, r in ss.iterrows()}
        for r in bids:
            k = (gp.norm(r["player"]), r["pos"], int(r["season"]) - 1)
            if k in rank:
                out.append((r["pos"], rank[k], float(r["bid"]), int(r["season"]), r["player"]))
    # this season's auction, at the preseason projection rank
    players = draft_tool_players(root)
    prank = {}
    for pos in gp.SKILL:
        pool = sorted(((float(p["proj_ppg"]), k) for k, p in players.items()
                       if k[1] == pos and p.get("proj_ppg") is not None), reverse=True)
        for i, (_, k) in enumerate(pool):
            prank[k] = float(i + 1)
    kept = _keepers_final(root, season)
    for t in cfg.teams:
        for e in t["roster"]:
            k = (_norm(e["name"]), e["pos"])
            price = float((e.get("keeper_cost") or {}).get("price") or 0)
            if (e["pos"] in gp.SKILL and str(e.get("acquired") or "").upper() == "DRAFT" and k[0] not in kept
                    and price > 0 and k in prank):
                out.append((e["pos"], prank[k], price, season, e["name"]))
    return out


def rank_prices(bids, min_bids=8):
    """{pos: (ranks 1..K, price)}: what the room pays for the k-th best player at each position.

    By order, not by any measure of the player: in every auction the k-th most expensive player at
    the position, averaged over auctions (recent ones count more, RECENCY), non-increasing in k.
    Order statistics keep the top honest -- this room has paid $60+ three or four times in most
    auctions (Jefferson $75, Bijan $68 twice, Lamb $65 twice), and a median or a fit against
    last year's points averages those away into the fifties. A position with fewer than `min_bids`
    bids is left out. Keepers never reach the auction, so the k-th price belongs to roughly the
    k-th best player AVAILABLE, a little below the k-th best overall: if anything this runs low
    at the very top.
    """
    out = {}
    for pos in {b[0] for b in bids}:
        v = [(s_, b) for p, _, b, s_, _ in bids if p == pos]
        if len(v) < min_bids:
            continue
        last = max(s_ for s_, _ in v)
        by = {}
        for s_, b in v:
            by.setdefault(s_, []).append(b)
        K = max(len(x) for x in by.values())
        prices = []
        for k in range(K):
            num = den = 0.0
            for s_, x in by.items():
                x = sorted(x, reverse=True)
                if k < len(x):
                    w = 1.0 / (1.0 + RECENCY * (last - s_))
                    num += w * x[k]; den += w
            prices.append(num / den)
        p_ = np.minimum.accumulate(np.array(prices))           # a better rank never costs less
        out[pos] = (np.arange(1, K + 1, dtype=float), p_)
    return out


def price_at_rank(rp, pos, rank):
    r_, p_ = rp[pos]
    return float(np.interp(rank, r_, p_, left=p_[0], right=p_[-1]))


def current_levels(cfg, min_games=2):
    """{pos: levels, best first}: the fitted rest-of-season ridge for every player at the position
    with `min_games`+ games this season -- the population a player's rank is read against."""
    from gm import players as gp
    season = int(cfg.raw["season"])
    weekly = gp._weekly_all(cfg, season)
    week = int(weekly[weekly.season == season].week.max())
    ffa, _ = gp.ffa_next_week(cfg, season, week)
    f = gp.history_features(weekly, season, week, ffa)
    lv = gp.ridge_predict(gp.load_model(), f)
    f = f.assign(lv=lv)[(f.ytd_games.fillna(0) >= min_games) & np.isfinite(lv)]
    return {pos: sorted(f[f.position == pos].lv.tolist(), reverse=True) for pos in gp.SKILL}


def league_price_curve(cfg, drafts_path=None, levels=None, min_bids=8):
    """{pos: (level grid, dollars)}: what THIS league pays at auction for a player at each level.

    Built by rank, not by raw ppg. Fitting price to last season's ppg compared a raw number with
    the regressed level the valuation plugs in, and averaged in the players the room rightly
    expected to fade (Keenan Allen at WR3 for $11, an injured Chris Godwin at WR2 for $5): it priced
    CeeDee Lamb, a $65 receiver both years, at $36. So: the price for the k-th best player is what
    the room pays for its k-th most expensive one (rank_prices), and a level is turned into a rank
    against this season's fitted levels at the position (current_levels). Age and injury are left
    to the valuation's own drift, where they belong.
    The rank prices this produces, per position, are at league_price_curve.rank_points.
    """
    rp = rank_prices(league_bids(cfg, drafts_path), min_bids)
    league_price_curve.rank_points = rp
    if not rp:
        return {}
    levels = levels or current_levels(cfg)
    out = {}
    for pos in rp:
        L = levels.get(pos) or []
        if len(L) < 2:
            continue
        xs = np.array(L[::-1], float)                            # ascending level
        ranks = np.arange(len(L), 0, -1, dtype=float)            # ... is descending rank
        ys = np.maximum.accumulate(np.array([price_at_rank(rp, pos, r) for r in ranks]))
        out[pos] = (xs, ys)
    return out


league_price_curve.rank_points = {}


def price_premium(cfg, drafts_path=None):
    """{(name, pos): factor}: what the room paid for a player this season over what his preseason
    rank implied, square-rooted (half of it is assumed to persist), between 1x and 2x.

    CeeDee Lamb went for $65 as the 10th-ranked receiver on projection, where that rank's price is
    about $37: this room pays a premium for him, and it has twice. Only players bought at this
    season's auction at a rank worth PREMIUM_MIN_BASE or more carry one; everyone else is 1.0.
    """
    bids = league_bids(cfg, drafts_path)
    rp = rank_prices(bids)
    season = int(cfg.raw["season"])
    out = {}
    for pos, rank, bid, s, name in bids:
        if s != season or pos not in rp:
            continue
        base = price_at_rank(rp, pos, rank)
        if base >= PREMIUM_MIN_BASE:
            out[(_norm(name), pos)] = float(np.clip((bid / base) ** PREMIUM_SHRINK, *PREMIUM_CAP))
    return out


_PRICE_CACHE = {}


def price_caps(cfg, drafts_path=None):
    """{pos: the most this room has ever paid at the position}. A premium times a good outcome can
    price a player past anything the league has bid (Ashton Jeanty x1.66 at RB1 came to $105, where the
    record is $68); nothing is priced above the record."""
    caps = {}
    for pos, _, bid, _, _ in league_bids(cfg, drafts_path):
        caps[pos] = max(caps.get(pos, 0.0), bid)
    return caps


def price_curve(cfg):
    """The curve keeper value is priced on: this league's own auction prices (league_price_curve),
    with the model's dollar_curve only for a position that has too few bids. Used by the trade
    calculator (next_season_board, gm_report) and the League Values tab alike, so a keeper is worth
    the same on both. Cached per scoring config: the fit reads three seasons of nflverse."""
    key = json.dumps(cfg.raw.get("scoring"), sort_keys=True)
    if key not in _PRICE_CACHE:
        _PRICE_CACHE[key] = ({**dollar_curve(), **league_price_curve(cfg)}, price_premium(cfg), price_caps(cfg))
    return _PRICE_CACHE[key][0]


def premiums(cfg):
    """price_premium for the same config price_curve was built for (cached alongside it)."""
    price_curve(cfg)
    return _PRICE_CACHE[json.dumps(cfg.raw.get("scoring"), sort_keys=True)][1]


def caps(cfg):
    """price_caps for the same config (cached alongside price_curve)."""
    price_curve(cfg)
    return _PRICE_CACHE[json.dumps(cfg.raw.get("scoring"), sort_keys=True)][2]


def dollars_at(curve, pos, ppg):
    if pos not in curve:
        return 1.0
    xs, ys = curve[pos]
    return float(np.interp(ppg, xs, ys, left=1.0, right=ys[-1]))


def expected_dollars(curve, pos, ppg, age=None, mult=1.0, cap=None):
    """Expected auction value next season for a player at `ppg` now: the dollar curve averaged over
    the measured year-over-year distribution of his level. Convexity is what makes a $1 dart worth
    more than the dollars at his expected level."""
    drift, sd = drift_vs_field(pos, age)            # prices are by rank, so the field ages too
    grid = ppg + drift + sd * np.linspace(-2.5, 2.5, 41)
    w = np.exp(-0.5 * np.linspace(-2.5, 2.5, 41) ** 2)
    d = mult * np.array([dollars_at(curve, pos, g) for g in grid])
    if cap is not None:
        d = np.minimum(d, cap)                      # never above what the room has ever paid
    return float(np.sum(w * d) / w.sum())


def expected_bump(cfg, p_playoffs):
    """The owner's inflation bump next year, in expectation over whether he makes the playoffs.

    The league pays +$5 on a keeper's price for a playoff team and +$3 for one that misses (+$0
    for the non-playoff owner who drafted the champion, which is not predictable and is ignored).
    A team at 35% to make it expects about $3.70, and that is what a $1 stash costs him to keep.
    """
    inf = cfg.raw.get("keepers", {}).get("inflation") or {}
    base = float(inf.get("base", NEXT_BUMP))
    missed = base + float(next((c["delta"] for c in inf.get("conditions", [])
                                if c.get("name") == "missed_playoffs"), -2))
    return base * p_playoffs + missed * (1.0 - p_playoffs)


def next_season_board(cfg, est, curve=None, players=None, bump=NEXT_BUMP):
    """A keeper board for NEXT season built from the current rosters and the fitted player means.

    keepers_2026.js was the board for keeping INTO this season -- costs from the 2025 draft, values
    from the 2026 preseason -- and that decision is made. What a trade changes is next year's:
      cost   = this year's basis (the auction price; $1 for any waiver pickup, per league rule)
               + the owner's bump (assumed NEXT_BUMP until the standings settle)
      value  = expected auction dollars at the player's healthy level (the fitted rest-of-season
               mean, which does not know he is hurt) after a measured year of drift and spread,
               priced on what this league pays at that position (price_curve: its own bids)
    Same shape as load_board() so the trade search reads it unchanged. Injured stashes are the
    case this exists for: their weekly value is near zero, their keeper value is not.
    `bump` is a number for every team or {team_id: bump} (see expected_bump).
    """
    prem = premiums(cfg) if curve is None else {}   # a player's own premium belongs to the league's prices
    cap = caps(cfg) if curve is None else {}
    curve = curve or price_curve(cfg)          # what this league pays (league bids), not the model
    players = players or draft_tool_players()
    out = {}
    for t in cfg.teams:
        tid = str(t["team_id"])
        b = float(bump.get(tid, NEXT_BUMP)) if isinstance(bump, dict) else float(bump)
        out[tid] = {"bump": b, "players": {}}
        for (tt, pid), v in est.items():
            if tt != tid or v["pos"] not in ("QB", "RB", "WR", "TE"):
                continue
            waiver = str(v.get("acquired") or "").upper() == "ADD"
            basis = 1.0 if waiver else float(v.get("keeper_price") or 0.0)
            p = players.get((_norm(v["name"]), v["pos"]))
            age = (p.get("age") + 1) if p and p.get("age") else None
            healthy = float(v.get("raw_mean", v["mean"]))
            value = expected_dollars(curve, v["pos"], healthy, age, prem.get((_norm(v["name"]), v["pos"]), 1.0),
                                     cap.get(v["pos"]))
            cost = basis + b
            out[tid]["players"][v["name"]] = {
                "name": v["name"], "pos": v["pos"], "value": round(value, 1), "basis": basis,
                "waiver": waiver, "cost": cost, "surplus": round(value - cost, 1),
                "healthy_ppg": round(healthy, 1), "age": age, "injury": v.get("injury"),
                "times_kept": v.get("times_kept", 0)}
    return out
