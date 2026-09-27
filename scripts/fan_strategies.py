"""Strategy lab: ways of combining the fans' calls, fixed in advance and scored every week.

Week 3 of 2026 (Thursday plus the 1pm games) suggested a few ideas: blend the crowd into one vote per player,
trust fades more than boosts, fade the mid-tier players the crowd boosts, and follow a few fans on the
positions they read well. One week is enough to suggest a rule and nowhere near enough to trust one --
the specialist rule in particular was picked after seeing the results. So the rules are REGISTERED here, on
2026-09-27, before any later week is played, and every week from REGISTERED_FROM on is an honest test.
Week 3 is shown, flagged in-sample, and kept out of the out-of-sample totals.

Every rule is replayed as whole-line swipes on the line we actually SENT (the same basis fan_grade scores
fans on) and scored the same way: fantasy points of error removed against the box score. + means the rule
would have beaten THE ORACLE on those players. "Swipe size" is not a replay: it reports the calls as the
fans made them, split by how far they pulled, to see whether conviction turns into accuracy.
"""
import numpy as np
import pandas as pd

REGISTERED = "2026-09-27"
IN_SAMPLE_WEEKS = {3}            # the week the ideas came from
PTS = {"pass_yds": 0.04, "pass_tds": 4.0, "pass_int": 1.0, "rush_yds": 0.1, "rush_tds": 6.0, "rec": 1.0, "rec_yds": 0.1, "rec_tds": 6.0}
SPECIALISTS = [("REB", {"WR", "TE"}), ("Nana Owusu", {"RB"}), ("RJS", {"RB"})]   # by the name the grade shows

RULES = [  # key, label, one line on what it does
    ("crowd", "Crowd majority, one vote per player", "every player anyone called, moved 10% the way most callers went"),
    ("fades", "Fades only", "every player any fan faded, faded 10%; boosts ignored"),
    ("fade_mid_boosts", "Fade mid-tier players fans boosted", "every player any fan boosted who projects 14-20 pts, faded 10%"),
    ("specialists", "Specialists", "REB on WR/TE, Nana Owusu on RBs, RJS on RBs, their direction at 10%"),
    ("size_1", "Calls at 10%, as made", "every call made with a 10% swipe (or a stat arrow), as the fan made it"),
    ("size_2", "Calls at 20-30%, as made", "every call made with a 20% or 30% swipe, as the fan made it"),
]


def decisions(g):
    """One row per fan, week and player: the direction of his call, its swipe size, projection, position, and
    the points of error it removed as made. `g` is fan_grade's graded frame; only rows with a box score and a
    send line count."""
    d = g[~g.pending & ~g.no_send].copy()
    if d.empty:
        return pd.DataFrame(columns=["fan", "week", "player", "pos", "dirn", "k", "proj", "pts"])
    d["sgn"] = np.sign(d.arrows)
    out = d.groupby(["fan", "week", "player"], as_index=False).agg(
        pos=("pos", "first"), net=("sgn", "sum"), k=("arrows", lambda s: int(abs(s).max()) if (s.abs() <= 3).all() else 1),
        rule=("rule", "first"), proj=("proj_pts", "first"), pts=("removed_pts", "sum"))
    out["dirn"] = np.sign(out.net).astype(int)
    out.loc[out.rule != "s1", "k"] = 1                  # a stat arrow is a precise call, not a pull: it counts with the 10% swipes
    return out[out.dirn != 0]


def _replay(g, calls):
    """Error removed (DK pts) by moving each (week, player)'s SENT line 10% in the given direction."""
    d = g[~g.pending & ~g.no_send]
    if d.empty or not calls:
        return 0.0, 0, 0
    ps = d.groupby(["week", "player", "stat"], as_index=False).agg(base=("base", "first"), actual=("actual", "first"))
    tot, n, helped = 0.0, 0, 0
    for (wk, pl), dirn in calls.items():
        x = ps[(ps.week == wk) & (ps.player == pl)]
        if x.empty or not dirn:
            continue
        adj = (x.base * (1 + 0.1 * dirn)).clip(lower=0)
        v = float((((x.actual - x.base).abs() - (x.actual - adj).abs()) * x.stat.map(PTS)).sum())
        tot += v; n += 1; helped += v > 0
    return tot, n, helped


def rule_calls(dec, key):
    """{(week, player): direction} for a replayed rule."""
    if key == "crowd":
        v = dec.groupby(["week", "player"]).dirn.sum()
        return {k: int(np.sign(x)) for k, x in v.items() if x}
    if key == "fades":
        return {k: -1 for k in dec[dec.dirn < 0].set_index(["week", "player"]).index}
    if key == "fade_mid_boosts":
        b = dec[(dec.dirn > 0) & (dec.proj >= 14) & (dec.proj < 20)]   # any fan's boost (the week-3 reading); a majority-only
        return {k: -1 for k in b.set_index(["week", "player"]).index}      # version read -2.2 there -- one reason these are only ideas
    if key == "specialists":
        out = {}
        for fan, pos in SPECIALISTS:
            for r in dec[(dec.fan == fan) & dec.pos.isin(pos)].itertuples():
                out[(r.week, r.player)] = r.dirn
        return out
    raise KeyError(key)


def lab(g):
    """The strategy table for grade.json: each rule's calls, share that helped and points, per week, over the
    out-of-sample weeks, and over everything."""
    dec = decisions(g)
    weeks = sorted(int(w) for w in dec.week.unique()) if len(dec) else []
    rows = []
    for key, label, how in RULES:
        per = {}
        for w in weeks:
            dw = dec[dec.week == w]
            if key.startswith("size_"):
                sel = dw[dw.k == 1] if key == "size_1" else dw[dw.k >= 2]
                tot, n, helped = float(sel.pts.sum()), int(len(sel)), int((sel.pts > 0).sum())
            else:
                tot, n, helped = _replay(g, rule_calls(dw, key))
            per[w] = {"calls": n, "helped": helped, "pts": round(tot, 2)}
        def total(ws):
            c = sum(per[w]["calls"] for w in ws); h = sum(per[w]["helped"] for w in ws); p = sum(per[w]["pts"] for w in ws)
            return {"calls": c, "helped": h, "pts": round(p, 2), "weeks": list(ws)}
        oos = [w for w in weeks if w not in IN_SAMPLE_WEEKS]
        rows.append({"key": key, "label": label, "how": how, "weeks": {str(w): v for w, v in per.items()},
                     "out_of_sample": total(oos), "all": total(weeks)})
    return {"registered": REGISTERED, "in_sample_weeks": sorted(IN_SAMPLE_WEEKS),
            "note": "Rules fixed on " + REGISTERED + " from week 3; only later weeks test them. Replays are 10% swipes on the sent line.",
            "rules": rows}
