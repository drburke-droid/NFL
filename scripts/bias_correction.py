"""A single rolling constant that keeps the send's Proj centred on players who play.

What it reads (2026-09-24 onward): `med_err_played` per position from docs/sabersim_accuracy.json,
the median of (actual - Median column) over players who appeared in the box score, weighted by
`n_played`. Proj is the median of the player's playing distribution, MAE is minimised by the
median, and the scoreboards we are ranked on are computed over players who played -- so the
quantity that should be zero is exactly this one. Blocks from grades made before the grader emitted
it are skipped, which means the correction is silent until the first new grade lands rather than
falling back to a number that measures something else.

What it used to read, and why that was wrong: the first version (2026-09-17) took the all-rows
mean error, inactives scored 0 included. In 2026 wk1 that was -0.55 while the played-only mean was
+0.11 -- the whole number came from 116 inactives we had left at 2-3 points. It happened to help
MAE anyway, because the send then carried the MEAN point and any downward shift moved it toward
the median; that accident is now made deliberate in the generator (Proj = playing median) and this
module goes back to being a small safety net for in-season drift.

Three deliberate choices, unchanged:

* ONE global constant, not per position. The pooled figure is consistent; its composition is not
  (RB -0.047 -> -1.288 between wk1 and wk2 in the old measurement). Fitting per position chases
  last week's variance.
* Skill positions only. K and DST come from a different source (the Subvertadown/kdst blend) and
  carry their own bias; this is measured on, and applies to, QB/RB/WR/TE.
* It does not reorder anybody. It is added to the playing mean before the quantiles are cut, so
  Proj, Median, Mean and the band move together.

The number comes from docs/sabersim_accuracy.json, which the grade workflow rewrites after every
slate, so it re-fits itself each week with no extra pipeline. --no-bias-cal turns it off.
"""
SKILL = ("QB", "RB", "WR", "TE")
WEEKS = 4      # rolling window of completed weeks
MIN_ROWS = 300  # below this the estimate is noise; correct nothing
CAP = 2.0      # a single bad week can never move a send by more than this
KEY, NKEY = "med_err_played", "n_played"   # what the grader emits; older grades lack both


def recent_bias(acc, season=None, weeks=WEEKS, min_rows=MIN_ROWS, cap=CAP, positions=SKILL,
                key=KEY, nkey=NKEY):
    """Row-weighted `key` over the most recent completed weeks, skill positions only.

    Returns (bias, meta). bias is what to ADD to a projection, so it is negative when the send
    runs high. Returns 0.0 whenever there is not enough graded history to be worth acting on --
    the caller does not need to special-case a cold start.
    """
    meta = {"applied": 0.0, "rows": 0, "weeks": [], "reason": "", "basis": key}
    if not isinstance(acc, dict):
        meta["reason"] = "no accuracy file"
        return 0.0, meta
    if season is not None and acc.get("season") not in (None, season):
        meta["reason"] = f"accuracy file is season {acc.get('season')}, not {season}"
        return 0.0, meta

    blocks = [w for w in acc.get("weeks", []) if isinstance(w, dict) and "week" in w]
    blocks = sorted(blocks, key=lambda w: w["week"])[-weeks:]
    num = den = 0.0
    used, skipped = [], []
    for w in blocks:
        by_pos = w.get("by_pos") or {}
        n_wk = 0
        for p in positions:
            b = by_pos.get(p)
            if not isinstance(b, dict):
                continue
            n, err = b.get(nkey), b.get(key)
            if not n or err is None:
                continue
            num += float(n) * float(err)
            den += float(n)
            n_wk += int(n)
        if n_wk:
            used.append({"week": int(w["week"]), "n": n_wk})
        else:
            skipped.append(int(w["week"]))

    meta["weeks"], meta["rows"] = used, int(den)
    if skipped:
        meta["skipped_weeks"] = skipped
    if den < min_rows:
        meta["reason"] = f"only {int(den)} graded played rows with {key}, need {min_rows}"
        if skipped and not used:
            meta["reason"] += f" (weeks {skipped} were graded before the grader emitted it)"
        return 0.0, meta

    raw = num / den
    bias = max(-cap, min(cap, raw))
    meta["raw"] = round(raw, 4)
    if bias != raw:
        meta["reason"] = f"capped from {raw:+.3f} to {bias:+.3f}"
    meta["applied"] = round(bias, 4)
    return bias, meta
