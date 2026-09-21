"""A single rolling constant that removes the send's standing over-projection.

Measured by scripts/sabersim_grade.py on the published sends against nflverse, the model runs
high on skill players in both graded weeks of 2026: bias -0.394 in wk1 and -0.674 in wk2. Fitting
that constant on week 1 alone and applying it to week 2 -- out of sample, against the Proj column
of the actual send CSVs -- moved MAE 3.609 -> 3.469 and bias -0.674 -> -0.304. On the 323
DK-priced rows it moved 4.179 -> 4.044, against DK's own 4.035.

Read that as modest and provisional. It is one train/test split on two weeks; the direction is
consistent but the magnitude is not (-0.394 vs -0.674 is a 70% swing), so the correction will
usually under- or over-shoot. 80% coverage moves 0.801 -> 0.792 against an 0.80 target when the
quantile band shifts with the mean, which is a small cost. --no-bias-cal turns it off.

Three deliberate choices:

* ONE global constant, not per position. The pooled bias is at least consistently negative; its
  composition is not -- RB went -0.047 -> -1.288 between the two weeks while WR went -0.739 ->
  -0.366. Fitting per position chases last week's variance.
* Skill positions only. K and DST come from a different source (the Subvertadown/kdst blend) and
  carry their own bias; this is measured on, and applies to, QB/RB/WR/TE.
* It does not reorder anybody. Spearman is 0.7712 before and after. This fixes how much a
  projection says a player is worth, not who it prefers -- which is the point, since an inflated
  projection misleads a lineup optimiser about a lineup's absolute value.

The number comes from docs/sabersim_accuracy.json, which the grade workflow rewrites after every
slate, so it re-fits itself each week with no extra pipeline.
"""
SKILL = ("QB", "RB", "WR", "TE")
WEEKS = 4      # rolling window of completed weeks
MIN_ROWS = 300  # below this the estimate is noise; correct nothing
CAP = 2.0      # a single bad week can never move a send by more than this


def recent_bias(acc, season=None, weeks=WEEKS, min_rows=MIN_ROWS, cap=CAP, positions=SKILL):
    """Mean (actual - Proj) over the most recent completed weeks, skill positions only.

    Returns (bias, meta). bias is what to ADD to a projection, so it is negative when the model
    runs high. Returns 0.0 whenever there is not enough graded history to be worth acting on --
    the caller does not need to special-case a cold start.
    """
    meta = {"applied": 0.0, "rows": 0, "weeks": [], "reason": ""}
    if not isinstance(acc, dict):
        meta["reason"] = "no accuracy file"
        return 0.0, meta
    if season is not None and acc.get("season") not in (None, season):
        meta["reason"] = f"accuracy file is season {acc.get('season')}, not {season}"
        return 0.0, meta

    blocks = [w for w in acc.get("weeks", []) if isinstance(w, dict) and "week" in w]
    blocks = sorted(blocks, key=lambda w: w["week"])[-weeks:]
    num = den = 0.0
    used = []
    for w in blocks:
        by_pos = w.get("by_pos") or {}
        n_wk = 0
        for p in positions:
            b = by_pos.get(p)
            if not isinstance(b, dict):
                continue
            n, bias = b.get("n"), b.get("bias")
            if not n or bias is None:
                continue
            num += float(n) * float(bias)
            den += float(n)
            n_wk += int(n)
        if n_wk:
            used.append({"week": int(w["week"]), "n": n_wk})

    meta["weeks"], meta["rows"] = used, int(den)
    if den < min_rows:
        meta["reason"] = f"only {int(den)} graded rows, need {min_rows}"
        return 0.0, meta

    raw = num / den
    bias = max(-cap, min(cap, raw))
    meta["raw"] = round(raw, 4)
    if bias != raw:
        meta["reason"] = f"capped from {raw:+.3f} to {bias:+.3f}"
    meta["applied"] = round(bias, 4)
    return bias, meta
