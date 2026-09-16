# Fan Picks (recorded, not used)

`docs/fan.html` shows the week's projected stat lines (baked by `scripts/fan_proj_bake.py` from a
whole-week generator run) and lets a fan boost (▲) or fade (▼) any stat, 10% of the baseline per press, unlimited presses (▼ x10 = zero). Submit produces a
code `FAN1.<season>.<week>.<base64>` the fan sends back by text / DM / email.

Weekly:
1. `python scripts/sabersim_weekly.py <pkg> --all-games --no-market --out outputs/fan/proj_<S>_wk<W>.csv`
2. `python scripts/fan_proj_bake.py outputs/fan/proj_<S>_wk<W>.csv <S> <W>` -> docs/fan/proj_latest.json (+ frozen copy); push
3. Drop received codes in `inbox/<anything>.txt` (one per line) and run `python scripts/fan_adjust_record.py --inbox`
   -> `fan_adjustments_long.csv` (received_at, season, week, fan, submitted_at, comment, player, stat, arrows, pct, baseline, adjusted, proj_pts)

The page shows the fan a 10%-per-press adjustment so the signal is concrete, but nothing downstream
uses it. Like the Subvertadown matchup bonuses, the arrows are collected first and graded against
actuals before anything decides whether a fan's +20% is worth +20%, +5% or nothing.
