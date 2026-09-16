# Fan Picks (recorded, not used)

`docs/fan.html` shows the week's projected stat lines (baked by `scripts/fan_proj_bake.py` from a
whole-week generator run) and lets a fan boost (▲ x1-3) or fade (▼ x1-3) any stat. Submit produces a
code `FAN1.<season>.<week>.<base64>` the fan sends back by text / DM / email.

Weekly:
1. `python scripts/sabersim_weekly.py <pkg> --all-games --no-market --out outputs/fan/proj_<S>_wk<W>.csv`
2. `python scripts/fan_proj_bake.py outputs/fan/proj_<S>_wk<W>.csv <S> <W>` -> docs/fan/proj_latest.json (+ frozen copy); push
3. Drop received codes in `inbox/<anything>.txt` (one per line) and run `python scripts/fan_adjust_record.py --inbox`
   -> `fan_adjustments_long.csv` (received_at, season, week, fan, submitted_at, comment, player, stat, arrows, baseline, proj_pts)

The arrows carry no numeric meaning yet. Like the Subvertadown matchup bonuses, they are collected
first and graded against actuals before anything decides whether ▲▲ is worth +10% or nothing.
