# Year-to-year skill model: does play-by-play skill signal beat the market?

Rows: 5,438 player-seasons (2012-2025 priors, targets 2016-2025 walk-forward).
Skill-feature coverage: 67% of rows; NGS coverage 19% (2017+ targets only).

## Central projection MAE (PPG), walk-forward 2016-2025

| Pos | A base | B +skill | C +skill+ngs | D market(FFA) | E market+skill+ngs | best |
|---|---|---|---|---|---|---|
| QB | 3.851 | 3.869 | 3.858 | 3.700 | 3.672 | E market+skill+ngs |
| RB | 3.014 | 2.996 | 2.995 | 2.746 | 2.749 | D market(FFA) |
| WR | 2.544 | 2.545 | 2.560 | 2.468 | 2.482 | D market(FFA) |
| TE | 1.935 | 1.920 | 1.925 | 1.817 | 1.838 | D market(FFA) |

Spearman rho (E vs D):
  QB: E 0.717 vs D 0.719
  RB: E 0.769 vs D 0.768
  WR: E 0.787 vs D 0.791
  TE: E 0.749 vs D 0.750

## Injury-return subgroup (2+ weeks OUT on injury report in prior season)

| Pos | n | A base MAE | B +skill MAE | Δ |
|---|---|---|---|---|
| QB | 43 | 3.551 | 3.769 | +0.217 |
| RB | 95 | 3.462 | 3.317 | -0.144 |
| WR | 188 | 3.279 | 3.305 | +0.026 |
| TE | 77 | 2.470 | 2.460 | -0.010 |

## New-feature importance (config E, full fit)

| Feature | importance | rank among all |
|---|---|---|
| skill_2yr_avg | 151 | 7/69 |
| p_ppg_z | 135 | 10/69 |
| p_skill_comp | 115 | 15/69 |
| p_d_vol | 108 | 17/69 |
| p_skill_trend2 | 103 | 22/69 |
| p_inseason_trend | 101 | 26/69 |
| p2_skill_comp | 93 | 32/69 |
| p_vol_z | 83 | 36/69 |
| p_clean_games | 80 | 37/69 |
| p_healthy_gap | 64 | 49/69 |

## 2026 projections (trained on all data, best all-feature config)

Production features: base+inj+skill+ngs (no 2026 FFA yet)

Saved skill_projections_2026: 532 players.

**QB top 10 (central PPG | floor-ceiling | archetype):**
- Josh Allen: 22.3 (18.7-23.2) — Young Solid Steady
- Lamar Jackson: 18.8 (16.5-21.6) — Prime Fringe Injury-Hit
- Bo Nix: 18.7 (17.1-20.6) — Young Solid Steady
- Patrick Mahomes: 18.6 (17.5-21.1) — Young Solid Steady
- Caleb Williams: 18.0 (17.0-20.0) — Young Solid Steady
- Jalen Hurts: 18.0 (16.9-19.8) — Young Solid Steady
- Justin Herbert: 18.0 (16.6-19.7) — Young Solid Steady
- Brock Purdy: 17.6 (9.3-19.7) — Prime Fringe Injury-Hit
- Jaxson Dart: 17.6 (12.8-17.4) — Young Solid Steady
- Drake Maye: 17.5 (14.6-21.0) — Young Solid Steady

**RB top 10 (central PPG | floor-ceiling | archetype):**
- Bijan Robinson: 19.2 (15.5-23.5) — Prime Elite Role-Growing
- Jahmyr Gibbs: 18.1 (13.4-21.8) — Prime Elite Role-Growing
- Christian McCaffrey: 16.8 (12.1-22.6) — Prime Elite Role-Growing
- Jonathan Taylor: 16.8 (13.7-19.8) — Prime Elite Role-Growing
- De'Von Achane: 16.5 (13.0-21.4) — Prime Elite Role-Growing
- James Cook: 14.9 (12.1-18.4) — Prime Elite Role-Growing
- Derrick Henry: 14.9 (11.9-19.6) — Prime Elite Role-Growing
- Josh Jacobs: 14.7 (11.0-16.7) — Prime Solid Role-Shrinking
- D'Andre Swift: 14.2 (9.8-14.3) — Prime Solid Role-Shrinking
- Saquon Barkley: 14.1 (12.1-16.9) — Prime Elite Role-Growing

**WR top 10 (central PPG | floor-ceiling | archetype):**
- Puka Nacua: 19.2 (14.4-22.0) — Prime Elite Role-Growing
- Ja'Marr Chase: 18.9 (15.4-21.0) — Prime Elite Role-Growing
- Amon-Ra St. Brown: 17.7 (15.2-18.5) — Prime Elite Role-Growing
- Jaxon Smith-Njigba: 17.5 (13.0-22.4) — Prime Elite Role-Growing
- George Pickens: 15.8 (12.2-16.7) — Prime Elite Role-Growing
- Nico Collins: 15.4 (12.6-18.4) — Prime Elite Role-Growing
- Rashee Rice: 15.3 (11.8-19.0) — Prime Elite Role-Growing
- Jameson Williams: 15.0 (12.1-16.1) — Prime Elite Role-Growing
- Malik Nabers: 15.0 (10.0-17.6) — Prime Fringe Fading
- A.J. Brown: 14.7 (12.1-15.5) — Prime Elite Role-Growing

**TE top 10 (central PPG | floor-ceiling | archetype):**
- Trey McBride: 14.8 (11.4-16.9) — Prime Elite Role-Growing
- Brock Bowers: 12.6 (10.7-14.0) — Prime Elite Steady
- George Kittle: 12.3 (10.3-13.0) — Prime Elite Role-Growing
- Colston Loveland: 11.2 (7.9-12.3) — Prime Elite Role-Growing
- Travis Kelce: 10.6 (9.8-11.7) — Prime Elite Role-Growing
- Tyler Warren: 10.5 (8.2-11.7) — Prime Elite Role-Growing
- Kyle Pitts: 10.1 (8.2-12.8) — Prime Elite Role-Growing
- Juwan Johnson: 10.1 (8.0-11.9) — Prime Elite Role-Growing
- Harold Fannin Jr.: 10.0 (8.1-11.7) — Prime Elite Role-Growing
- Dallas Goedert: 9.9 (8.7-10.9) — Prime Elite Role-Growing

## Verdict (synthesis across the three studies)

1. **What is skill vs noise** (outputs/reports/skill_trajectories.md): QB completion%,
   sack rate, success rate and aDOT are true skill (YoY r 0.46-0.54). WR/TE aDOT is the
   stickiest metric in football (r 0.61-0.69) but it's deployment, not talent. RB per-play
   efficiency (EPA/rush, YPC, success) is nearly pure noise year-to-year (r 0.15-0.26) —
   never pay for an RB efficiency spike.
2. **Aging is asymmetric**: QBs lose skill while keeping their job (skill fades first from
   24 on); WRs 28+ lose their role before their skill; RBs lose both together mid-20s.
3. **Injuries**: playing while listed Q/D costs -0.04 to -0.10 EPA/play; there is NO
   detectable return-rust after multi-week absences. Injury-report features improve
   projections only for RB injury-returns (MAE -0.14, n=95).
4. **Spike fade is real**: at the same skill level, players who *rose* into the tier give
   back ~1.3 more PPG next season than players who held it despite a dip (QB -2.5 vs -1.2,
   WR -2.0 vs -0.7). "How you got here" matters; the trend2 feature carries this.
5. **Model lift is honest but small**: PBP-skill features beat the internal base slightly
   (RB/TE) and add on top of the FFA market anchor ONLY for QB (3.700 -> 3.672 MAE). For
   RB/WR/TE the market already prices the skill signal. Consistent with every prior study
   in this repo: volume/role >> efficiency for prediction, and the market is hard to beat.
6. **Where the new work earns its keep**: trajectory archetypes (outputs/reports/
   trajectory_archetypes.md) as risk profiles — e.g. RB "Injury-Hit" states lose only
   -0.5 PPG next year vs -2.1 for "Elite Role-Growing" (reversion), and "Fading" states
   carry 45-49% bust rates — plus the QB MAE lift and the RB injury-return correction.
