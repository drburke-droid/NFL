# Alpha skill: why the Elo buried Ja'Marr Chase, and the fix that validates

**Question.** The Player Elo (opponent+teammate-adjusted EPA/play, Kalman-smoothed)
ranks 2025 Ja'Marr Chase **WR25** — behind Alec Pierce and Devaughn Vele. Is the
metric wrong, and what modification bears fruit?

## Diagnosis

Chase 2025: **185 targets** (most in the WR sample) at **+0.224 EPA/target** —
exactly the plays-weighted WR field average (+0.226) — with **Jake Browning /
Joe Flacco throwing for 10 of his 16 games** (Burrow out weeks 2–12). Two
candidate distortions:

1. **No QB term** — the ridge adjusts for opponent (WR1-split) but not passer.
2. **No volume credit** — EPA/target is pure per-play efficiency; commanding
   targets counts for nothing, so 50-target boutique deep threats outrank alphas.

## Mod 1: passer term in the ridge — FAILS validation

`scripts/skill_elo_qb.py` adds primary-passer dummies to the WR/TE ridge
(epa = player + defense@rank + passer). Out-of-time corr with next season's raw
skill z **drops** for WR (0.249 → 0.216) and TE (0.308 → 0.296), and Chase gets
*worse* (2022 obs −0.68): within a team-season the WR1 and his QB are nearly
collinear, so the QB dummy steals the star receiver's credit in the good years.
Intuitive fix, empirically wrong. Kept as a negative result
(table `player_skill_elo_qb`).

## Mod 2: blend earned volume into the skill — WORKS

Earning targets IS receiver skill (getting open, coverage gravity), and it isn't
diluted efficiency: within player-season, weekly targets vs EPA/target r = −0.04.
`scripts/skill_volume_blend.py` Kalman-smooths vol_z (plays/game, era-z) with the
same validated machinery, then blends causal estimates:
`alpha_skill = w·elo + (1−w)·vol`, with w chosen per position by partial corr
with **next-season PPG controlling prior PPG + age** (the forward-value criterion).

| Pos | best w (eff) | partial r, pure Elo | partial r, blend |
|---|---|---|---|
| QB | 0.7 | +0.167 | **+0.186** |
| RB | 0.7 | +0.110 | **+0.136** |
| WR | 0.5 | +0.102 | **+0.249** |
| TE | 0.5 | +0.124 | **+0.263** |

For WR the 50/50 blend carries **2.4× the forward signal** of pure efficiency.

## 2025 WR top 10 under alpha_skill (table `player_skill_alpha`)

| # | Player | alpha | eff (Elo) | vol | Elo rank |
|---|---|---|---|---|---|
| 1 | Puka Nacua | +1.37 | +1.07 | +1.67 | 1 |
| 2 | **Ja'Marr Chase** | **+1.14** | +0.18 | +2.10 | 25 |
| 3 | Amon-Ra St. Brown | +1.10 | +0.69 | +1.51 | 2 |
| 4 | Jaxon Smith-Njigba | +0.86 | +0.56 | +1.16 | 4 |
| 5 | Malik Nabers | +0.77 | −0.14 | +1.67 | 76 |
| 6 | CeeDee Lamb | +0.75 | +0.10 | +1.40 | 35 |
| 7 | George Pickens | +0.61 | +0.59 | +0.64 | 3 |
| 8 | Rashee Rice | +0.60 | +0.07 | +1.13 | 41 |
| 9 | Chris Olave | +0.58 | −0.07 | +1.23 | 68 |
| 10 | A.J. Brown | +0.57 | +0.35 | +0.80 | 11 |

Chase's alpha has *risen* every year (0.55 → 0.89 → 0.81 → 1.00 → 1.14): the
efficiency dip with backup QBs is more than offset by a causal volume estimate
(+2.10) that is the highest in football.

## Caveats

- Volume partially encodes role/opportunity, not just talent — but the partial
  corr already controls prior PPG (which contains volume), so the +0.147 WR
  gain over pure Elo is incremental signal, not double counting.
- The passer-term failure means QB context is still unmodeled; a WR whose QB
  upgrades may be underrated by eff and vol both. Candidate future fix: shrink
  the QB term with a much heavier separate ridge penalty, or use QB effects
  estimated only from *other* receivers (leave-one-out).

**Verdict:** don't adjust for QB; blend in earned volume. `alpha_skill`
(w=0.5 WR/TE, 0.7 QB/RB) is the list the draft board should show.
