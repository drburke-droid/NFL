# Breakout & Rookie-Impact Signals

Two questions: (1) which signals flag veterans who **significantly outperform** their prior
year, and (2) which flag **rookies who produce out of the gate**. Both validated
walk-forward, 2011–2025.

---

## 1. Veteran breakouts — "significantly outperform last year"
**Breakout = next-season PPG ≥ +4 above prior season** (~68-pt jump), players with a real
prior role (≥4 games). **Base rate 11.7%** (QB 16%, RB 14%, WR 12%, TE 7%).

**Walk-forward classifier:** ROC AUC **0.682**; its top-20% of flagged players hit at
**22.3% — 2.0× the base rate**, capturing 40% of all breakouts. So you can't *nail*
breakouts, but you can **roughly double your hit rate** by screening for the right profile.

**The breakout profile (rate by signal quartile):**
| Signal | Low → High breakout rate | Direction |
|---|---|---|
| **Age** | 15% → 8% | **younger** breaks out |
| **Prior PPG** | 14% → 8% | **lower base** (room to grow) |
| **Prior snap %** | 16% → 9% | **buried** (role can expand) |
| **Prior volatility (CV)** | 9% → 14% | **boom/bust** players spike |
| **FFA ceiling headroom** | 11% → 16% | consensus already sees upside |

Also discriminating: **low prior TD rate** (positive TD regression coming), **draft pedigree**
(former high picks), and prior efficiency flashes (receiving EPA, air-yards share).

**The actionable archetype:** a **young (year 2–3), former-pedigree player in a limited but
expanding role** — low snaps/targets last year but efficient when used, modest prior scoring,
and TD-regression upside. That's the breakout recipe. (Top model signals: prior_ppg,
prior_ppr_std, prior_cv, prior2_ppg, draft_pick, receiving_epa, snap %.)

> Honest read: breakouts are genuinely hard — the ~2× lift is the realistic ceiling, because
> the market (FFA/ADP) already prices much of this in. The edge is in the *tails* of the profile.

---

## 2. Rookies "out of the gate" — much more predictable
**Hit = rookie-year PPG at a startable level** (QB 14 / RB 10 / WR 9 / TE 7). **Base rate 15%.**

**Walk-forward classifier:** ROC AUC **0.789** — *far* stronger than veteran breakouts. Its
top-20% hit at **44.8% — 3.0× base**, capturing 60% of all rookie hits. **Rookie immediate
impact is one of the more predictable things in fantasy.**

**Draft capital dominates** — immediate-hit rate by draft slot:
| Pos | R1 top-15 | R1–2 | R3 | R4–5 | R6–UDFA |
|---|---|---|---|---|---|
| **RB** | **90%** | 84% | 40% | 12% | 5% |
| **WR** | 75% | 44% | 32% | 13% | 5% |
| **TE** | 67% | 43% | 12% | 11% | 3% |

**Top discriminating signals:** draft pick / round (ρ ≈ −0.39, dominant), then **athleticism**
— 40-yard dash (ρ −0.23), 3-cone, broad jump, vertical (explosive testers hit) — and **young
draft age** (Q1 23% vs Q4 9%). Landing spot (prior-year vacancy / pass volume) is a *minor*
add. Notably hits skew to **worse prior-year offenses** (opportunity/vacancy).

**The actionable archetype:** an **early-round (R1–2), young, explosive athlete** walking into
playing time. Draft capital + athletic profile + age >> situation. (Top model signals:
draft_pick, weight, team pass volume, team offense, landing-spot opportunity, height.)

---

## Bottom line
- **Rookies are ~1.5× more predictable than veteran breakouts** (AUC 0.79 vs 0.68; 3× vs 2× lift) — lean on **draft capital + athleticism + youth** and you'll be right far more often.
- **Veteran breakouts** reward a specific screen: **young, buried, pedigreed, TD-regression-positive, high-variance** — it doubles your hit rate but won't make it a coin flip.

*Scripts: `breakout_signals.py`, `rookie_signals.py`. Both reuse `season_dataset`/`nflv_ffa_proj` and the rookie dataset.*
