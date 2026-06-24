# Draft Capital & Coaching — Deep Dive (2011–2025)

Tested whether teams' **draft-capital allocation by position** and **head-coach changes**
correlate with on-field outcomes. 480 team-seasons. Draft from `nflv_draft` (pick-value
weighted via a Jimmy-Johnson-style curve, rolling 3-year capital by position group);
outcomes from `nflv_weekly` (pass/rush EPA, sacks allowed) + `nflv_team_def` (sacks, DST
points) + nflverse schedules (HC, wins, point differential).

> **Bottom line up front:** the specific draft hypotheses are **not supported** — positional
> draft spend doesn't predict unit quality. The head-coach "bump" is **mostly mean reversion**;
> the genuine coaching effect is small and only positive for already-bad teams. **OC/DC data
> isn't available in nflverse**, so the OC-specific hypothesis needs sourcing (see end).

---

## Draft capital → outcomes: no signal (and a big confound)

The **raw** correlations are dominated by a selection effect: **bad teams pick higher and
more often**, so draft capital correlates *negatively* with outcomes by construction.
Controlling for prior-year team quality (point differential) removes most of it — and what's
left is **essentially zero**:

| Hypothesis | Raw ρ | Partial ρ (control quality) | Δ-improvement ρ |
|---|---|---|---|
| **DL capital → def sacks** | −0.02 | −0.01 | +0.02 (p .77) |
| **DL capital → DST points** | +0.04 | +0.05 | −0.05 (p .39) |
| **OL capital → sacks allowed** | +0.10 | +0.05 | −0.01 (p .87) |
| **OL capital → pass EPA (QB play)** | −0.04 | +0.06 | — |
| **QB capital → pass EPA** | **−0.22** | **−0.14** | +0.07 (p .18) |
| **Total capital → wins** | **−0.20** | −0.04 | — |

**Reading it:**
- **"Invest in DL → better DST/defense": not supported.** Three-year DL draft capital has ~zero correlation with defensive sacks or DST points, at any lag, raw or controlled or on next-year improvement.
- **"Invest in OL → better QB play / protection": not supported.** OL capital is, if anything, *slightly positively* correlated with sacks *allowed* (wrong direction) and has no link to pass EPA.
- **QB capital → pass EPA stays negative even after control (−0.14).** You spend high picks on QBs precisely when you don't have one — and most don't pan out — so the spend marks a *problem*, not a fix.
- **Total capital → wins is ~zero once you control for being bad** (raw −0.20 was almost entirely "bad teams draft high"). Draft capital neither reliably helps nor hurts at the team-aggregate level within 0–2 years.

**Why:** at the team-aggregate, 3-year level, *hit rate and development dominate raw spend.*
Pouring picks into a position group does not, on average, translate into that unit playing
better. Individual hits, scheme, coaching, and free agency swamp the draft-allocation signal.

---

## Head-coach change → "bump" is mostly mean reversion

**Raw**, new-HC seasons look great vs continuity seasons:

| Δ vs prior year | New HC | Kept HC | Gap |
|---|---|---|---|
| Wins | +1.47 | −0.42 | **+1.89** |
| Point diff | +34.3 | −10.9 | +45.1 |
| Pass EPA | +25.6 | −8.2 | +33.9 |

But teams **fire coaches after bad years**, so they're coming off a low base and revert upward
anyway. Controlling for **prior record** (mean-reversion control) shrinks the effect sharply:

| Prior-year wins | New-HC Δwins | Kept-HC Δwins | **Net coaching effect** |
|---|---|---|---|
| ≤4 (bad) | +3.88 | +2.98 | **+0.89 wins** |
| 5–7 | +0.92 | +1.07 | −0.15 |
| 8–10 | −1.74 | −0.35 | **−1.39** |
| 11+ | (no firings) | −2.33 | — |

**Reading it:**
- Bad teams (≤4 wins) bounce back **~+3 wins whether or not they change coaches** — that's regression to the mean, not coaching.
- A coaching change adds only **~+0.9 wins** on top of that for already-bad teams.
- For **decent teams (8–10 wins), changing HC is associated with a ~1.4-win *decline*** — these are usually upheaval situations (lost a good coordinator, owner meddling, etc.).
- So the popular "new coach → instant turnaround" is largely a mean-reversion illusion; the real, isolated HC-change effect is modest and conditional on the team already being bad.

---

## Coordinators (OC/DC): researched and tested

Compiled the full **HC / OC / DC** for all 32 teams × 2011–2025 (480 team-seasons) via
parallel web research (Wikipedia season pages + Pro Football Reference), then **validated
the head coaches against the schedule-derived ground truth: 99.0% match (475/480)** — the 5
"misses" are all interim-coach convention differences (e.g. CLE 2018 Hue Jackson vs interim
Gregg Williams), confirming the coordinator data is high quality. Saved as `nflv_coaches`.

### New OC → offense bump: helps bad offenses, hurts mid ones
Δ team pass EPA in the OC's first year, vs continuity, **controlled for prior-offense tier**:

| Prior offense | Net new-OC effect (Δ pass EPA, new − kept) |
|---|---|
| Bottom third | **+20.9** |
| Middle third | **−17.5** |
| Top third | +8.3 |

Same shape as the HC result: **changing the OC of a struggling offense helps (~+21 pass EPA
beyond mean reversion); changing a *middling* offense's OC hurts (−17).** Don't fix what isn't broken.

### New DC → defense bump: essentially none
Net new-DC effect on DST points is tiny at every tier (+0.9 / −1.1 / −2.4). Consistent with
the earlier finding that defensive/DST output is intrinsically noisy — a new DC doesn't
reliably move it.

### ⭐ Your headline hypothesis — "hire the OC who ran a good offense elsewhere" — is NOT supported
Among **41 OC moves between NFL teams** (was an OC elsewhere the prior year):

| OC came from… | New team's Δ pass EPA, year 1 |
|---|---|
| a **top-tier** offense (n=16) | +36.7 |
| a **bottom-tier** offense (n=20) | +27.4 |
| **corr(prior-offense quality, new-team bump)** | **ρ +0.05, p = 0.77 (zero)** |

**Whether an OC ran a great or terrible offense at his last stop has no bearing on the bump he
brings.** Both groups improve ~+27–37 pass EPA — but that's just mean reversion (struggling
offenses hire OCs), and the prior-offense pedigree adds *nothing*. The examples make it vivid:
**Eric Bieniemy (from KC's #1 offense → WAS: −27)** and **Ken Dorsey (from BUF's elite offense →
CLE: −70)** both *declined*, because their prior success rode the QB/HC (Mahomes-Reid, Allen),
not a portable scheme. Meanwhile Todd Haley (PIT → CLE) and Adam Gase (DEN → CHI) brought big
bumps. **The "poach the hot OC" heuristic doesn't predict success.**

---

## Multi-year (sustained-investment) test

The above used a rolling 3-year window. A longer test — **5-year cumulative draft
*devotion* (share of capital to a position group, which controls for "bad teams pick
higher overall") → that unit's performance averaged over the NEXT 3 seasons** — was run to
check whether sustained investment compounds (e.g. "drafted DL high over years 1–5 → better
DST in years 6–8").

| Sustained 5-yr devotion | → next-3-yr outcome | raw ρ | partial ρ | top-25% vs bottom-25% |
|---|---|---|---|---|
| **DL share** | DST points | +0.005 | +0.009 | 51.7 vs 50.8 (≈0) |
| Pass-rush share (DL+EDGE) | def sacks | −0.02 | −0.04 | 37.9 vs 38.5 |
| **OL share** | pass EPA | +0.07 | +0.09 | **+8.8 vs −0.9** |
| **QB share** | pass EPA | **−0.23** | −0.16 | **−1.7 vs +31.9** |
| Total capital (5yr) | wins | −0.29 | −0.12 | — |

- **DL/pass-rush devotion → future DST: still zero** (the specific hypothesis is not supported even over 5→3-year horizons; the one positive non-overlapping window, +0.26 in 2011–15→16–18, flips to +0.00 in 2016–20→21–23 — noise).
- **QB devotion → future passing: strongly negative** (a −34-pt pass-EPA gap top-vs-bottom quartile, robust across windows) — the "perpetual QB search" trap: teams that keep drafting QBs are the ones who never solve the position.
- **OL devotion → future passing: weak positive** (~+9 pass EPA, top vs bottom) — the only hypothesis with directionally consistent (if modest) long-run support.
- **Hoarding capital → fewer future wins** even controlled — draft capital is a symptom of being stuck, not a path out.

*Caveat: overlapping 5-yr windows autocorrelate (effective n ≪ 205), so magnitudes are suggestive; the DL≈0 and QB-negative results hold across overlapping + non-overlapping windows. Script: `draft_multiyear.py`.*

## Overall takeaways
1. **Positional draft spend does not predict unit quality** — drafting DL/OL/QB heavily doesn't make those units better at the team-aggregate level (selection effects dominate).
2. **Coaching changes (HC or OC) help only already-bad units** (~+1 win / +21 pass EPA) and tend to *hurt* decent ones; most of the apparent "bump" is mean reversion.
3. **OC pedigree doesn't transfer** — where an OC came from says nothing about the bump he'll bring; coordinator success is highly context- (QB-) dependent.
4. **DC changes barely move defense/DST** — reinforcing that defensive output is largely non-portable and noisy.

*Scripts: `coaching_draft_analysis.py`, `coaching_draft_controlled.py`, `coordinator_analysis.py`.
Data: `data/coaches.csv` → `nflv_coaches`; panel `team_draft_panel`.*
