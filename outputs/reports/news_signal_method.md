# Beat-writer / practice-report signal: what we have, what we can test, how to collect it

Date: 2026-09-10. Collector: `scripts/news_collector.py`. Data: `data/news/`.

## 1. The premise, checked against the league's own drafts (2023-25, 487 skill picks)

Surplus = actual season PPR − FFA preseason projection (league scoring). Hit = actual ≥ 1.2 × projection.

| group | n | mean surplus | z within pos-season | hit rate |
|---|---|---|---|---|
| Bears fans (3 declared), Bears picks | 11 | −34.6 | +0.26 | 18% |
| Bears fans, non-Bears picks | 99 | −41.1 | +0.03 | 14% |
| non-fans, Bears picks | 7 | +1.2 | +0.55 | 14% |
| Bears fans, Bears darts (≤$5) | 4 | −2.7 | +0.26 | 50% |
| all owners, fan-team darts (declared + inferred) | 15 | −30.3 | +0.09 | 27% |
| all owners, non-fan darts | 172 | −47.0 | +0.02 | 17% |

The Bears-dart story is Cole Kmet at $1 and Roschon Johnson at $4 in 2023; the league-wide
version is four hits in fifteen (add Sutton $2 in 2024, Bucky Irving $1 in 2024) versus 17%
base. That is 1.5 extra hits over expectation. It is consistent with the hypothesis and equally
consistent with nothing. Fan-team picks at full price show no edge at all (z +0.06 vs 0.00).
So the league data cannot carry the question; it has to be tested on the scalable proxy.

## 2. The scalable proxy: local coverage as "fan knowledge"

If the mechanism is "people who read the local beat every day know role changes before the
consensus prices them", the measurable version is: does a role/usage signal that appears in
local beat coverage predict (actual − line) after the market has had its say, or predict the
line's own subsequent move? Two different questions, and only one is testable historically.

## 3. What the market side allows

| line data | coverage | usable for |
|---|---|---|
| DK closing props (T−2h) | 2023-25, every game | "does news the market already saw still predict the outcome?" (weak test: closing residual) |
| early DK lines (≥24h pre-kick) | 4 of 256 events in 2025; 0 at ≥48h | nothing — we never stored openers |
| 2026 generation caches (Wed/Thu/Sun) | from Week 1 on | the real test, going forward |

Our 2025 prop history is closing-only: median first snapshot 2.1 h before kickoff. A
historical "news beat the line" backtest is not possible from what exists; only the closing-
residual test is, and a beat-writer signal that survives it would be news the market ignored.

## 4. What the news side allows (all free, verified today)

| source | timestamps | history | per-player tag | verdict |
|---|---|---|---|---|
| Bluesky beat writers (public API, author feeds) | post-level | back to Nov 2024 for most NFL writers | text only (name match) | **primary**. 187 candidate accounts found across 32 teams (`data/news/beat_writers_bsky.csv`, 21 flagged as other sports, needs a human pass; TB/HOU/JAX/TEN thin). Low volume from the big names (Biggs ~15 posts/month). |
| ESPN NFL news API | article-level | forward only | athlete + team tags | secondary; national, not local |
| NFL.com injury page | page snapshot | forward only | player, practice status, game status, 30 teams | **the official Wed/Thu/Fri report**, snapshot daily |
| Sleeper trending adds/drops | daily counts | forward only | player id | crowd attention, forward only |
| Wikipedia daily pageviews | daily | 2015-07 onward | article | **the only historical attention proxy with dates**; validated Puka 2023 (Aug 1: 146 → Sep 1: 607 → Wk1: 30k) |
| Reddit team subs | — | — | — | needs OAuth; skipped |
| X / Twitter | — | — | — | $100+/mo API; where most beat writers still post first |
| Rotowire / NBC Edge news | dated blurbs | deep archive | player | scrape only; terms of service issue, not used |

## 5. Method (revised 2026-09-10 after the user's decision: no Odds API spend)

The projection is sent ~75 min before kickoff, so the only market that matters is the final
line. Whether a line moved from Wednesday is irrelevant to that use; what matters is whether
information that exists at T−75 is still absent from the line. That test needs closing lines
only, which we hold for 2023-25, and it costs nothing.

**The structural window.** Our stored closing snapshot is T−2h. Official inactives post at
T−90 min. The generator runs at T−75. So the line we compare against predates the inactives,
and the one upgrade that survived this week (Questionable with no DK line → plays 20%) is
exactly a T−2h-to-T−75 information gap. The news test is a generalisation of that: which
pregame information (practice progression, beat-writer role notes, attention spikes, late
inactives) predicts (actual − closing line) at all.

**Historical (running now, free):** Wikipedia daily views 2023-08 → 2026-01 for the 2,313
players in nflv_wiki_titles; Bluesky beat-writer posts back to Nov 2024; nflverse injuries
(2025 only at present). Signals, all stamped before T−2h: attention spike (Wed-Sat views vs
own trailing median), beat-writer role keywords, practice progression. Target: actual − DK
closing line on the props frame, and actual − FFA, walk-forward by week, overs/unders separately.

**Forward (free):** `news_collector.py snapshot` on the days we generate anyway; the
generator's own T−75 DK pull is the "final line" and is already cached per send. No extra
credits.

**The fan-knowledge test proper:** local-only beat posts (not echoed by ESPN within 6 h) vs
national ones, same target.

## 6. Honest expectations

Four studies and ~45 features show FFA + DK already price everything derivable from box scores
and charting. News is the one input class that is not box-score-derived, and the one upgrade
that survived this week (Questionable with no DK line → 20% plays) is a news-timing effect. So
this is the right place to look. But the beat writers with the most information still post on
X first, the market reads them within minutes, and our history is closing-only. The realistic
prize is the Sunday-morning window (inactives, pregame reports) and the $1-3 dart pool in
August, not Wednesday props.
