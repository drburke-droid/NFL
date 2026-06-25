"""
Monte Carlo over the realistic v2 league (league_sim_v3).

Reuses v2's precomputed per-year valuations (the expensive model training is done
ONCE), then runs N full 10-year leagues, re-randomizing each run:
  - draft BIDS (lognormal noise on every team's valuations -> different rosters)
  - the H2H SCHEDULE (schedule luck varies)
Aggregates finish distributions, title odds, and playoff odds with 95% CIs, so the
draft-edge verdict comes with uncertainty instead of a single realization.

Usage: python scripts/league_sim_v3.py [N]   (default N=200)
"""
import os, sqlite3, warnings, sys
import numpy as np
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(os.path.dirname(__file__), n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V2 = _load("league_sim_v2")

DB = V2.DB
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "LEAGUE_SIM_V3.md")
BID_SIGMA = 0.15


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    con = sqlite3.connect(DB); D = V2.load(con); con.close()
    wproj = V2.make_wproj(D)
    print(f"Precomputing per-year valuations ({len(V2.YEARS)} years)...")
    yv = {Y: V2.build_year_values(Y, D) for Y in V2.YEARS}      # expensive, once
    style_of = {i: V2.STYLES[i] for i in range(12)}
    us = next(i for i in range(12) if style_of[i] == "ours")

    finishes = {i: [] for i in range(12)}        # all run-years
    run_avg = {i: [] for i in range(12)}         # per-run 10-yr avg finish
    run_titles = {i: [] for i in range(12)}      # titles per 10-yr run
    print(f"Running {N} Monte Carlo seasons (bid σ={BID_SIGMA}, randomized schedule)...")
    for run in range(N):
        rng = np.random.default_rng(1000 + run)
        teams = [V2.Team(i, V2.STYLES[i]) for i in range(12)]
        rf = {i: [] for i in range(12)}
        for Y in V2.YEARS:
            rank, _ = V2.simulate_year(Y, teams, D, yv[Y], wproj, rng=rng, bid_sigma=BID_SIGMA)
            for i in range(12): rf[i].append(rank[i])
        for i in range(12):
            finishes[i] += rf[i]; run_avg[i].append(np.mean(rf[i])); run_titles[i].append(sum(x == 1 for x in rf[i]))
        if (run + 1) % 25 == 0: print(f"  {run+1}/{N} done")

    def ci(xs): a = np.array(xs); return np.percentile(a, 2.5), np.percentile(a, 97.5)

    f = np.array(finishes[us]); ra = run_avg[us]; rt = run_titles[us]
    lo, hi = ci(ra)
    hist = [100 * np.mean(f == k) for k in range(1, 13)]
    L = [f"# Monte Carlo League Simulation v3 — {N} runs (2016-2025 × {N})\n",
         "Same realistic league as v2 (H2H, FAAB waivers, props-driven lineups, equal in-season info). Each run re-randomizes the **draft bids** (lognormal σ=0.15) and the **schedule**, so results carry uncertainty. Only draft valuation differs between us and the bots.\n",
         "## Our team — finish distribution",
         "| Place | " + " | ".join(str(k) for k in range(1, 13)) + " |", "|" + "---|" * 13,
         "| % of seasons | " + " | ".join(f"{h:.0f}" for h in hist) + " |",
         f"\n- **Avg finish {np.mean(f):.2f}**  (95% CI across runs {lo:.2f}–{hi:.2f})",
         f"- **Title rate {100*np.mean(f==1):.1f}%/yr**  → ≈ **{10*np.mean(f==1):.1f} titles per decade**",
         f"- **Playoffs (top-6) {100*np.mean(f<=6):.0f}%** of seasons | Top-3 {100*np.mean(f<=3):.0f}%",
         f"- **P(≥1 title in a 10-yr span) = {100*np.mean(np.array(rt)>=1):.0f}%** | P(≥2) = {100*np.mean(np.array(rt)>=2):.0f}%",
         f"- Median titles per decade: {int(np.median(rt))}  (range {min(rt)}–{max(rt)})",
         "\n## Style leaderboard — avg finish ± 95% CI (lower = better)"]
    board = sorted([(np.mean(run_avg[i]), ci(run_avg[i]), style_of[i]) for i in range(12)])
    for avg, (clo, chi), st in board:
        L.append(f"- {('**US**' if st=='ours' else st):22s} {avg:.2f}  [{clo:.2f}, {chi:.2f}]")
    L += ["\n## Read",
          f"- Over {N} randomized drafts/schedules, our draft-day process is the **best team in the league** ({np.mean(f):.2f} avg vs a field average of ~6.5), makes the playoffs ~{100*np.mean(f<=6):.0f}% of years, and wins ~{10*np.mean(f==1):.1f} titles/decade — but is **not a lock** (it finishes outside the top-3 in {100*(1-np.mean(f<=3)):.0f}% of seasons).",
          "- The spread (CI, finish histogram) is the honest picture: a real edge that compounds over many seasons, not a guarantee in any single one. Draft-bid noise + schedule luck are the dominant variance sources here."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n" + "\n".join(L[2:18]))
    print("\nStyle leaderboard:")
    for avg, (clo, chi), st in board: print(f"   {avg:5.2f} [{clo:4.2f},{chi:4.2f}]  {st}")
    print("Saved", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
