"""
One-command 2026 draft-board refresh.

Run this whenever new market data lands (preseason FFA consensus and/or ADP).
It re-ingests the market data and rebuilds the entire 2026 board end to end:

    python scripts/refresh_2026.py

To upgrade the projections from prior-year guesses to MARKET-ANCHORED, first drop
the new FantasyFootballAnalytics preseason file here:

    data/ffanalytics/FFAn/projections_2026_wk0.csv

The board auto-detects 2026 FFA coverage and switches to the FFA-anchored model
(the validated best season model) when it's present; otherwise it stays a
prior-year-anchored guess and says so.

Chain: fetch_ffa -> fetch_adp -> build_2026_targets -> report_2026_targets -> build_draft_tool
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FFA_2026 = os.path.join(ROOT, "data", "ffanalytics", "FFAn", "projections_2026_wk0.csv")

# (label, script, required?) — ADP refresh is best-effort (network/source may vary)
STEPS = [
    ("Ingest FFA consensus projections", "fetch_ffa.py", True),
    ("Ingest league-scored FFA (model anchor)", "fetch_ffa_league.py", True),
    ("Refresh ADP / ECR snapshot",       "fetch_adp.py", False),
    ("FFC crowd ADP (live mocks)",       "fetch_ffc_adp.py", False),
    ("Fit projection + boom/bust models", "projection_overhaul.py", True),
    ("Build 2026 board (vets + rookies)", "build_2026_targets.py", True),
    # Must run after build_2026_targets (it replaces board_2026) and before the
    # report + draft tool, so both see every FFA-projected player.
    ("Backfill FFA-only players",        "backfill_ffa_pool.py", True),
    ("Generate targets report",          "report_2026_targets.py", True),
    ("2025 half-season split (Risers)",  "half_season_split.py", False),
    ("Half-season trend features",       "build_half_trend.py", False),
    ("Opportunity flags (2026 rosters)", "build_opportunity.py", False),
    ("Value-leap sleeper scores",        "build_value_leap.py", False),
    ("Implosion / fade-risk scores",     "build_implosion.py", False),
    ("Lottery-ticket tags ($1-3 stars)", "build_lottery_tag.py", False),
    ("WR2 crowding tags",                "build_crowding_tag.py", False),
    ("Rebuild draft tool (data.js)",     "build_draft_tool.py", True),
    # Bakes the FFA consensus into docs/ffa_2026.js so the Blend/FFA toggles work
    # without a per-browser CSV upload.
    ("Bake FFA consensus for the UI",    "build_ffa_js.py", True),
    ("Refresh career comps",             "comp_finder.py", False),
    ("Refresh rookie comps",             "rookie_comps.py", False),
]


def run(script):
    return subprocess.run([sys.executable, os.path.join(HERE, script)],
                          capture_output=True, text=True)


def main():
    print("=" * 64)
    print("2026 DRAFT BOARD REFRESH")
    print("=" * 64)
    if os.path.exists(FFA_2026):
        print(f"[ok] 2026 FFA file found: {os.path.relpath(FFA_2026, ROOT)} -> projections will be MARKET-ANCHORED")
    else:
        print(f"[!] No 2026 FFA file at {os.path.relpath(FFA_2026, ROOT)}")
        print("    Board will be a prior-year-anchored GUESS. Drop the file there and re-run for market-anchored numbers.")

    for label, script, required in STEPS:
        print(f"\n--- {label}  ({script}) ---")
        r = run(script)
        tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
        if tail: print(tail)
        if r.returncode != 0:
            print(f"  stderr: {(r.stderr or '').strip().splitlines()[-3:]}")
            if required:
                print(f"\nFAILED at '{script}'. Fix the error above and re-run.")
                sys.exit(1)
            else:
                print(f"  (non-critical step '{script}' failed — continuing.)")

    print("\n" + "=" * 64)
    print("DONE. Open outputs/draft_tool/index.html and read outputs/models/DRAFT_2026_TARGETS.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
