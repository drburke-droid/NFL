"""Pull the league state a rest-of-season simulator needs: schedule, standings, playoff shape.

scripts/fetch_espn.py already pulls settings and rosters for the Draft Room, but its normalized
output drops the three things a simulator cannot run without -- the matchup schedule, each team's
record, and the playoff structure. Without the schedule there is no win curve, because a team's
title odds depend on who it still has to play; without the playoff shape there is no bracket to
simulate; without records there is no seeding. So this fetches the same league with the views that
carry them and writes outputs/espn_league_state.json.

It deliberately does not touch espn_league.json. That file feeds the Draft Room and is pulled by
the same workflow; a simulator wanting extra fields is no reason to change a format something else
already consumes.

Raw responses land in data/espn/ (gitignored) so a parsing question can be answered without
spending another authenticated call.

Auth is the same as fetch_espn.py: ESPN_S2 / ESPN_SWID from the environment (the Action passes
repo Secrets) or data/espn_cookies.json locally.

Usage:  python scripts/fetch_espn_league_state.py [leagueId] [season]
"""
import os, sys, json
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LEAGUE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1211359110
# One or more seasons. Past seasons are what calibration needs: the weekly team scores that say
# how far a team's week actually swings, which is the input the simulator currently guesses at.
SEASONS = [int(a) for a in sys.argv[2:]] or [2026]
CURRENT = max(SEASONS)


def out_path(season):
    """The newest season also writes the unsuffixed name that build_gm_league.py reads."""
    return os.path.join(ROOT, "outputs", f"espn_league_state_{season}.json")
RAW = os.path.join(ROOT, "data", "espn"); os.makedirs(RAW, exist_ok=True)


def cookies():
    s2, swid = os.environ.get("ESPN_S2"), os.environ.get("ESPN_SWID")
    f = os.path.join(ROOT, "data", "espn_cookies.json")
    if (not s2 or not swid) and os.path.exists(f):
        d = json.load(open(f)); s2 = s2 or d.get("espn_s2"); swid = swid or d.get("SWID") or d.get("swid")
    return s2, swid


def fetch(season, s2, swid):
    url = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
           f"/segments/0/leagues/{LEAGUE_ID}")
    params = [("view", v) for v in ("mSettings", "mTeam", "mMatchup", "mMatchupScore")]
    r = requests.get(url, params=params, cookies={"espn_s2": s2, "SWID": swid},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if r.status_code != 200:
        # raise rather than exit: a season the league did not exist for must not abort the rest
        raise RuntimeError(f"HTTP {r.status_code} ({r.text[:120]}) "
                           "- 401 means cookies missing/expired, 404 a bad leagueId/season")
    data = r.json()
    json.dump(data, open(os.path.join(RAW, f"league_state_{season}.json"), "w"), indent=1)

    st = data.get("settings", {}) or {}
    sched_s = st.get("scheduleSettings", {}) or {}
    acq_s = st.get("acquisitionSettings", {}) or {}
    trade_s = st.get("tradeSettings", {}) or {}
    members = {m.get("id"): (m.get("displayName") or m.get("firstName", "")) for m in data.get("members", [])}

    teams = []
    for t in data.get("teams", []):
        ov = ((t.get("record") or {}).get("overall") or {})
        tc = t.get("transactionCounter") or {}
        teams.append({
            "team_id": t.get("id"),
            "name": (t.get("name") or f"{t.get('location','')} {t.get('nickname','')}").strip(),
            "abbrev": t.get("abbrev"),
            "owner": next((members.get(o) for o in (t.get("owners") or []) if members.get(o)), None),
            "wins": ov.get("wins"), "losses": ov.get("losses"), "ties": ov.get("ties"),
            "points_for": round(float(ov.get("pointsFor") or 0), 2),
            "points_against": round(float(ov.get("pointsAgainst") or 0), 2),
            "playoff_seed": t.get("playoffSeed"),
            "faab_spent": tc.get("acquisitionBudgetSpent"),
            "acquisitions": tc.get("acquisitions"),
        })

    # ESPN keys games by matchupPeriodId. In a league whose playoff rounds span two scoring
    # periods those are not the same as NFL weeks, so carry both and let the config say which.
    games = []
    for m in data.get("schedule", []) or []:
        h, a = m.get("home") or {}, m.get("away") or {}
        games.append({
            "matchup_period": m.get("matchupPeriodId"),
            "home_team_id": h.get("teamId"), "away_team_id": a.get("teamId"),
            "home_points": h.get("totalPoints"), "away_points": a.get("totalPoints"),
            "winner": m.get("winner"), "playoff_tier": m.get("playoffTierType"),
        })

    out = {
        "leagueId": LEAGUE_ID, "season": season, "name": st.get("name"), "size": st.get("size"),
        "schedule_settings": {
            "matchup_periods": sched_s.get("matchupPeriodCount"),
            "playoff_team_count": sched_s.get("playoffTeamCount"),
            "playoff_round_length": sched_s.get("playoffMatchupPeriodLength"),
            "playoff_seeding_rule": sched_s.get("playoffSeedingRule"),
            "playoff_reseed": sched_s.get("playoffReseed"),
            "matchup_period_length": sched_s.get("matchupPeriodLength"),
            "divisions": len((sched_s.get("divisions") or [])),
        },
        "acquisition_settings": {
            "budget": acq_s.get("acquisitionBudget"),
            "type": acq_s.get("acquisitionType"),
            "waiver_hours": acq_s.get("waiverHours"),
            "final_place_transaction_period": acq_s.get("finalPlaceTransactionPeriod"),
        },
        "trade_settings": {
            "deadline_epoch_ms": trade_s.get("deadlineDate"),
            "max": trade_s.get("max"),
            "veto_votes_required": trade_s.get("vetoVotesRequired"),
        },
        "teams": teams,
        "games": games,
    }
    json.dump(out, open(out_path(season), "w"), indent=1)
    if season == CURRENT:
        json.dump(out, open(os.path.join(ROOT, "outputs", "espn_league_state.json"), "w"), indent=1)

    s = out["schedule_settings"]
    print(f"League {out['name']!r} ({out['size']} teams, {season})")
    print(f"  regular season matchup periods : {s['matchup_periods']}")
    print(f"  playoff teams / round length   : {s['playoff_team_count']} / {s['playoff_round_length']}")
    print(f"  seeding rule / reseed          : {s['playoff_seeding_rule']} / {s['playoff_reseed']}")
    print(f"  acquisition budget / type      : {out['acquisition_settings']['budget']} / {out['acquisition_settings']['type']}")
    print(f"  games in schedule              : {len(games)}")
    played = [g for g in games if (g['home_points'] or 0) or (g['away_points'] or 0)]
    print(f"  games with a score so far      : {len(played)}")
    print(f"  wrote {os.path.relpath(out_path(season), ROOT)}")


def main():
    s2, swid = cookies()
    if not s2 or not swid:
        print("No ESPN cookies. Set ESPN_S2 / ESPN_SWID, or create data/espn_cookies.json")
        sys.exit(1)
    for season in SEASONS:
        try:
            fetch(season, s2, swid)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  {season}: failed ({str(e)[:80]})")


if __name__ == "__main__":
    main()
