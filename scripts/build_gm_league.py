"""Turn the ESPN pull into a gm/ league config the simulator can load.

Generated rather than hand-written, so it can be rebuilt after every ESPN pull instead of drifting
away from the league it describes. Reads outputs/espn_league.json (settings, rosters, keeper
prices), outputs/espn_league_state.json (schedule, standings, playoff shape) and
outputs/espn_drafts.json (draft history, which is where the keeper clock comes from).

Two things are decoded rather than copied.

ESPN scores by numeric statId. The twelve that drive skill-position scoring are mapped below and
were checked against this league's own values: they come back internally consistent (6/6/6 for the
three touchdown types, 2/2/2 for conversions, 0.04 and 0.1 per yard, 1.0 per reception). Kicker and
defence IDs are carried through unmapped in scoring.espn_raw, because K and DST projections come
from the Subvertadown blend rather than from a stat line, and guessing at field-goal distance
buckets would put wrong numbers somewhere they look right.

times_kept is derived the way scripts/predict_keepers.py derives it -- count the drafts in which a
player was flagged as a keeper. It is the field no platform reports, and the three-keep rule needs
it, so it is computed here rather than left for someone to maintain by hand.
"""
import os, re, json, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")

# ESPN statId -> the stat names the projection pipeline emits. Only the skill-position scoring
# that a projected stat line can actually be scored against.
STAT_IDS = {
    3: "pass_yds", 4: "pass_tds", 19: "pass_2pt", 20: "pass_int",
    24: "rush_yds", 25: "rush_tds", 26: "rush_2pt",
    42: "rec_yds", 43: "rec_tds", 44: "rec_2pt", 53: "rec",
    72: "fumbles_lost",
}


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def main():
    lg = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json")))
    stt = json.load(open(os.path.join(ROOT, "outputs", "espn_league_state.json")))
    drafts = json.load(open(os.path.join(ROOT, "outputs", "espn_drafts.json")))

    # ---- the keeper clock, from draft history ----
    kept_years = {}
    for r in drafts:
        if str(r.get("keeper")).lower() in ("true", "1"):
            k = norm(r["player"]); kept_years[k] = kept_years.get(k, 0) + 1

    # ---- scoring ----
    per_stat, unmapped = {}, {}
    for si in lg["scoring_raw"]:
        sid, pts = si["statId"], si["points"]
        if sid in STAT_IDS:
            per_stat[STAT_IDS[sid]] = pts
        elif pts:                                    # keep only the non-zero unknowns
            unmapped[str(sid)] = pts

    # ---- schedule: N matchup periods of regular season, then one week per playoff round ----
    ss = stt["schedule_settings"]
    reg = list(range(1, int(ss["matchup_periods"]) + 1))
    n_po, rlen = int(ss["playoff_team_count"]), int(ss["playoff_round_length"] or 1)
    byes = 2 ** (n_po - 1).bit_length() - n_po       # 6 teams -> 8-team bracket -> 2 byes
    rounds = (n_po - byes).bit_length()              # 4 teams after round 1 -> 3 rounds total
    po = [reg[-1] + 1 + i * rlen for i in range(rounds)]

    budget = float(stt["acquisition_settings"]["budget"] or 0)
    slots = lg["roster_slots"]
    order = ["QB", "RB", "WR", "TE", "FLEX", "K", "DST"]

    teams, by_id = [], {t["id"]: t for t in lg["teams"]}
    for t in stt["teams"]:
        rost = []
        for e in (by_id.get(t["team_id"], {}) or {}).get("roster", []):
            rost.append({"player_id": str(e["espn_id"]), "name": e["name"], "pos": e["pos"],
                         "nfl_team": e["team"], "slot": e["slot"], "acquired": e.get("acq"),
                         "keeper_cost": {"price": e.get("keeper_price") or 0},
                         "times_kept": kept_years.get(norm(e["name"]), 0)})
        teams.append({"team_id": str(t["team_id"]), "name": t["name"], "owner": t["owner"],
                      "record": {"w": t["wins"] or 0, "l": t["losses"] or 0, "t": t["ties"] or 0},
                      "points_for": t["points_for"], "points_against": t["points_against"],
                      "playoff_seed": t["playoff_seed"],
                      "faab_remaining": round(budget - float(t["faab_spent"] or 0), 2),
                      "roster": rost})

    cfg = {
        "schema_version": 1,
        "league_id": str(lg["leagueId"]), "name": lg["name"], "platform": "espn",
        "season": lg["season"], "my_team_id": str(lg["myTeamId"]),
        "roster": {
            "starters": [{"slot": s, "count": slots[s]} for s in order if slots.get(s)],
            "bench": slots.get("BENCH", 0), "ir": slots.get("IR", 0),
            "flex_eligibility": {"FLEX": ["RB", "WR", "TE"]},
        },
        "scoring": {"per_stat": per_stat, "bonuses": [], "espn_raw": unmapped,
                    "note": "K and DST are not scored from a stat line here; they come from the "
                            "Subvertadown/kdst blend. espn_raw holds the non-zero statIds this "
                            "mapping does not decode."},
        "schedule": {
            "regular_season_weeks": reg, "playoff_weeks": po, "championship_weeks": [po[-1]],
            "playoff_teams": n_po, "first_round_byes": byes,
            "seeding": "record_then_points_for",
            "tiebreakers": ["points_for"],
            "seeding_note": "Derived from the live standings, not assumed: every 1-0 team seeds "
                            "above every 0-1 team, and within each group the order is points-for "
                            "descending. ESPN's playoffSeedingRule TOTAL_POINTS_SCORED is the "
                            "tiebreak, not the primary sort.",
        },
        "keepers": {
            "enabled": True, "max_per_team": lg["draft"]["keeperCount"],
            "cost_model": "auction", "max_times_kept": 3,
            "auction_escalation_pct": 0,
            "inflation": {
                "base": 5,
                "conditions": [{"name": "missed_playoffs", "delta": -2},
                               {"name": "missed_playoffs_and_drafted_champion", "delta": -3}],
                "note": "+$5 default, +$3 for a team that missed the playoffs, +$0 for the one "
                        "non-playoff owner who drafted the eventual champion in the side draft. "
                        "The side-draft result is not in ESPN's API; see fetch_espn_standings.py.",
            },
            "clock_note": "Keepers began in 2024, so the 3-year cap does not bind for 2026.",
        },
        "acquisitions": {
            "waiver_type": "faab", "faab_budget": budget, "faab_carryover": False, "min_bid": 0,
            "trade_deadline_week": 13,
            "tradeable": {"faab": True, "draft_picks": False},
            "note": "ESPN reports acquisitionType WAIVERS_TRADITIONAL, but every team carries an "
                    "acquisitionBudgetSpent and several are non-zero, so the league is being "
                    "played as FAAB. Worth confirming before the sim prices a waiver claim.",
        },
        "teams": teams,
        "matchups": [{"week": g["matchup_period"], "home": str(g["home_team_id"]),
                      "away": str(g["away_team_id"])}
                     for g in stt["games"] if g["home_team_id"] and g["away_team_id"]],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(cfg, open(OUT, "w"), indent=1)

    sys.path.insert(0, ROOT)
    from gm.config import load
    c = load(OUT)                                    # refuse to ship a config that will not load
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    print(f"  {c.raw['name']}: {len(c.teams)} teams, starters {c.starters}")
    print(f"  regular weeks {reg[0]}-{reg[-1]}, playoffs {po}, {n_po} teams, {byes} bye(s)")
    print(f"  scoring decoded: {per_stat}")
    print(f"  undecoded non-zero statIds (K/DST/misc): {sorted(unmapped, key=int)}")
    print(f"  keeper clock: {sum(1 for t in teams for e in t['roster'] if e['times_kept'])} "
          f"roster spots carry a keeper year")
    print(f"  expiring next year (at {c.raw['keepers']['max_times_kept']-1} keeps): "
          f"{len(c.expiring_next_year())}")


if __name__ == "__main__":
    main()
