"""The league description a rest-of-season simulator needs, and the checks that keep it honest.

A trade is only good or bad relative to a league. The same player is worth a playoff berth to one
team and nothing to another, and that depends on the standings, the playoff shape, what it costs to
keep him next year, and what is sitting on waivers. So the simulator takes a league, not a pair of
players, and this module is the contract for what "a league" means.

Two things drive the design.

The first is that most of this is available from the platform API and one part is not. Sleeper and
ESPN will hand over rosters, scoring, standings and matchups. Neither knows how many times a player
has been kept, because that is league lore living in a spreadsheet or somebody's memory. The rule
that a player kept three times becomes ineligible therefore needs `times_kept` carried per roster
entry, maintained by hand, and it is the one field nothing can reconstruct for you.

The second is that a config that is merely well-formed is not enough. A simulator fed a bracket
with more playoff teams than the league has, or a schedule where a team plays twice in a week, will
not crash -- it will return confident nonsense. So validate() checks the things that would corrupt
a simulation quietly rather than loudly, and every failure names the field and the offending value.

Scoring keys match the stat columns the projection pipeline already emits (pass_yds, rush_tds,
rec, ...), so a projected stat line can be scored by this config without translation.
"""
import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
# A slot is either a position or a multi-position slot defined in roster.flex_eligibility.
COST_MODELS = ("draft_round", "auction", "none")
# Inflation conditions are a fixed vocabulary, not an expression language. A league rule that
# cannot be said with these names should get a new name here, where it is visible and testable,
# rather than an eval() that silently accepts anything.
INFLATION_CONDITIONS = ("missed_playoffs", "missed_playoffs_and_drafted_champion")
WAIVER_TYPES = ("faab", "rolling", "reverse_standings")


class LeagueConfigError(ValueError):
    """Raised with the offending field and value; never a bare 'invalid config'."""


@dataclass
class LeagueConfig:
    raw: dict

    # ---- roster ----
    @property
    def starters(self):
        """[(slot, count)] in lineup order."""
        return [(s["slot"], int(s["count"])) for s in self.raw["roster"]["starters"]]

    @property
    def starting_size(self):
        return sum(c for _, c in self.starters)

    def slot_eligible(self, slot, position):
        if slot in POSITIONS:
            return slot == position
        return position in self.raw["roster"].get("flex_eligibility", {}).get(slot, [])

    # ---- scoring ----
    def score(self, statline):
        """Points for a stat line dict. Unknown stats are ignored; missing ones count as zero."""
        sc = self.raw["scoring"]
        pts = sum(float(statline.get(k, 0) or 0) * float(v) for k, v in sc["per_stat"].items())
        for b in sc.get("bonuses", []):
            if float(statline.get(b["stat"], 0) or 0) >= float(b["threshold"]):
                pts += float(b["points"])
        return pts

    # ---- schedule ----
    @property
    def regular_weeks(self):
        return list(self.raw["schedule"]["regular_season_weeks"])

    @property
    def playoff_weeks(self):
        return list(self.raw["schedule"]["playoff_weeks"])

    def is_playoff_week(self, week):
        return week in self.playoff_weeks

    # ---- keepers ----
    @property
    def keepers_on(self):
        return bool(self.raw.get("keepers", {}).get("enabled"))

    def keeper_eligible(self, entry):
        """A roster entry may be kept again only if it has not exhausted its keeper years.

        This is the rule no platform can answer for you: with max_times_kept = 3, a player already
        kept three times returns to the draft pool next year no matter how good he is. That makes a
        keeper year a depreciating asset -- two identical players are not worth the same if one has
        one year of control left and the other has three.
        """
        if not self.keepers_on:
            return False
        k = self.raw["keepers"]
        return int(entry.get("times_kept", 0)) < int(k["max_times_kept"])

    def keeper_cost(self, entry):
        """What keeping this player costs next year, in the league's own currency.

        draft_round: the pick you forfeit, escalating by round_penalty per year kept.
        auction:     last year's price grown by auction_escalation_pct per year kept.
        Returns None when the entry cannot be kept at all.
        """
        if not self.keeper_eligible(entry):
            return None
        k = self.raw["keepers"]
        model, cost = k["cost_model"], entry.get("keeper_cost") or {}
        if model == "draft_round":
            base = cost.get("round", k.get("undrafted_cost_round"))
            if base is None:
                return None
            return {"round": max(1, int(base) - int(k.get("round_penalty", 0)))}
        if model == "auction":
            base = cost.get("price")
            if base is None:
                return None
            return {"price": round(float(base) * (1 + float(k.get("auction_escalation_pct", 0)) / 100), 2)}
        return {}

    def keeper_inflation(self, made_playoffs, drafted_champion=False):
        """Next season's bump on this team's keeper prices, in auction dollars.

        Kuhn and Friends charges +$5 by default, reduced to +$3 for a team that missed the
        playoffs and to +$0 for the one non-playoff owner who drafted the eventual champion in
        the side draft. That makes inflation a function of where a team FINISHES, so a simulator
        cannot price next year's keepers until it has simulated this year's standings -- the two
        are a loop, not a sequence. Deltas accumulate in listed order.
        """
        inf = self.raw.get("keepers", {}).get("inflation")
        if not inf:
            return 0.0
        facts = {"missed_playoffs": not made_playoffs,
                 "missed_playoffs_and_drafted_champion": (not made_playoffs) and drafted_champion}
        bump = float(inf["base"])
        for c in inf.get("conditions", []):
            if facts.get(c["name"]):
                bump += float(c["delta"])
        return bump

    def expiring_next_year(self):
        """Players league-wide who hit their last keeper year -- next season's known supply.

        Front offices track the upcoming free agent class because it changes what you should pay
        today. The 3-keep rule makes that class computable exactly: if three elite backs are forced
        into the pool next spring, the price of a back right now should reflect it.
        """
        if not self.keepers_on:
            return []
        limit = int(self.raw["keepers"]["max_times_kept"])
        out = []
        for t in self.raw["teams"]:
            for e in t.get("roster", []):
                if int(e.get("times_kept", 0)) == limit - 1:
                    out.append({"team_id": t["team_id"], **e})
        return out

    # ---- teams ----
    @property
    def teams(self):
        return self.raw["teams"]

    def team(self, team_id):
        for t in self.raw["teams"]:
            if t["team_id"] == team_id:
                return t
        raise LeagueConfigError(f"teams: no team with team_id {team_id!r}")


def _require(cond, msg):
    if not cond:
        raise LeagueConfigError(msg)


def validate(raw):
    """Reject the configs that would make a simulator lie rather than fail."""
    _require(isinstance(raw, dict), "config must be a JSON object")
    _require(raw.get("schema_version") == SCHEMA_VERSION,
             f"schema_version: expected {SCHEMA_VERSION}, got {raw.get('schema_version')!r}")
    for k in ("roster", "scoring", "schedule", "teams"):
        _require(k in raw, f"missing required section {k!r}")

    # roster
    flex = raw["roster"].get("flex_eligibility", {})
    starters = raw["roster"].get("starters") or []
    _require(starters, "roster.starters: at least one starting slot is required")
    for s in starters:
        slot = s["slot"]
        _require(slot in POSITIONS or slot in flex,
                 f"roster.starters: slot {slot!r} is neither a position nor defined in flex_eligibility")
        _require(int(s["count"]) >= 1, f"roster.starters: {slot!r} count must be >= 1")
    for slot, allowed in flex.items():
        bad = [p for p in allowed if p not in POSITIONS]
        _require(not bad, f"roster.flex_eligibility[{slot!r}]: unknown position(s) {bad}")

    # scoring
    _require(raw["scoring"].get("per_stat"), "scoring.per_stat: must define at least one stat")

    # schedule
    sch = raw["schedule"]
    reg, po = set(sch.get("regular_season_weeks", [])), set(sch.get("playoff_weeks", []))
    _require(reg, "schedule.regular_season_weeks: must not be empty")
    _require(po, "schedule.playoff_weeks: must not be empty")
    _require(not (reg & po),
             f"schedule: weeks {sorted(reg & po)} are listed as both regular season and playoff")
    champ = set(sch.get("championship_weeks", []))
    _require(champ <= po,
             f"schedule.championship_weeks: {sorted(champ - po)} are not in playoff_weeks")

    # teams
    teams = raw["teams"]
    ids = [t["team_id"] for t in teams]
    _require(len(ids) == len(set(ids)), "teams: team_id values must be unique")
    _require(len(teams) >= 2, "teams: need at least two teams")
    seen = {}
    for t in teams:
        for e in t.get("roster", []):
            pid = e.get("player_id")
            _require(pid not in seen,
                     f"teams: player {pid!r} is on both {seen.get(pid)!r} and {t['team_id']!r}")
            seen[pid] = t["team_id"]

    # playoff bracket has to be fillable
    n_po = int(sch.get("playoff_teams", 0))
    _require(1 <= n_po <= len(teams),
             f"schedule.playoff_teams: {n_po} with only {len(teams)} teams in the league")
    byes = int(sch.get("first_round_byes", 0))
    _require(0 <= byes < n_po, f"schedule.first_round_byes: {byes} is not below playoff_teams {n_po}")
    _require((n_po - byes) % 2 == 0,
             f"schedule: {n_po} playoff teams with {byes} bye(s) leaves an odd first round")

    # keepers
    if raw.get("keepers", {}).get("enabled"):
        k = raw["keepers"]
        _require(k.get("cost_model") in COST_MODELS,
                 f"keepers.cost_model: {k.get('cost_model')!r} not in {COST_MODELS}")
        limit = int(k.get("max_times_kept", 0))
        _require(limit >= 1, "keepers.max_times_kept: must be at least 1 when keepers are enabled")
        _require(int(k.get("max_per_team", 0)) >= 1, "keepers.max_per_team: must be at least 1")
        for t in teams:
            for e in t.get("roster", []):
                tk = int(e.get("times_kept", 0))
                _require(0 <= tk <= limit,
                         f"teams[{t['team_id']}].roster[{e.get('player_id')}]: times_kept {tk} "
                         f"outside 0..{limit}")

        inf = k.get("inflation")
        if inf is not None:
            _require("base" in inf, "keepers.inflation: needs a 'base' bump")
            for c in inf.get("conditions", []):
                _require(c.get("name") in INFLATION_CONDITIONS,
                         f"keepers.inflation: condition {c.get('name')!r} not in {INFLATION_CONDITIONS}")
                _require("delta" in c, f"keepers.inflation: condition {c['name']!r} needs a 'delta'")

    # acquisitions
    acq = raw.get("acquisitions", {})
    if acq:
        _require(acq.get("waiver_type") in WAIVER_TYPES,
                 f"acquisitions.waiver_type: {acq.get('waiver_type')!r} not in {WAIVER_TYPES}")
        if acq.get("waiver_type") == "faab":
            budget = float(acq.get("faab_budget", 0))
            _require(budget > 0, "acquisitions.faab_budget: must be positive for a FAAB league")
            for t in teams:
                if "faab_remaining" in t:
                    rem = float(t["faab_remaining"])
                    _require(0 <= rem <= budget,
                             f"teams[{t['team_id']}].faab_remaining: {rem} outside 0..{budget}")
        dl = acq.get("trade_deadline_week")
        if dl is not None:
            _require(dl in reg,
                     f"acquisitions.trade_deadline_week: {dl} is not a regular season week")

    # schedule table: every team plays exactly once per regular week
    for m in raw.get("matchups", []):
        for side in ("home", "away"):
            _require(m[side] in set(ids), f"matchups: unknown team_id {m[side]!r} in week {m['week']}")
        _require(m["home"] != m["away"], f"matchups: team {m['home']!r} plays itself in week {m['week']}")
    by_week = {}
    for m in raw.get("matchups", []):
        by_week.setdefault(m["week"], []).extend([m["home"], m["away"]])
    for wk, played in sorted(by_week.items()):
        dupes = {t for t in played if played.count(t) > 1}
        _require(not dupes, f"matchups: week {wk} has team(s) {sorted(dupes)} playing more than once")
    # A simulator handed a week with a missing game still scores every team but awards no win to
    # the ones left out, so records and playoff odds come back quietly wrong rather than failing.
    # An empty matchup table is the same bug at full size.
    if raw.get("matchups"):
        everyone = set(ids)
        byes_allowed = len(ids) % 2            # an odd league sits exactly one team out each week
        for wk in reg:
            missing = everyone - set(by_week.get(wk, []))
            _require(len(missing) <= byes_allowed,
                     f"matchups: regular week {wk} is missing {sorted(missing)}")
    else:
        _require(all(t["record"]["w"] + t["record"]["l"] + t["record"]["t"] == 0
                     for t in teams if "record" in t),
                 "matchups: a league with games already played needs its schedule")
    return raw


def load(src):
    """Load and validate. `src` is a path or an already-parsed dict."""
    raw = src if isinstance(src, dict) else json.load(open(src))
    return LeagueConfig(validate(raw))
