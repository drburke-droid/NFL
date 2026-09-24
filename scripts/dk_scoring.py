"""DraftKings NFL scoring, the one scoring every fan-facing number is compared in.

Why DraftKings: the weekly accuracy the big sites publish (docs/fan/sites_accuracy.json, from
Fantasy Football Analytics' DFS accuracy page) is graded in DraftKings points. A model error
measured in any other scoring cannot be ranked against theirs, and the model's own grading target,
nflverse fantasy_points_ppr, differs from DraftKings in three places:

    interception        PPR -2   DraftKings -1
    fumble lost         PPR -2   DraftKings -1
    300 pass / 100 rush / 100 rec yards   PPR nothing   DraftKings +3 each

So actual DraftKings points are rescored here from the raw box score, and projected DraftKings
points from the projected stat line. The stat line is scoring-neutral; only the points built on it
are not, which is why the model does not need retraining to be compared fairly.

The bonuses are the one non-linear piece. A projection is a mean, and +3 only when the mean crosses
100 would be wrong in both directions: an 85-yard projection clears 100 about a third of the time.
The projected bonus is therefore its expectation, 3 x P(yards >= threshold | projected mean), with
weekly yards modelled as a gamma distribution whose shape was fitted by maximum likelihood on
2018-25 regular-season player-games (the player's season mean standing in for his projection):

    pass_yds >= 300   shape 10.9   (QBs with 15+ attempts)
    rush_yds >= 100   shape 1.83   (14,440 player-games)
    rec_yds  >= 100   shape 1.81   (32,721 player-games)

Checked against our own 2026 weeks 1-2 projections before use: expected bonuses 18.2 / 24.8 / 33.0
against 18 / 20 / 33 observed. docs/fan.html carries the same constants for the live card; a test
pins the two copies together.
"""
import numpy as np
from scipy.stats import gamma

# points per unit of each projected stat (send CSV / fan bake column names)
LINEAR = {"pass_yds": 0.04, "pass_tds": 4.0, "pass_int": -1.0, "rush_yds": 0.1, "rush_tds": 6.0,
          "rec": 1.0, "rec_yds": 0.1, "rec_tds": 6.0, "fumbles_lost": -1.0}
BONUS_AT = {"pass_yds": 300.0, "rush_yds": 100.0, "rec_yds": 100.0}
BONUS_PTS = 3.0
SHAPE = {"pass_yds": 10.9, "rush_yds": 1.83, "rec_yds": 1.81}
SKILL = ("QB", "RB", "WR", "TE")


def p_bonus(stat, mean):
    """P(a week's yards reach the bonus line | projected mean yards). Vectorised over `mean`."""
    m = np.asarray(mean, dtype=float)
    k, t = SHAPE[stat], BONUS_AT[stat]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = gamma.sf(t, k, scale=np.where(m > 0, m, 1.0) / k)
    return np.where(m > 0, p, 0.0)


def projected_points(stats):
    """Expected DraftKings points of one projected stat line (a dict; missing stats count 0)."""
    v = lambda k: float(stats.get(k) or 0.0)
    pts = sum(v(k) * w for k, w in LINEAR.items())
    return pts + sum(BONUS_PTS * float(p_bonus(k, v(k))) for k in BONUS_AT)


def projected_frame(df):
    """projected_points over a DataFrame whose columns use the send CSV's stat names."""
    col = lambda k: df[k].fillna(0.0).astype(float) if k in df.columns else 0.0
    pts = sum(col(k) * w for k, w in LINEAR.items())
    for k in BONUS_AT:
        pts = pts + BONUS_PTS * p_bonus(k, col(k))
    return pts


def actual_frame(a):
    """Actual DraftKings points from an nflverse stats_player_week frame.

    Includes what DraftKings pays that a projection line does not carry: two-point conversions,
    return and fumble-recovery touchdowns, and every fumble lost (sacks included).
    """
    z = lambda c: a[c].fillna(0.0).astype(float) if c in a.columns else 0.0
    fl = z("fumbles_lost_total") if "fumbles_lost_total" in a.columns else \
        z("sack_fumbles_lost") + z("rushing_fumbles_lost") + z("receiving_fumbles_lost")
    pts = (0.04 * z("passing_yards") + 4 * z("passing_tds") - z("passing_interceptions")
           + 0.1 * z("rushing_yards") + 6 * z("rushing_tds")
           + z("receptions") + 0.1 * z("receiving_yards") + 6 * z("receiving_tds")
           + 2 * (z("passing_2pt_conversions") + z("rushing_2pt_conversions") + z("receiving_2pt_conversions"))
           + 6 * (z("special_teams_tds") + z("fumble_recovery_tds"))
           - fl)
    pts = pts + BONUS_PTS * ((z("passing_yards") >= 300).astype(float) + (z("rushing_yards") >= 100).astype(float)
                             + (z("receiving_yards") >= 100).astype(float))
    return pts
