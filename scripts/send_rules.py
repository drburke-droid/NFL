"""When a SaberSim send counts, for every grader (sabersim_grade, fan_grade, sabersim_scenarios).

Until 2026-09-27 the send went at T-80 so its email reached SaberSim inside their T-75 cutoff, and the
graders counted only sends generated at least 75 minutes before kickoff. The owner no longer emails
SaberSim (2026-09-24), so from the 4:25 PM ET slate of 2026-09-27 the send goes at T-60, after the
inactives have reached the feeds (they are released at T-90 and the feeds carried them by ~T-73 on
2026-09-13), and a send counts if it was generated at least 55 minutes before kickoff.

Games before the cutover keep the 75-minute rule, so earlier weeks' grades do not move: several of them
have later reruns (a T-68 resend on 2026-09-13) that were never eligible and must stay that way.
"""
import pandas as pd

CUTOVER = pd.Timestamp("2026-09-27T20:10:00Z")     # the first kickoff after it: 4:25 PM ET, 2026-09-27
LEAD_BEFORE = 75.0
LEAD_AFTER = 55.0


def min_lead(kick):
    """Minutes before kickoff a send must be generated to count, for a kickoff Timestamp or a Series of them."""
    if isinstance(kick, pd.Series):
        k = pd.to_datetime(kick, utc=True)
        return k.map(lambda t: LEAD_AFTER if t >= CUTOVER else LEAD_BEFORE).astype(float)
    return LEAD_AFTER if pd.Timestamp(kick).tz_convert("UTC") >= CUTOVER else LEAD_BEFORE


def eligible(kick, lead_min, override=None):
    """Boolean(s): generated early enough. `override` (a --min-lead given on the command line) applies one
    threshold to every game instead of the dated rule."""
    return lead_min >= (override if override is not None else min_lead(kick))
