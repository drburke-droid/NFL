"""When a SaberSim send counts, for every grader (sabersim_grade, fan_grade, sabersim_scenarios).

Until 2026-09-27 the send went at T-80 so its email reached SaberSim inside their T-75 cutoff, and the
graders counted only sends generated at least 75 minutes before kickoff. The owner no longer emails
SaberSim (2026-09-24), so from the 4:25 PM ET slate of 2026-09-27 the send goes at T-60, after the
inactives have reached the feeds (they are released at T-90 and the feeds carried them by ~T-73 on
2026-09-13), and a send counts if it was generated at least 55 minutes before kickoff.

From the 2026-10-25 London game the T-55 tick is a fallback when the T-60 one did not send (on 2026-10-19 the
T-60 tick never arrived and MNF went unsent), so a send counts if it was generated at least 50 minutes before
kickoff: the fallback generates at ~T-54, which the 55-minute rule would have thrown away.

Games before the cutover keep the 75-minute rule, so earlier weeks' grades do not move: several of them
have later reruns (a T-68 resend on 2026-09-13) that were never eligible and must stay that way.
"""
import pandas as pd

CUTOVER = pd.Timestamp("2026-09-27T20:10:00Z")     # the first kickoff after it: 4:25 PM ET, 2026-09-27
LEAD_BEFORE = 75.0
LEAD_AFTER = 55.0
CUTOVER_FALLBACK = pd.Timestamp("2026-10-25T13:30:00Z")   # the first kickoff with the T-55 fallback: PIT @ NO, London
LEAD_FALLBACK = 50.0


def _lead(t):
    return LEAD_BEFORE if t < CUTOVER else LEAD_AFTER if t < CUTOVER_FALLBACK else LEAD_FALLBACK


def min_lead(kick):
    """Minutes before kickoff a send must be generated to count, for a kickoff Timestamp or a Series of them."""
    if isinstance(kick, pd.Series):
        k = pd.to_datetime(kick, utc=True)
        return k.map(_lead).astype(float)
    return _lead(pd.Timestamp(kick).tz_convert("UTC"))


def eligible(kick, lead_min, override=None):
    """Boolean(s): generated early enough. `override` (a --min-lead given on the command line) applies one
    threshold to every game instead of the dated rule."""
    return lead_min >= (override if override is not None else min_lead(kick))


# The send clock around each kickoff, in minutes before it: when the automatic send starts, when its email
# should be in the inbox (the page tells the owner to run the slate by hand if it is not), and the last
# generation time that still counts. T-80/T-77/T-75 before the cutover (SaberSim's T-75 cutoff), T-60/T-57/T-55 after,
# T-60/T-57/T-50 once the T-55 tick is a fallback (the email stays due by T-57; the fallback is the late case).
CLOCK_BEFORE = {"send_at": 80, "email_by": 77, "deadline": LEAD_BEFORE}
CLOCK_AFTER = {"send_at": 60, "email_by": 57, "deadline": LEAD_AFTER}
CLOCK_FALLBACK = {"send_at": 60, "email_by": 57, "deadline": LEAD_FALLBACK}


def clock(kick):
    """{send_at, email_by, deadline} minutes before this kickoff."""
    t = pd.Timestamp(kick).tz_convert("UTC")
    return CLOCK_BEFORE if t < CUTOVER else CLOCK_AFTER if t < CUTOVER_FALLBACK else CLOCK_FALLBACK


def rule_text(override=None):
    """How the graded send is chosen, in words, for the accuracy JSON and report."""
    if override is not None:
        return f"latest send generated >= {override:g} min before kickoff"
    return (f"latest send generated >= {LEAD_FALLBACK:g} min before kickoff (>= {LEAD_AFTER:g} for kickoffs before "
            f"{CUTOVER_FALLBACK.strftime('%Y-%m-%d %H:%MZ')}, when the T-55 fallback began; >= {LEAD_BEFORE:g} before "
            f"{CUTOVER.strftime('%Y-%m-%d %H:%MZ')}, when the send moved from T-80 to T-60)")
