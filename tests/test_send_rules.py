"""When a send counts: T-75 before the 2026-09-27 4:25 PM ET slate, T-55 from it (sends moved to T-60), T-50 from
the 2026-10-25 London game (the T-55 tick became a fallback)."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import send_rules as R


def test_the_rule_changes_at_the_cutover():
    assert R.min_lead(pd.Timestamp("2026-09-27T20:05Z")) == 75.0     # the 4:05 slate, sent at T-80
    assert R.min_lead(pd.Timestamp("2026-09-27T20:25Z")) == 55.0     # the first T-60 slate
    assert R.min_lead(pd.Timestamp("2026-09-27T16:25-04:00")) == 55.0   # any timezone


def test_earlier_reruns_stay_ineligible():
    """A T-68 resend in week 1 was never counted; the new rule must not start counting it."""
    k = pd.Series(pd.to_datetime(["2026-09-13T17:00Z", "2026-09-27T20:25Z", "2026-09-27T20:25Z"], utc=True))
    assert R.eligible(k, pd.Series([68.0, 60.0, 50.0])).tolist() == [False, True, False]


def test_a_command_line_threshold_overrides_the_dates():
    k = pd.Series(pd.to_datetime(["2026-09-13T17:00Z"], utc=True))
    assert R.eligible(k, pd.Series([68.0]), override=60.0).tolist() == [True]


def test_the_send_clock_follows_the_cutover():
    """The Upcoming sends page tells the owner to run a slate by hand if its email hasn't arrived by
    email_by. After the cutover that has to be T-57, or he forces a send 20 minutes early and the T-60
    run with the inactives is skipped."""
    assert R.clock(pd.Timestamp("2026-09-27T20:05Z")) == {"send_at": 80, "email_by": 77, "deadline": 75.0}
    assert R.clock(pd.Timestamp("2026-09-27T20:25Z")) == {"send_at": 60, "email_by": 57, "deadline": 55.0}


def test_the_exported_rule_describes_both_sides():
    t = R.rule_text()
    assert ">= 50" in t and ">= 55" in t and ">= 75" in t and "2026-09-27" in t and "2026-10-25" in t
    assert R.rule_text(60) == "latest send generated >= 60 min before kickoff"


def test_the_t55_fallback_counts_from_the_london_game():
    """The T-55 tick generates at ~T-54. It counts from 2026-10-25; earlier games keep the 55-minute rule, so
    their grades do not move."""
    assert R.min_lead(pd.Timestamp("2026-10-23T00:15Z")) == 55.0     # TNF wk7, before the fallback
    assert R.min_lead(pd.Timestamp("2026-10-25T13:30Z")) == 50.0     # PIT @ NO, London: the first one
    k = pd.Series(pd.to_datetime(["2026-10-23T00:15Z", "2026-10-25T17:00Z", "2026-10-25T17:00Z"], utc=True))
    assert R.eligible(k, pd.Series([53.8, 53.8, 49.0])).tolist() == [False, True, False]
    assert R.clock(pd.Timestamp("2026-10-25T17:00Z")) == {"send_at": 60, "email_by": 57, "deadline": 50.0}
