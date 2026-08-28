"""Tests for the public accuracy record.

This is the number the site shows to justify believing it, so the ways it could
flatter itself matter more than the ways it could crash. It cannot be exercised
against real data until December, so the behaviour is pinned here.
"""

import pandas as pd
import pytest

import scorecard as S

ROSTER = pd.DataFrame({
    "leaid": ["3604870", "3602400"],
    "district": ["Binghamton City School District",
                 "Afton Central School District"],
    "elev_m": [261.0, 469.0],
})


def calls(rows):
    """The frozen 8pm record: (date, leaid, p_closed)."""
    return pd.DataFrame([
        {"target_date": d, "leaid": l, "district": "Binghamton",
         "p_closed": p, "p_disrupted": min(1.0, p + 0.1), "snow_in": 2.0,
         "called_at": f"{d}T20:00:00-05:00",
         "model_trained_at": "2026-08-27T00:00:00+00:00"}
        for d, l, p in rows])


def checks(dates, ok=True):
    return pd.DataFrame([{"date": d, "ok": ok, "records": 1} for d in dates])


def outcomes(rows):
    return pd.DataFrame([{"date": d, "leaid": l, "status": s}
                         for d, l, s in rows],
                        columns=["date", "leaid", "status"])


def test_a_morning_we_never_checked_is_not_scored():
    """The flattering bug this guards against: with no checks log, an empty
    outcomes file looks like 'nobody closed', and the tool would award itself
    fifty correct 'open' calls for a morning it never observed."""
    card = S.build(calls([("2026-01-15", "3604870", 0.9)]), outcomes([]),
                   checks([]), ROSTER)
    assert card.empty


def test_a_checked_morning_with_no_closures_scores_as_open():
    card = S.build(calls([("2026-01-15", "3604870", 0.01)]), outcomes([]),
                   checks(["2026-01-15"]), ROSTER)
    assert len(card) == 1
    assert bool(card["actual_closed"].iloc[0]) is False
    assert bool(card["correct"].iloc[0]) is True


def test_a_recorded_closure_is_scored_as_closed():
    card = S.build(calls([("2026-01-15", "3604870", 0.9)]),
                   outcomes([("2026-01-15", "3604870", "closed")]),
                   checks(["2026-01-15"]), ROSTER)
    assert bool(card["actual_closed"].iloc[0]) is True
    assert bool(card["correct"].iloc[0]) is True


def test_the_stored_probability_is_used_verbatim():
    """The integrity rule. Scoring must read the frozen number, never
    recompute one, or retraining would rewrite the published record."""
    card = S.build(calls([("2026-01-15", "3604870", 0.37)]), outcomes([]),
                   checks(["2026-01-15"]), ROSTER)
    assert card["p_closed"].iloc[0] == pytest.approx(0.37)


def test_summary_counts_hits_misses_and_false_alarms():
    card = pd.DataFrame({
        "target_date": ["d1", "d1", "d2", "d2"],
        "p_closed": [0.9, 0.1, 0.8, 0.2],
        "actual_closed": [True, True, False, False],
    })
    s = S.summarize(card)
    assert s["closures_called"] == 1     # 0.9 and closed
    assert s["closures_missed"] == 1     # 0.1 and closed
    assert s["false_alarms"] == 1        # 0.8 and open
    assert s["closures_seen"] == 2


def test_skill_is_measured_against_always_saying_open():
    """On a 3% base rate, plain accuracy looks superb for a tool that always
    says open. Skill against that baseline is the honest headline."""
    good = pd.DataFrame({"target_date": ["d"] * 4,
                         "p_closed": [0.95, 0.02, 0.9, 0.03],
                         "actual_closed": [True, False, True, False]})
    s = S.summarize(good)
    assert s["brier"] < s["brier_if_we_always_said_open"]
    assert s["skill_vs_always_open"] > 0.9


def test_a_useless_forecaster_shows_negative_skill():
    """Hedging at 50% every morning is worse than useless at this base rate.

    The fixture matters: closures run about 3% of school mornings, so a
    constant 0.5 scores far worse than simply always saying open. At a 50/50
    base rate the same forecaster would look skilful, which is exactly why the
    baseline has to be the real one.
    """
    n = 100
    actual = [True] * 3 + [False] * (n - 3)
    useless = pd.DataFrame({"target_date": ["d"] * n,
                            "p_closed": [0.5] * n,
                            "actual_closed": actual})
    s = S.summarize(useless)
    assert s["skill_vs_always_open"] < 0


def test_calibration_table_compares_claimed_against_observed():
    card = pd.DataFrame({
        "p_closed": [0.02, 0.03, 0.8, 0.85],
        "actual_closed": [False, False, True, False],
    })
    cal = S.calibration(card)
    assert {"mornings", "we_said", "actually_closed"} <= set(cal.columns)
    assert cal["mornings"].sum() == 4


def test_empty_inputs_never_raise():
    empty = pd.DataFrame()
    assert S.build(empty, empty, empty, ROSTER).empty
    assert S.build(None, empty, empty, ROSTER).empty
    assert S.summarize(pd.DataFrame())["scored_days"] == 0
    assert S.calibration(pd.DataFrame()).empty


def test_lead_time_needs_a_morning_capture_before_it_reports():
    got = S.lead_time_value(calls([("2026-01-15", "3604870", 0.2)]),
                            pd.DataFrame(), outcomes([]),
                            checks(["2026-01-15"]), ROSTER)
    assert got is None


def _calls(**kw):
    base = {"target_date": "2026-01-15", "leaid": "3604870",
            "district": "Binghamton", "p_closed": 0.8, "p_disrupted": 0.9,
            "snow_in": 6.0}
    base.update(kw)
    return pd.DataFrame([base])


def _checks(date="2026-01-15"):
    return pd.DataFrame([{"date": date, "ok": True}])


def _outcomes(status):
    return pd.DataFrame([{"date": "2026-01-15", "leaid": "3604870",
                          "status": status}])


def test_a_delay_is_not_counted_as_a_missed_closure():
    """The bug this replaced.

    actual_closed used to be closed-OR-delay-OR-remote while p_closed is
    trained on full closures only, so every delay landed on the published
    record as a closure the model had failed to predict. Delays are about a
    quarter of all disruptions, so the record would have understated the model
    from the first storm of the season and the calibration table would have
    shown it as under-confident everywhere.
    """
    card = S.build(_calls(), _outcomes("delay"), _checks())
    assert not bool(card["actual_closed"].iloc[0])
    # It is still a disruption, and the head that predicts disruptions is
    # scored against it.
    assert bool(card["actual_disrupted"].iloc[0])


def test_a_closure_counts_for_both_heads():
    card = S.build(_calls(), _outcomes("closed"), _checks())
    assert bool(card["actual_closed"].iloc[0])
    assert bool(card["actual_disrupted"].iloc[0])


def test_an_early_dismissal_scores_as_neither():
    """An early dismissal is decided mid-morning. What gets scored here is the
    overnight call, so it is not an outcome this tool claimed to predict."""
    card = S.build(_calls(), _outcomes("early"), _checks())
    assert not bool(card["actual_closed"].iloc[0])
    assert not bool(card["actual_disrupted"].iloc[0])


def test_summary_reports_the_two_heads_separately():
    card = S.build(_calls(), _outcomes("delay"), _checks())
    s = S.summarize(card)
    assert s["closures_seen"] == 0
    assert s["disruptions_seen"] == 1
    # The closure head said 80% on a morning that did not close: that is a
    # false alarm for closures and a hit for disruptions.
    assert s["false_alarms"] == 1
    assert s["disruptions_called"] == 1
