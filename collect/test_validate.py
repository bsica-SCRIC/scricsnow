"""Tests for the commit gate.

Two failure modes, and they are not equally bad. Letting a broken file through
puts a traceback on a public page. Failing on an ordinary night trains everyone
to ignore the alarm, so that by February a real failure is another red tick
nobody opens. Most of what is pinned here is the second kind: things that must
NOT fire.
"""

import pandas as pd

import validate as V

GOOD = pd.DataFrame([
    {"date": "2026-01-15", "leaid": "3604870", "status": "closed"},
    {"date": "2026-01-15", "leaid": "3602400", "status": "delay"},
])
KNOWN = {"3604870", "3602400", "3615900"}


def test_a_good_frame_passes():
    assert V.check(GOOD, "outcomes", required=["date", "leaid", "status"],
                   known_ids=KNOWN) == []


def test_a_missing_column_is_caught_and_named():
    problems = V.check(GOOD.drop(columns=["status"]), "outcomes",
                       required=["date", "leaid", "status"])
    assert len(problems) == 1
    assert "status" in problems[0]


def test_a_missing_column_stops_the_other_checks():
    """Otherwise one dropped column reports as five problems and the real one
    is buried."""
    problems = V.check(pd.DataFrame([{"nonsense": 1}]), "outcomes",
                       required=["date", "leaid"], known_ids=KNOWN)
    assert len(problems) == 1


def test_a_row_with_no_leaid_is_caught():
    bad = pd.concat([GOOD, pd.DataFrame([{"date": "2026-01-15", "leaid": None,
                                          "status": "closed"}])])
    problems = V.check(bad, "outcomes", required=["date", "leaid"])
    assert any("no leaid" in p for p in problems)


def test_a_leaid_off_the_roster_is_caught_and_the_message_says_why():
    """This is nearly always a spreadsheet eating a leading zero, so the
    message points at that rather than making somebody rediscover it."""
    bad = pd.DataFrame([{"date": "2026-01-15", "leaid": "999999"}])
    problems = V.check(bad, "outcomes", required=["date", "leaid"],
                       known_ids=KNOWN)
    assert any("leading zero" in p for p in problems)


def test_an_unparsable_date_is_caught_with_an_example():
    bad = pd.DataFrame([{"date": "15/01/2026 maybe", "leaid": "3604870"}])
    problems = V.check(bad, "outcomes", required=["date", "leaid"],
                       known_ids=KNOWN)
    assert any("unparsable date" in p for p in problems)
    assert any("maybe" in p for p in problems)


def test_an_empty_frame_is_caught():
    problems = V.check(pd.DataFrame(columns=["date", "leaid"]), "outcomes",
                       required=["date", "leaid"])
    assert any("0 row" in p for p in problems)


def test_none_is_distinguished_from_empty():
    """A job that produced no frame at all failed differently from one that
    produced an empty one, and the message has to say which."""
    problems = V.check(None, "outcomes")
    assert any("nothing to write at all" in p for p in problems)


# --------------------------------------------------------------------------
# What must NOT fire
# --------------------------------------------------------------------------

def test_a_partial_night_is_not_a_problem():
    """nws.snapshot is built to write 47 of 50 districts rather than nothing.
    Failing here would throw away rows that cannot be recollected."""
    partial = pd.DataFrame([{"date": "2026-01-15", "leaid": "3604870"}])
    assert V.check(partial, "forecast", required=["date", "leaid"],
                   known_ids=KNOWN) == []


def test_a_frame_with_no_date_column_is_not_forced_to_have_one():
    """Not every file this gates is dated."""
    assert V.check(pd.DataFrame([{"leaid": "3604870", "days_built_in": 4}]),
                   "bank", required=["leaid"], known_ids=KNOWN) == []


def test_an_unknown_roster_is_not_treated_as_every_id_being_wrong():
    """A missing roster file must not turn every row into a violation. The
    gate would fail every job on every run and be switched off within a week.
    """
    assert V.check(GOOD, "outcomes", required=["date", "leaid"],
                   known_ids=set()) == []


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_no_problems_means_exit_zero():
    assert V.report([]) == 0


def test_problems_mean_exit_non_zero(capsys):
    code = V.report(["outcomes: something is wrong"])
    assert code == 1
    assert "::error::" in capsys.readouterr().out


def test_gate_is_check_and_report_together(capsys):
    assert V.gate(GOOD, "outcomes", required=["date", "leaid"],
                  known_ids=KNOWN) == 0
    assert V.gate(None, "outcomes") == 1


def test_the_real_roster_loads():
    ids = V.roster_ids()
    assert len(ids) >= 50
    assert all(isinstance(i, str) and len(i) == 7 for i in ids)
