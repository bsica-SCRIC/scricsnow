"""Tests for the emergency-day bank burndown.

The arithmetic is small but it is subtracting one number from another and
drawing the result, so the interesting cases are all at or past zero. A bank
that goes negative is normal in a hard winter, and the display has to survive
it without asking for a negative snowflake.
"""

import pandas as pd
import pytest

import bank


def outcomes(rows):
    return pd.DataFrame(rows, columns=["leaid", "date", "status"])


# --- what spends a day ------------------------------------------------------

def test_only_closures_spend_a_day():
    """A delay or an early dismissal still holds a session, so the bank is
    untouched. Getting this wrong burns the bank down roughly twice as fast."""
    o = outcomes([("3600001", "2026-01-05", "closed"),
                  ("3600001", "2026-01-06", "delay"),
                  ("3600001", "2026-01-07", "early")])
    assert bank.spent(o, "3600001") == 1


def test_remote_days_do_not_spend_a_day():
    """Deliberate and documented: whether remote counts toward the 180 is
    unsettled in NY, and under-counting is the safer of the two errors."""
    o = outcomes([("3600001", "2026-01-05", "remote"),
                  ("3600001", "2026-01-06", "closed")])
    assert bank.spent(o, "3600001") == 1


def test_the_same_day_twice_spends_one_day():
    """The feed is archived several times a morning and a district can be
    restated. Counting rows instead of dates would double-charge them."""
    o = outcomes([("3600001", "2026-01-05", "closed"),
                  ("3600001", "2026-01-05", "closed")])
    assert bank.spent(o, "3600001") == 1


def test_other_districts_do_not_count():
    o = outcomes([("3600002", "2026-01-05", "closed")])
    assert bank.spent(o, "3600001") == 0


def test_empty_season_spends_nothing():
    assert bank.spent(outcomes([]), "3600001") == 0
    assert bank.spent(pd.DataFrame(), "3600001") == 0


# --- the pictogram ----------------------------------------------------------

def test_row_keeps_the_bank_width_until_it_is_blown():
    """Unspent plus spent is the bank, so the row does not shrink as days go
    and the size of the bank stays readable in March."""
    for used in range(0, 5):
        unspent, gone, makeup = bank.pictogram(4, used)
        assert unspent + gone == 4
        assert makeup == 0


def test_exactly_spent_shows_a_full_faded_row():
    assert bank.pictogram(4, 4) == (0, 4, 0)


def test_overspending_adds_makeup_marks():
    """The case this exists for. Four faded snowflakes and two calendar days,
    not minus two snowflakes."""
    assert bank.pictogram(4, 6) == (0, 4, 2)


def test_nothing_in_the_pictogram_can_go_negative():
    """Whatever the inputs, these three numbers are drawn as repeat counts,
    so a negative would raise or silently render nothing."""
    for built, used in [(0, 0), (0, 3), (4, 99), (-1, 2), (4, -2)]:
        marks = bank.pictogram(built, used)
        assert all(m >= 0 for m in marks), marks


# --- the caption ------------------------------------------------------------

def test_caption_states_the_position_in_each_regime():
    assert bank.caption(4, 1) == "3 left of 4 built into the calendar"
    assert bank.caption(4, 4) == "none left of 4 built into the calendar"
    assert "2 to make up" in bank.caption(4, 6)


def test_caption_does_not_alarm():
    """Overspending is routine and the district handles it by adding days.
    The wording stays flat on purpose."""
    text = bank.caption(4, 7).lower()
    assert not any(w in text for w in ["!", "warning", "over budget", "risk"])


# --- seasons and files ------------------------------------------------------

def test_season_boundary_is_july():
    """A January closure belongs to the season that began the previous July.
    Everything that touches a season file uses this rule and must agree."""
    assert bank.season_of("2026-01-15") == 2025
    assert bank.season_of("2026-06-30") == 2025
    assert bank.season_of("2026-07-01") == 2026


def test_missing_season_file_reads_as_no_closures(tmp_path):
    """Pre-season is the normal state for most of the year and is not an
    error, though it looks the same as a collector that never ran."""
    o = bank.load_outcomes(2026, out_dir=str(tmp_path))
    assert len(o) == 0
    assert bank.spent(o, "3600001") == 0


def test_season_file_is_read_when_present(tmp_path):
    outcomes([("3600001", "2026-01-05", "closed"),
              ("3600001", "2026-01-06", "closed")]).to_csv(
        tmp_path / "outcomes_2025.csv", index=False)
    o = bank.load_outcomes(2025, out_dir=str(tmp_path))
    assert bank.spent(o, "3600001") == 2


def test_leaid_keeps_its_leading_zero_through_the_file(tmp_path):
    """leaid is a string everywhere. If pandas infers it as an integer the
    join against the roster silently matches nothing and the burndown is
    always zero, which looks like a quiet season rather than a bug."""
    outcomes([("0600001", "2026-01-05", "closed")]).to_csv(
        tmp_path / "outcomes_2025.csv", index=False)
    o = bank.load_outcomes(2025, out_dir=str(tmp_path))
    assert o["leaid"].iloc[0] == "0600001"
    assert bank.spent(o, "0600001") == 1


def test_missing_bank_file_reads_as_empty(tmp_path):
    b = bank.load(path=str(tmp_path / "nope.csv"))
    assert len(b) == 0
    assert "days_built_in" in b.columns
