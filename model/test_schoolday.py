"""Tests for the school-in-session check.

The failure this prevents: offering a 62% chance of a snow day for Christmas.
Beyond looking foolish, it would enter the accuracy record as a free correct
"open" for a morning nobody was going to school anyway.
"""

from datetime import date, datetime

import pytest

import schoolday as SD

CAL = {
    date(2026, 12, 24): "break",
    date(2026, 12, 25): "break",
    date(2026, 12, 28): "break",
    date(2026, 12, 29): "break",
    date(2027, 1, 1): "break",
    date(2027, 1, 18): "holiday",
}


def test_a_normal_weekday_is_a_school_day():
    assert SD.is_school_day(date(2027, 1, 12), CAL)


def test_weekends_are_always_off_even_with_an_empty_calendar():
    assert SD.reason_off(date(2027, 1, 16), {}) == "weekend"   # Saturday
    assert SD.reason_off(date(2027, 1, 17), {}) == "weekend"   # Sunday


def test_calendar_days_are_off():
    assert SD.reason_off(date(2026, 12, 25), CAL) == "break"
    assert SD.reason_off(date(2027, 1, 18), CAL) == "holiday"


def test_next_school_day_skips_a_whole_break():
    assert SD.next_school_day(date(2026, 12, 24), CAL) == date(2026, 12, 30)


def test_next_school_day_skips_a_holiday_that_follows_a_weekend():
    """The MLK case: Friday night, and the next school day is Tuesday."""
    assert SD.next_school_day(date(2027, 1, 16), CAL) == date(2027, 1, 19)


def test_tomorrow_is_predictable_on_an_ordinary_evening():
    target, ok, why, lead = SD.target_for(datetime(2027, 1, 12, 20, 0), CAL)
    assert (target, ok, why, lead) == (date(2027, 1, 13), True, None, 1)


def test_a_morning_run_is_about_today_not_tomorrow():
    target, ok, _, lead = SD.target_for(datetime(2027, 1, 13, 5, 0), CAL)
    assert target == date(2027, 1, 13) and ok and lead == 0


def test_the_page_rolls_over_at_the_end_of_the_school_day():
    """Until 4pm the page is still about today, so a district that closed this
    morning keeps reading as closed through the afternoon."""
    target, _, _, _ = SD.target_for(datetime(2027, 1, 13, 15, 59), CAL)
    assert target == date(2027, 1, 13)
    target, _, _, _ = SD.target_for(datetime(2027, 1, 13, 16, 0), CAL)
    assert target == date(2027, 1, 14)


def test_friday_evening_gets_no_number():
    """We only call it the night before. Monday from Friday is three days out,
    far outside the lead time the accuracy record is measured at."""
    target, ok, why, lead = SD.target_for(datetime(2027, 1, 15, 20, 0), CAL)
    assert why == "weekend"
    assert ok is False
    assert lead > SD.MAX_LEAD_DAYS


def test_sunday_evening_predicts_monday():
    cal = {}                       # no holidays, so Monday is a school day
    target, ok, why, lead = SD.target_for(datetime(2027, 1, 17, 20, 0), cal)
    assert target == date(2027, 1, 18) and ok and lead == 1


def test_christmas_eve_offers_no_number():
    target, ok, why, _ = SD.target_for(datetime(2026, 12, 24, 20, 0), CAL)
    assert why == "break"
    assert ok is False
    assert target == date(2026, 12, 30)


def test_a_date_beyond_the_calendar_falls_back_to_weekends_only():
    """Failing toward showing a prediction is safer than wrongly hiding one."""
    assert SD.is_school_day(date(2030, 3, 6), CAL)          # a Wednesday
    assert not SD.is_school_day(date(2030, 3, 9), CAL)      # a Saturday


def test_next_school_day_gives_up_rather_than_looping_forever():
    every_day_off = {}
    always = type("cal", (), {"get": lambda self, k, d=None: "break"})()
    assert SD.next_school_day(date(2027, 1, 4), always, limit=5) is None


@pytest.mark.parametrize("reason", ["weekend", "break", "holiday", None])
def test_describe_always_says_something_useful(reason):
    text = SD.describe(reason, date(2027, 1, 4))
    assert "Next school day" in text


def test_the_shipped_calendar_loads_and_covers_this_winter():
    cal = SD.load()
    assert len(cal) > 100
    assert SD.covered_through(cal) >= date(2027, 4, 1)
    assert SD.reason_off(date(2026, 12, 25), cal) is not None


def test_summer_is_never_a_school_day():
    """The calendar lists exception days WITHIN a school year, so July and
    August appear nowhere in it and would otherwise look like ordinary school
    days. This is what stops a snow day probability for a Tuesday in August."""
    assert SD.reason_off(date(2026, 8, 25), CAL) == "summer"   # a Tuesday
    assert SD.reason_off(date(2026, 7, 8), CAL) == "summer"
    assert not SD.is_school_day(date(2026, 8, 28), CAL)


def test_the_school_year_months_are_not_summer():
    assert SD.reason_off(date(2026, 9, 15), CAL) is None
    assert SD.reason_off(date(2027, 6, 8), CAL) is None


def test_summer_has_its_own_wording():
    assert "summer" in SD.describe("summer", date(2026, 9, 8)).lower()


def test_september_before_labor_day_is_still_summer():
    """NY districts start the day after Labor Day. Without this the page
    announces the next school day is 1 September when it is really the 8th."""
    assert SD.labor_day(2026) == date(2026, 9, 7)
    assert SD.reason_off(date(2026, 9, 1), CAL) == "summer"   # a Tuesday
    assert SD.reason_off(date(2026, 9, 4), CAL) == "summer"   # a Friday
    assert SD.reason_off(date(2026, 9, 8), CAL) is None       # back to school


@pytest.mark.parametrize("year,expected", [
    (2026, date(2026, 9, 7)), (2027, date(2027, 9, 6)),
    (2028, date(2028, 9, 4)), (2029, date(2029, 9, 3)),
])
def test_labor_day_lands_on_the_first_monday(year, expected):
    assert SD.labor_day(year) == expected
    assert expected.weekday() == 0
