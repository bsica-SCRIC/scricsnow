"""Is school in session?

Reads reference/school_calendar.csv, the aggregate no-school calendar. Without
this the site will cheerfully offer a 62% chance of a snow day for Christmas,
which is worse than saying nothing.

WE ONLY CALL IT THE NIGHT BEFORE. The model was trained on roughly a
twelve-hour lead time and its published accuracy is measured at that lead. A
three-day-out number would be a different thing entirely, scored as if it were
the same, and the accuracy record is the only reason to believe any of this. So
on a Friday evening this reports Monday as the next school day and declines to
put a number on it, rather than quietly serving a forecast the model was never
built for.

Fallbacks, in order: weekends are always off; a date in the calendar file is
off; anything else is a school day. If a date falls beyond the last season in
the file, only the weekend rule applies, which fails toward showing a
prediction rather than wrongly hiding one.
"""

import os
from datetime import date, timedelta

import pandas as pd

CALENDAR_CSV = os.path.join(os.path.dirname(__file__), "..", "reference",
                            "school_calendar.csv")

# Summer. The calendar file lists exception days WITHIN a school year, so the
# months when there is no school year at all appear nowhere in it and would
# otherwise read as ordinary school days. Without this the site offers a snow
# day probability for a Tuesday in August.
#
# July and August, plus the start of September up to Labor Day. New York
# districts almost universally start the day after Labor Day, so without that
# last part the site announces that the next school day is 1 September when it
# is really the 8th. June stays in session; the calendar handles its final days.
SUMMER_MONTHS = {7, 8}

# How far ahead we are willing to put a number on it. One means tomorrow.
MAX_LEAD_DAYS = 1

# Give up rather than loop forever if the calendar is nonsense.
SEARCH_LIMIT = 60


def load(path=CALENDAR_CSV):
    """date -> reason, for every weekday most districts have off."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {pd.Timestamp(d).date(): r
            for d, r in zip(df["date"], df["reason"])}


def covered_through(calendar):
    """Last date the calendar knows about, or None."""
    return max(calendar) if calendar else None


def labor_day(year):
    """First Monday in September."""
    d = date(year, 9, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def reason_off(day, calendar):
    """Why there is no school, or None if there is."""
    if day.month in SUMMER_MONTHS:
        return "summer"
    if day.month == 9 and day < labor_day(day.year):
        return "summer"
    if day.weekday() >= 5:
        return "weekend"
    return calendar.get(day)


def is_school_day(day, calendar):
    return reason_off(day, calendar) is None


def next_school_day(after, calendar, limit=SEARCH_LIMIT):
    """First school day on or after `after`."""
    day = after
    for _ in range(limit):
        if is_school_day(day, calendar):
            return day
        day += timedelta(days=1)
    return None


def target_for(now, calendar, morning_before_hour=12, max_lead=MAX_LEAD_DAYS):
    """What day to talk about, and whether to put a number on it.

    Returns (target, predictable, reason_off, lead_days).

      target      the day being discussed -- the next school day
      predictable True only if that day is within max_lead
      reason_off  why the immediate next day has no school, or None
      lead_days   days from now until target
    """
    today = now.date()
    # A morning run is about today; any later run is about tomorrow.
    immediate = today if now.hour < morning_before_hour else today + timedelta(days=1)

    why = reason_off(immediate, calendar)
    if why is None:
        return immediate, True, None, (immediate - today).days

    target = next_school_day(immediate, calendar)
    if target is None:
        return immediate, False, why, 0

    lead = (target - today).days
    return target, lead <= max_lead, why, lead


def describe(reason, target):
    """Plain wording for why tonight has no number."""
    when = target.strftime("%A %d %B").replace(" 0", " ") if target else "soon"
    if reason == "weekend":
        return f"It is the weekend. Next school day is {when}."
    if reason == "break":
        return f"School is on break. Next school day is {when}."
    if reason == "holiday":
        return f"It is a holiday. Next school day is {when}."
    if reason == "summer":
        return "It is summer. Come back when the leaves turn."
    return f"No school tomorrow. Next school day is {when}."


if __name__ == "__main__":
    from datetime import datetime

    cal = load()
    print(f"calendar entries: {len(cal)}   through {covered_through(cal)}")
    print()
    for label, when in [
        ("a normal Tuesday evening", datetime(2027, 1, 12, 20, 0)),
        ("Friday evening", datetime(2027, 1, 15, 20, 0)),
        ("Christmas Eve evening", datetime(2026, 12, 23, 20, 0)),
        ("mid winter break", datetime(2026, 12, 29, 20, 0)),
        ("Sunday evening", datetime(2027, 1, 17, 20, 0)),
    ]:
        t, ok, why, lead = target_for(when, cal)
        verdict = f"predict {t}" if ok else f"no number ({why}), next is {t}"
        print(f"  {label:26} {when:%a %d %b %H:%M} -> {verdict}  lead {lead}d")
