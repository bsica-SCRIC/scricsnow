"""The emergency-day bank and its burndown.

The bank is a budget, not a count. A district builds in some number of closure
days above New York's 180 instructional days, spends them over the winter, and
in a bad year overspends and makes the difference up at the end of the year.

reference/district_bank.csv holds the budget, hand-maintained, one number per
district. This module counts what has been spent against it and works out how
the result should be drawn.

Drawing is the part with a wrinkle. A bank can go negative and there is no such
thing as a negative snowflake, so the row carries two different marks: unspent
and spent days are snowflakes, and days past the bank are calendar marks,
because that is literally what they become, days added back to the calendar in
June. Spent days stay on the row rather than disappearing, so the size of the
bank is still readable in March when most of it is gone.
"""

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
BANK_CSV = os.path.join(ROOT, "reference", "district_bank.csv")
OUTCOMES_DIR = os.path.join(ROOT, "data", "outcomes")

# Statuses that spend a day from the bank.
#
# Only a full closure does. A delay or an early dismissal still holds a session
# and costs nothing. `remote` is deliberately excluded: whether a remote day
# counts toward the 180 has moved around in NY since 2022 and has never been
# pinned down here. Excluding it means that if districts do start taking
# remote days, the burndown under-counts rather than inventing days a parent
# would then not have to make up.
SPENDS_A_DAY = {"closed"}


def season_of(d):
    """Seasons run July to June, so a January closure belongs to the year
    before it. Everything that writes or reads a season file uses this same
    boundary, and they have to agree or the burndown reads the wrong file."""
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


def load(path=BANK_CSV):
    """The budget worksheet, or an empty frame if it has not been built."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["leaid", "district", "days_built_in"])
    return pd.read_csv(path, dtype={"leaid": str})


def load_outcomes(season, out_dir=OUTCOMES_DIR):
    """Recorded closings for one season.

    A missing file means the season has not started yet, which is a real zero
    rather than an error. It is worth knowing that it also looks identical to
    a collector that never ran, so a silent burndown of zero in February is a
    signal to go and check the collector, not to trust the number.
    """
    path = os.path.join(out_dir, f"outcomes_{season}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["leaid", "date", "status"])
    return pd.read_csv(path, dtype={"leaid": str})


def spent(outcomes, leaid):
    """Bank days this district has spent so far.

    Counted on distinct dates. The feed is archived several times a morning
    and a district can appear more than once for the same day as a closing is
    restated, so counting rows would burn the bank down too fast.
    """
    if not len(outcomes) or "status" not in outcomes.columns:
        return 0
    mine = outcomes[(outcomes["leaid"] == leaid)
                    & (outcomes["status"].isin(SPENDS_A_DAY))]
    return int(mine["date"].nunique())


def pictogram(built, used):
    """How many of each mark to draw, as (unspent, spent, makeup).

    unspent + spent always equals the bank, so the row keeps its full width
    until the bank is blown; makeup days then extend it. Nothing here can go
    negative, which is the whole point of doing the arithmetic in one place.
    """
    built = max(0, int(built))
    used = max(0, int(used))
    spent_marks = min(used, built)
    return built - spent_marks, spent_marks, max(0, used - built)


def caption(built, used):
    """The line under the row. Deliberately flat.

    Overspending a bank is normal in a hard winter and the district handles it
    by adding days in June. It is not a crisis and should not be dressed as
    one, so this states the position and stops.
    """
    built = max(0, int(built))
    used = max(0, int(used))
    left = built - used
    if left > 0:
        return f"{left} left of {built} built into the calendar"
    if left == 0:
        return f"none left of {built} built into the calendar"
    return (f"{built} built into the calendar, all spent "
            f"&middot; {-left} to make up in June")
