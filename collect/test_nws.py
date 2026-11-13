"""Tests for NWS forecast parsing.

The nightly snapshot cannot be re-run for a missed night, so parsing bugs here
are permanent: they corrupt rows that can never be recollected. Everything is
pinned against synthetic gridpoint payloads, and nothing touches the network.

Storm timing gets the most attention because it is the feature most likely to
separate a delay from a closure, and it is the one that cannot be sanity-checked
against live data until December.
"""

from datetime import date, datetime, timedelta

import pytest

import nws

TARGET = date(2026, 1, 15)


def block(start_local, hours, value):
    """One gridpoint value covering `hours` from `start_local` (local time)."""
    t = datetime(2026, 1, start_local[0], start_local[1], tzinfo=nws.LOCAL_TZ)
    return {"validTime": f"{t.isoformat()}/PT{hours}H", "value": value}


def props(**series):
    return {k: {"values": v} for k, v in series.items()}


@pytest.mark.parametrize("text,hours", [
    ("PT1H", 1), ("PT3H", 3), ("PT6H", 6),
    ("P1D", 24), ("P1DT6H", 30), ("PT30M", 0.5),
])
def test_parse_duration(text, hours):
    assert nws.parse_duration_hours(text) == hours


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        nws.parse_duration_hours("banana")


def test_accumulation_is_divided_across_its_span():
    """A 6-hour block holding 6 mm is six hours of 1 mm. Copying 6 into each
    hour would sextuple the storm, and most blocks beyond ~36 hours out are
    multi-hour."""
    df = nws.expand_series({"values": [block((14, 18), 6, 6.0)]},
                           is_accumulation=True)
    assert len(df) == 6
    assert df["value"].sum() == pytest.approx(6.0)
    assert df["value"].iloc[0] == pytest.approx(1.0)


def test_state_values_are_copied_not_divided():
    """Temperature is a state, not a total. Dividing it would be nonsense."""
    df = nws.expand_series({"values": [block((14, 18), 3, -5.0)]},
                           is_accumulation=False)
    assert len(df) == 3
    assert list(df["value"]) == [-5.0, -5.0, -5.0]


def test_missing_property_is_empty_not_an_error():
    assert nws.expand_series(None).empty
    assert nws.expand_series({}).empty


def test_window_is_evening_before_through_9am():
    start, end = nws.window_bounds(TARGET)
    assert start.hour == 18 and start.date() == date(2026, 1, 14)
    assert end.hour == 9 and end.date() == TARGET
    assert (end - start) == timedelta(hours=15)


def test_storm_ending_before_dawn_scores_lower_than_one_ending_at_rush_hour():
    """The ordering is the whole point of the feature: a storm ending at 4am
    gets ploughed and school opens; the same storm still falling at 8am does
    not. Later must always mean a larger number."""
    early = nws.features(props(snowfallAmount=[
        block((14, 22), 6, 60.0)]), TARGET)            # 10pm-4am
    late = nws.features(props(snowfallAmount=[
        block((15, 2), 6, 60.0)]), TARGET)             # 2am-8am

    assert early["snow_end_hour"] < late["snow_end_hour"]
    assert early["snow_end_hour"] == 27                # 3am, last full hour
    assert late["snow_end_hour"] == 31                 # 7am


def test_evening_hours_sort_before_morning_hours():
    evening = nws.features(props(snowfallAmount=[block((14, 19), 2, 20.0)]),
                           TARGET)
    morning = nws.features(props(snowfallAmount=[block((15, 1), 2, 20.0)]),
                           TARGET)
    assert evening["snow_start_hour"] == 19
    assert morning["snow_start_hour"] == 25
    assert evening["snow_start_hour"] < morning["snow_start_hour"]


def test_snow_still_falling_at_9am_is_flagged():
    ongoing = nws.features(props(snowfallAmount=[block((15, 4), 6, 60.0)]),
                           TARGET)
    assert ongoing["snow_ongoing_at_9am"] is True

    done = nws.features(props(snowfallAmount=[block((14, 20), 4, 40.0)]),
                        TARGET)
    assert done["snow_ongoing_at_9am"] is False


def test_flurries_do_not_count_as_a_storm():
    """Below the threshold nothing needs ploughing, so there is no storm and no
    start or end time to report."""
    f = nws.features(props(snowfallAmount=[block((14, 20), 6, 0.6)]), TARGET)
    assert f["snow_mm"] == pytest.approx(0.6)
    assert f["snow_start_hour"] is None
    assert f["snow_end_hour"] is None


def test_snow_outside_the_window_is_excluded():
    """Snow at noon the day before is not overnight snow."""
    f = nws.features(props(snowfallAmount=[block((14, 10), 4, 80.0)]), TARGET)
    assert f["snow_mm"] == 0.0


def test_overnight_subtotal_starts_at_midnight():
    f = nws.features(props(snowfallAmount=[
        block((14, 20), 2, 20.0),      # 8pm-10pm, evening only
        block((15, 1), 2, 30.0),       # 1am-3am, after midnight
    ]), TARGET)
    assert f["snow_mm"] == pytest.approx(50.0)
    assert f["snow_mm_overnight"] == pytest.approx(30.0)


def snap(date="2026-01-15", slot=nws.EVENING, snow=10.0, leaid="3604870"):
    import pandas as pd
    return pd.DataFrame([{"target_date": date, "leaid": leaid,
                          "district": "Binghamton", "capture_slot": slot,
                          "snow_mm": snow}])


def test_write_snapshot_replaces_a_rerun_of_the_same_slot(tmp_path):
    """Re-running a night must overwrite, not duplicate. The job gets retried
    by hand after a failure, and duplicate rows would silently double-weight
    that night in training."""
    import pandas as pd

    nws.write_snapshot(snap(snow=10.0), forecast_dir=str(tmp_path))
    path = nws.write_snapshot(snap(snow=42.0), forecast_dir=str(tmp_path))

    out = pd.read_csv(path)
    assert len(out) == 1
    assert out["snow_mm"].iloc[0] == 42.0


def test_write_snapshot_keeps_other_nights(tmp_path):
    import pandas as pd

    nws.write_snapshot(snap(date="2026-01-15"), forecast_dir=str(tmp_path))
    path = nws.write_snapshot(snap(date="2026-01-16"),
                              forecast_dir=str(tmp_path))
    assert len(pd.read_csv(path)) == 2


def test_the_two_slots_coexist_rather_than_overwriting(tmp_path):
    """The whole point of the slot. An evening rerun must not wipe the 5am row
    it is meant to be compared against."""
    import pandas as pd

    nws.write_snapshot(snap(slot=nws.EVENING, snow=10.0),
                       forecast_dir=str(tmp_path))
    nws.write_snapshot(snap(slot=nws.MORNING, snow=80.0),
                       forecast_dir=str(tmp_path))
    path = nws.write_snapshot(snap(slot=nws.EVENING, snow=12.0),
                              forecast_dir=str(tmp_path))

    out = pd.read_csv(path)
    assert len(out) == 2
    by_slot = out.set_index("capture_slot")["snow_mm"]
    assert by_slot[nws.EVENING] == 12.0
    assert by_slot[nws.MORNING] == 80.0


def test_slot_is_derived_from_the_clock():
    """Derived rather than passed, so a cron firing late still lands right."""
    from datetime import datetime
    early = datetime(2026, 1, 15, 5, 0, tzinfo=nws.LOCAL_TZ)
    late = datetime(2026, 1, 15, 20, 0, tzinfo=nws.LOCAL_TZ)
    assert nws.current_slot(early) == nws.MORNING
    assert nws.current_slot(late) == nws.EVENING


def test_already_captured_is_slot_aware(tmp_path):
    from datetime import date
    nws.write_snapshot(snap(slot=nws.EVENING), forecast_dir=str(tmp_path))
    d = date(2026, 1, 15)
    assert nws.already_captured(d, nws.EVENING, str(tmp_path)) == {"3604870"}
    assert nws.already_captured(d, nws.MORNING, str(tmp_path)) == set()


def test_a_legacy_file_without_slots_is_read_as_evening(tmp_path):
    """Snapshots captured before slots existed are evening captures."""
    import pandas as pd
    legacy = tmp_path / "forecast_2025.csv"
    pd.DataFrame([{"target_date": "2026-01-15", "leaid": "3604870",
                   "district": "Binghamton", "snow_mm": 10.0}]).to_csv(
        legacy, index=False)
    from datetime import date
    got = nws.already_captured(date(2026, 1, 15), nws.EVENING, str(tmp_path))
    assert got == {"3604870"}


# --------------------------------------------------------------------------
# The hourly series
#
# api.weather.gov serves no history, so an hour not written down tonight is
# gone permanently. That makes shape bugs here the same kind of unrecoverable
# as a missed snapshot, which is why they get pinned rather than eyeballed.
# --------------------------------------------------------------------------

def test_the_window_is_fifteen_hours_starting_at_six_pm():
    rows = nws.hourly(props(), TARGET)
    assert len(rows) == 15
    assert rows[0]["time_local"].startswith("2026-01-14T18:00")
    assert rows[-1]["time_local"].startswith("2026-01-15T08:00")


def test_hour_uses_the_same_scale_as_the_feature_row():
    """18-23 is the evening before, 24 onward is the target morning, so later
    always means larger. features() already documents this; the two have to
    agree or a column joined across them silently inverts."""
    rows = nws.hourly(props(), TARGET)
    assert [r["hour"] for r in rows] == list(range(18, 33))


def test_accumulations_are_per_hour_and_sum_back_to_the_feature_total():
    """The rows have to reconcile with features(), or the archive quietly
    disagrees with the row the model was actually served."""
    payload = props(snowfallAmount=[block((14, 21), 3, 6.0)])
    rows = nws.hourly(payload, TARGET)
    hourly_total = sum(r["snow_mm"] or 0 for r in rows)
    assert hourly_total == pytest.approx(6.0)
    assert hourly_total == pytest.approx(nws.features(payload, TARGET)["snow_mm"])


def test_a_state_series_is_copied_not_divided():
    """Temperature is not an accumulation. Dividing it across a block would
    make a 3-hour block at -6 read as three hours of -2."""
    rows = nws.hourly(props(temperature=[block((14, 20), 3, -6.0)]), TARGET)
    got = [r["temp_c"] for r in rows if r["hour"] in (20, 21, 22)]
    assert got == [-6.0, -6.0, -6.0]


def test_hours_the_forecast_says_nothing_about_are_none_not_zero():
    """Zero snow and no forecast are different claims, and the difference is
    the whole reason this is archived rather than recomputed later."""
    rows = nws.hourly(props(snowfallAmount=[block((14, 22), 1, 2.0)]), TARGET)
    assert rows[0]["snow_mm"] is None
    assert [r["snow_mm"] for r in rows if r["hour"] == 22] == [2.0]


def test_an_empty_payload_still_produces_the_full_window():
    """A district whose gridpoint carries nothing must still leave a record
    that we looked, or a gap becomes indistinguishable from a missed capture."""
    rows = nws.hourly({}, TARGET)
    assert len(rows) == 15
    assert all(r["snow_mm"] is None for r in rows)


def test_every_hourly_column_is_present_on_every_row():
    payload = props(snowfallAmount=[block((14, 20), 6, 12.0)],
                    temperature=[block((14, 18), 15, -4.0)])
    rows = nws.hourly(payload, TARGET)
    expected = set(nws.HOURLY_COLUMNS) - {
        "target_date", "leaid", "capture_slot", "captured_at"}
    for row in rows:
        assert set(row) == expected


def test_rerunning_a_slot_replaces_its_rows_rather_than_doubling(tmp_path):
    import pandas as pd

    def rows(slot, snow):
        return pd.DataFrame([
            {"target_date": "2026-01-15", "leaid": "3604870",
             "capture_slot": slot, "captured_at": "2026-01-14T20:00:00-05:00",
             "time_local": "2026-01-14T18:00", "hour": 18, "snow_mm": snow,
             "ice_mm": 0.0, "qpf_mm": 0.0, "temp_c": -2.0, "wind_kmh": 10.0,
             "gust_kmh": 20.0, "pop": 50.0}])

    d = str(tmp_path)
    nws.write_hourly(rows("evening", 1.0), hourly_dir=d)
    nws.write_hourly(rows("morning", 2.0), hourly_dir=d)
    nws.write_hourly(rows("evening", 9.0), hourly_dir=d)

    out = pd.read_csv(nws.hourly_path("2026-01-15", d), dtype={"leaid": str})
    assert len(out) == 2, "a re-run doubled the rows instead of replacing them"
    by_slot = dict(zip(out["capture_slot"], out["snow_mm"]))
    assert by_slot == {"evening": 9.0, "morning": 2.0}


def test_one_gzipped_file_per_night(tmp_path):
    """Write-once files, so a readable diff buys nothing and the bytes are
    real: 53 MB a season plain against 9 gzipped."""
    assert nws.hourly_path("2026-01-15", str(tmp_path)).endswith(
        "2026-01-15.csv.gz")


def test_the_night_file_round_trips_through_gzip(tmp_path):
    import pandas as pd
    row = pd.DataFrame([{
        "target_date": "2026-01-15", "leaid": "3604870",
        "capture_slot": "evening", "captured_at": "2026-01-14T20:00:00-05:00",
        "time_local": "2026-01-14T18:00-05:00", "hour": 18, "snow_mm": 1.5,
        "ice_mm": 0.0, "qpf_mm": 0.4, "temp_c": -2.0, "wind_kmh": 10.0,
        "gust_kmh": 20.0, "pop": 50.0}])
    path = nws.write_hourly(row, hourly_dir=str(tmp_path))
    back = pd.read_csv(path, dtype={"leaid": str})
    assert list(back.columns) == nws.HOURLY_COLUMNS
    assert back["snow_mm"].iloc[0] == 1.5
    assert back["leaid"].iloc[0] == "3604870"
