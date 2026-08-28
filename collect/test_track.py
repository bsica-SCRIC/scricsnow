"""Tests for the call record and the prediction history.

The rule these exist to protect: a published accuracy record must not be
editable after the fact. If retraining the model could rewrite last winter's
calls, the accuracy on the page would improve without a single new observation,
and it would be fiction.

No network. write_call and write_history are pure given a prediction frame.
"""

from datetime import datetime

import pandas as pd
import pytest

import track as T


def pred(target="2026-01-16", rows=(("3604870", "Binghamton", 0.62, 4.1),)):
    return pd.DataFrame([
        {"target_date": target, "leaid": leaid, "district": name,
         "p_closed": p, "p_disrupted": min(1.0, p + 0.1), "snow_in": snow,
         "captured_at": "2026-01-15T20:00:00-05:00",
         "capture_slot": "evening"}
        for leaid, name, p, snow in rows])


MODEL = {"trained_at": "2026-08-27T00:00:00+00:00"}

# Fixed clock inside the call window, so these tests do not depend on when the
# suite happens to run.
EVENING = datetime(2026, 1, 15, 20, 0, tzinfo=T.nws.LOCAL_TZ)


# --- the call: written once, never rewritten --------------------------------

def test_a_call_is_written_the_first_time(tmp_path):
    path, why = T.write_call(pred(), model=MODEL, calls_dir=str(tmp_path), now=EVENING)
    assert why == "written"
    assert len(pd.read_csv(path)) == 1


def test_a_second_call_for_the_same_night_is_refused(tmp_path):
    """The integrity rule. A better forecast arriving at 10pm does not get to
    replace what the page said at 8pm."""
    T.write_call(pred(), model=MODEL, calls_dir=str(tmp_path), now=EVENING)
    better = pred(rows=(("3604870", "Binghamton", 0.95, 12.0),))
    path, why = T.write_call(better, model=MODEL, calls_dir=str(tmp_path), now=EVENING)

    assert path is None
    assert "already made" in why

    stored = pd.read_csv(str(tmp_path) + "/calls_2025.csv")
    assert stored["p_closed"].iloc[0] == pytest.approx(0.62)


def test_force_exists_for_a_genuine_correction(tmp_path):
    T.write_call(pred(), model=MODEL, calls_dir=str(tmp_path), now=EVENING)
    corrected = pred(rows=(("3604870", "Binghamton", 0.95, 12.0),))
    path, why = T.write_call(corrected, model=MODEL, calls_dir=str(tmp_path),
                             force=True)
    assert why == "written"
    assert pd.read_csv(path)["p_closed"].iloc[0] == pytest.approx(0.95)


def test_a_different_night_is_not_blocked_by_an_earlier_call(tmp_path):
    T.write_call(pred(target="2026-01-16"), model=MODEL,
                 calls_dir=str(tmp_path), now=EVENING)
    path, why = T.write_call(pred(target="2026-01-20"), model=MODEL,
                             calls_dir=str(tmp_path), now=EVENING)
    assert why == "written"
    assert len(pd.read_csv(path)) == 2


def test_the_call_records_which_model_made_it(tmp_path):
    """So a later retrain is visible as a different model, not a silent edit."""
    path, _ = T.write_call(pred(), model=MODEL, calls_dir=str(tmp_path), now=EVENING)
    assert pd.read_csv(path)["model_trained_at"].iloc[0] == MODEL["trained_at"]


def test_load_call_returns_only_that_night(tmp_path):
    T.write_call(pred(target="2026-01-16"), model=MODEL,
                 calls_dir=str(tmp_path), now=EVENING)
    T.write_call(pred(target="2026-01-20"), model=MODEL,
                 calls_dir=str(tmp_path), now=EVENING)
    got = T.load_call("2026-01-20", calls_dir=str(tmp_path))
    assert len(got) == 1
    assert got["target_date"].iloc[0] == "2026-01-20"


# --- history: only when something is happening ------------------------------

def test_a_quiet_night_records_nothing(tmp_path):
    """Otherwise the file fills with flat lines at 1% and the chart is noise."""
    quiet = pred(rows=(("3604870", "Binghamton", 0.004, 0.0),))
    assert T.write_history(quiet, history_dir=str(tmp_path)) is None


def test_a_snowy_night_is_recorded_even_at_low_probability():
    """Watching the line move on a marginal night is exactly the interesting
    case, so snowfall alone is enough to start recording."""
    marginal = pred(rows=(("3604870", "Binghamton", 0.01, 2.0),))
    assert T.worth_recording(marginal)


def test_one_district_above_threshold_records_the_whole_region(tmp_path):
    mixed = pred(rows=(("3604870", "Binghamton", 0.004, 0.0),
                       ("3615060", "Hunter-Tannersville", 0.30, 3.0)))
    path = T.write_history(mixed, history_dir=str(tmp_path))
    assert len(pd.read_csv(path)) == 2


def test_history_appends_rather_than_replacing(tmp_path):
    first = pred(rows=(("3604870", "Binghamton", 0.20, 2.0),))
    T.write_history(first, history_dir=str(tmp_path))
    later = first.copy()
    later["captured_at"] = "2026-01-15T21:00:00-05:00"
    later["p_closed"] = 0.42
    path = T.write_history(later, history_dir=str(tmp_path))

    out = pd.read_csv(path)
    assert len(out) == 2
    assert sorted(out["p_closed"]) == pytest.approx([0.20, 0.42])


def test_rerunning_the_same_capture_replaces_itself(tmp_path):
    first = pred(rows=(("3604870", "Binghamton", 0.20, 2.0),))
    T.write_history(first, history_dir=str(tmp_path))
    same = first.copy()
    same["p_closed"] = 0.25
    path = T.write_history(same, history_dir=str(tmp_path))

    out = pd.read_csv(path)
    assert len(out) == 1
    assert out["p_closed"].iloc[0] == pytest.approx(0.25)


def test_history_keeps_only_thin_columns():
    """It is written hourly all winter. The full feature set lives in the
    forecast archive; this file only needs to draw a line."""
    quiet = pred(rows=(("3604870", "Binghamton", 0.30, 2.0),))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = T.write_history(quiet, history_dir=d)
        cols = set(pd.read_csv(path).columns)
    assert cols == {"captured_at", "target_date", "leaid", "p_closed",
                    "p_disrupted", "snow_in"}


def test_season_boundary_is_july():
    assert T.season_of("2026-01-16") == 2025
    assert T.season_of("2026-08-16") == 2026


def test_the_call_can_only_be_made_near_8pm(tmp_path):
    """It is called the 8pm call. An afternoon run must not publish it early;
    the hourly job passes --call every hour and this is what picks the right
    one."""
    afternoon = datetime(2026, 1, 15, 16, 0, tzinfo=T.nws.LOCAL_TZ)
    evening = datetime(2026, 1, 15, 20, 0, tzinfo=T.nws.LOCAL_TZ)

    path, why = T.write_call(pred(), model=MODEL, calls_dir=str(tmp_path),
                             now=afternoon)
    assert path is None and "window" in why

    path, why = T.write_call(pred(), model=MODEL, calls_dir=str(tmp_path),
                             now=evening)
    assert why == "written"


def test_call_window_bounds():
    def at(h):
        return datetime(2026, 1, 15, h, 0, tzinfo=T.nws.LOCAL_TZ)
    assert not T.in_call_window(at(18))
    assert T.in_call_window(at(19))
    assert T.in_call_window(at(20))
    assert T.in_call_window(at(21))
    assert not T.in_call_window(at(22))


def test_no_call_is_made_for_a_day_school_is_out(tmp_path):
    """Publishing a closure probability for Christmas, then scoring it, would
    hand the accuracy record a free correct answer for a morning nobody was
    going to school anyway."""
    christmas = pred(target="2026-12-25")
    path, why = T.write_call(christmas, model=MODEL, calls_dir=str(tmp_path),
                             now=EVENING)
    assert path is None
    assert "no school" in why


def test_no_call_on_a_weekend(tmp_path):
    saturday = pred(target="2026-01-17")
    path, why = T.write_call(saturday, model=MODEL, calls_dir=str(tmp_path),
                             now=EVENING)
    assert path is None and "weekend" in why


def test_force_still_overrides_the_school_day_guard(tmp_path):
    """An escape hatch has to actually escape."""
    path, why = T.write_call(pred(target="2026-12-25"), model=MODEL,
                             calls_dir=str(tmp_path), now=EVENING, force=True)
    assert why == "written"
