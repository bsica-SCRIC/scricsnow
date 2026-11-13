"""Tests for the verdict wording.

The file is meant to be hand-edited over time, so the tests are mostly about
catching an edit that would leave some probability with nothing to say. That
failure is invisible until the night it matters.
"""

import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

import verdicts as V


def test_the_shipped_file_is_valid():
    assert V.validate(V.load())


def test_every_probability_gets_a_verdict():
    """The gap that would otherwise surface on a storm night."""
    data = V.load()
    for i in range(0, 101):
        msg, _ = V.for_probability(i / 100, data)
        assert msg


@pytest.mark.parametrize("p", [-5.0, 0.0, 1.0, 2.0, float("inf")])
def test_out_of_range_probabilities_are_clamped_not_crashed(p):
    msg, _ = V.for_probability(p, V.load())
    assert msg


def test_bands_ascend_and_a_calm_night_differs_from_a_blizzard():
    data = V.load()
    rng = random.Random(0)
    calm = {V.for_probability(0.01, data, random.Random(s))[0] for s in range(40)}
    storm = {V.for_probability(0.95, data, random.Random(s))[0] for s in range(40)}
    assert not (calm & storm)
    assert V.for_probability(0.01, data, rng)[0]


def test_more_than_one_line_per_band_so_refreshing_varies():
    data = V.load()
    seen = {V.for_probability(0.95, data, random.Random(s))[0] for s in range(60)}
    assert len(seen) > 1


def test_a_band_that_stops_short_of_one_is_rejected():
    bad = {"bands": [{"name": "only", "max": 0.8, "messages": ["x"]}]}
    with pytest.raises(V.VerdictError, match="must have max: 1.0"):
        V.validate(bad)


def test_descending_bands_are_rejected():
    bad = {"bands": [{"name": "a", "max": 0.5, "messages": ["x"]},
                     {"name": "b", "max": 0.2, "messages": ["y"]},
                     {"name": "c", "max": 1.0, "messages": ["z"]}]}
    with pytest.raises(V.VerdictError, match="ascend"):
        V.validate(bad)


def test_a_band_with_no_messages_is_rejected():
    bad = {"bands": [{"name": "empty", "max": 1.0, "messages": []}]}
    with pytest.raises(V.VerdictError, match="no messages"):
        V.validate(bad)


def test_announcement_wording_covers_every_status_the_feed_can_emit():
    data = V.load()
    for status in ("closed", "delay", "early", "remote", "something_new"):
        msg, _ = V.for_announcement(status, data)
        assert msg


def test_unavailable_has_wording():
    msg, _ = V.unavailable(V.load())
    assert msg


def test_no_verdict_sounds_like_permission_to_skip_school():
    """This page is entertainment. Nothing in it should read as advice."""
    data = yaml.safe_load(open(V.DEFAULT_PATH, encoding="utf-8"))
    banned = ("stay home", "skip school", "don't go", "do not go",
              "sleep in tomorrow", "no school today")
    lines = []
    for band in data["bands"]:
        lines += band["messages"]
    for section in (data.get("announced") or {}).values():
        lines += section.get("messages", [])
    for line in lines:
        low = line.lower()
        assert not any(b in low for b in banned), f"reads as advice: {line}"


def test_no_school_and_too_far_have_wording():
    data = V.load()
    for fn in (V.no_school, V.too_far):
        msg, _ = fn(data)
        assert msg


def test_too_far_wording_is_honest_about_the_limit():
    """The whole credibility of the site rests on the accuracy record, which is
    measured at a one-night lead. These lines must not promise more."""
    import yaml
    data = yaml.safe_load(open(V.DEFAULT_PATH, encoding="utf-8"))
    lines = " ".join(data["too_far"]["messages"]).lower()
    assert "night before" in lines or "evening before" in lines


def test_refreshing_varies_the_no_school_line():
    import random
    data = V.load()
    seen = {V.no_school(data, random.Random(s))[0] for s in range(40)}
    assert len(seen) > 1


# --------------------------------------------------------------------------
# How old the number is
# --------------------------------------------------------------------------

STALE_AFTER = timedelta(hours=2)
LOCAL_TZ = ZoneInfo("America/New_York")


def test_a_fresh_reading_is_stamped_with_its_local_time():
    at = pd.Timestamp("2026-01-15T20:42:00-05:00")
    got = V.reading_age(at, "live", LOCAL_TZ, STALE_AFTER,
                        now=datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc)
                        + timedelta(hours=4))
    assert got == "Updated 8:42 PM"


def test_a_stale_reading_says_so_instead_of_hiding_it():
    """The page used to answer this by fetching NWS per viewer, which turned
    one broken capture job into thousands of blocking requests at the busiest
    moment of the year. Telling the reader the number is old is the cheaper
    and more honest answer."""
    at = pd.Timestamp("2026-01-15T20:42:00-05:00")
    now = at.to_pydatetime() + timedelta(hours=5)
    got = V.reading_age(at, "live", LOCAL_TZ, STALE_AFTER, now=now)
    assert got.startswith("Last updated 8:42 PM")
    assert "Nothing newer" in got


def test_the_frozen_call_carries_no_capture_time_and_is_not_called_stale():
    """The ordinary-night path, and the one that used to trigger a fetch for
    every viewer: on a quiet night no history row is written, so the number
    comes from the call, which has no captured_at."""
    assert V.reading_age(None, "call", LOCAL_TZ, STALE_AFTER) \
        == "This is tonight's 8pm Call."


def test_an_unknown_source_says_nothing_rather_than_guessing():
    assert V.reading_age(None, "none", LOCAL_TZ, STALE_AFTER) == ""
