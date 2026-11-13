"""Does the page actually render?

Every other test in this repo exercises a function. This one runs the whole
Streamlit script the way Streamlit runs it, because the failure this project
most needs to avoid is not a wrong number, it is a red traceback on a public
page on the one night thousands of people are looking at it.

AppTest is Streamlit's own harness: it executes the script in-process and
collects the elements rather than serving them, so this is the real page, not
a mock of it. It is slower than the rest of the suite -- a few seconds -- and
worth it.

Deliberately NOT asserted here: any particular number, verdict or date. In
September the page is in its no-school state and in January it is not, the
verdict wording is picked at random from verdicts.yaml, and pinning any of that
would produce a test that fails every time somebody improves a sentence. What
is pinned is structural: it runs, it raises nothing, and the map is drawn and
is not enormous.
"""

import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1",
                                 reason="needs streamlit >= 1.28")

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snowday.py")

# The map is the largest single element on the page and it is re-sent on every
# rerun, so its size is a load figure rather than a cosmetic one. Measured at
# about 61 KB simplified, against 139 KB before. The ceiling is generous enough
# not to fail on a boundary redraw and tight enough to catch simplification
# being switched off, which would otherwise be invisible.
MAX_SVG_BYTES = 90_000


@pytest.fixture(scope="module")
def page():
    return st_testing.AppTest.from_file(APP, default_timeout=120).run()


def test_the_page_renders_without_raising(page):
    """The whole point. A Streamlit script that raises puts a traceback where
    the forecast should be, and every loader on this page reads a file that
    something else writes."""
    assert not page.exception, [str(e.value) for e in page.exception]


def test_the_map_is_drawn(page):
    svgs = [str(m.value) for m in page.markdown if "<svg" in str(m.value)]
    assert len(svgs) == 1, "expected exactly one region map"
    assert svgs[0].count("<title>") >= 50, "districts missing from the map"


def test_the_map_stays_within_its_payload_budget(page):
    """Guards the simplification. Turning it off costs every viewer 78 KB on
    every refresh and nothing would otherwise complain."""
    svg = next(str(m.value) for m in page.markdown if "<svg" in str(m.value))
    assert len(svg) < MAX_SVG_BYTES, (
        f"map is {len(svg)} bytes; simplification may have regressed")


def test_the_district_picker_offers_the_whole_roster(page):
    assert page.selectbox, "the district picker did not render"
    assert len(page.selectbox[0].options) >= 50


def test_changing_district_does_not_break_the_page(page):
    """The one interaction the page exists to serve."""
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    other = at.selectbox[0].options[-1]
    at.selectbox[0].set_value(other).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("<svg" in str(m.value) for m in at.markdown)


# --------------------------------------------------------------------------
# What happens when a file is broken
#
# This is the scenario the whole design is arranged around. Every file below is
# written by an unattended job, some of them at three in the morning, and the
# page reads them at six. A malformed CSV must cost one card, never the page.
# --------------------------------------------------------------------------

import re  # noqa: E402
import shutil  # noqa: E402

import streamlit as st  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CORRUPTIBLE = [
    "reference/verdicts.yaml",
    "reference/school_calendar.csv",
    "reference/district_shapes.geojson",
    "reference/district_bank.csv",
    "reference/district_coverage.csv",
    "data/forecasts/forecast_2026.csv",
]

GARBAGE = 'leaid,"unterminated\n\x00\x00 not,valid;;;\n'


def render():
    # Cleared between runs, or the first failure's fallback is cached and
    # answers for every later one. That is correct in production -- retrying a
    # broken file on every rerun helps nobody -- and it makes this test lie.
    st.cache_data.clear()
    at = st_testing.AppTest.from_file(APP, default_timeout=180).run()
    body = " ".join(re.sub(r"<[^>]+>", " ", str(m.value)) for m in at.markdown)
    return at, re.sub(r"\s+", " ", body)


@pytest.mark.parametrize("relpath", CORRUPTIBLE)
def test_a_corrupt_file_costs_one_card_not_the_page(relpath, tmp_path):
    path = os.path.join(ROOT, relpath)
    if not os.path.exists(path):
        pytest.skip(f"{relpath} is not present in this checkout")

    backup = str(tmp_path / "backup")
    shutil.copy(path, backup)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(GARBAGE)
        at, body = render()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "Snow Day Predictor" in body
    finally:
        shutil.move(backup, path)
        st.cache_data.clear()


def test_a_missing_roster_degrades_deliberately_rather_than_crashing(tmp_path):
    """The one file the page cannot do without: without it there is no district
    to pick. It gets an explicit card instead of an accidental traceback, and
    the card says nothing about why, because a reader can do nothing with a
    filename."""
    path = os.path.join(ROOT, "reference", "roster.csv")
    backup = str(tmp_path / "roster")
    shutil.copy(path, backup)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(GARBAGE)
        at, body = render()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "Back shortly" in body
        assert ".csv" not in body, "internal detail leaked onto the page"
    finally:
        shutil.move(backup, path)
        st.cache_data.clear()
