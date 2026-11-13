"""Tests for the outcome resolver and, more importantly, the coverage gate.

The interesting failure here is silent and flattering. Reading "no closure on
file" as "the district was open" costs nothing visible: no exception, no empty
page, just an accuracy record that is too good in a direction nobody audits.
So most of what is pinned below is about what does NOT get scored.
"""

import os

import pandas as pd

import outcomes as O


def write(out_dir, name, rows):
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, name), index=False)


def coverage(**kw):
    """Coverage table for two districts. Defaults are the shipped defaults:
    calendar covers Binghamton, news covers nobody until it proves otherwise."""
    rows = [
        {"leaid": "3604870", "district": "Binghamton",
         "calendar": kw.get("bing_calendar", "yes"),
         "news": kw.get("bing_news", "unknown")},
        {"leaid": "3615900", "district": "Johnson City",
         "calendar": kw.get("jc_calendar", "no"),
         "news": kw.get("jc_news", "unknown")},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------

def test_a_human_override_beats_the_calendar_and_the_feed(tmp_path):
    d = str(tmp_path)
    write(d, "outcomes_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "delay"}])
    write(d, "calendar_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "open"}])
    write(d, "overrides_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])

    res = O.resolve(out_dir=d)
    assert len(res) == 1
    assert res["status"].iloc[0] == "closed"
    assert res["source"].iloc[0] == "manual"


def test_the_calendar_beats_the_feed(tmp_path):
    d = str(tmp_path)
    write(d, "outcomes_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "delay"}])
    write(d, "calendar_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])

    res = O.resolve(out_dir=d)
    assert res["status"].iloc[0] == "closed"
    assert res["source"].iloc[0] == "calendar"


def test_the_sources_are_kept_apart_so_they_can_be_compared_later(tmp_path):
    """Merging on disk would answer 'what happened' and destroy 'was the feed
    right', which is the question that decides whether year three needs any of
    the manual apparatus."""
    d = str(tmp_path)
    write(d, "outcomes_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "delay"}])
    write(d, "calendar_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])

    stacked = O.load_all(out_dir=d)
    assert set(stacked["source"]) == {"news", "calendar"}
    assert len(stacked) == 2


def test_the_calendar_file_is_not_swallowed_by_the_news_reader(tmp_path):
    """Both readers scan by prefix. A calendar file named outcomes_calendar_*
    would match `outcomes_` and be counted twice, once under the wrong source
    and at the wrong precedence."""
    d = str(tmp_path)
    write(d, "calendar_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])
    news = O.load_source("news", out_dir=d)
    assert news.empty


def test_a_restated_closing_does_not_become_two_days(tmp_path):
    """The feed is archived several times a morning and repeats itself."""
    d = str(tmp_path)
    write(d, "outcomes_2025.csv", [
        {"date": "2026-01-15", "leaid": "3604870", "status": "closed"},
        {"date": "2026-01-15", "leaid": "3604870", "status": "closed"},
    ])
    assert len(O.resolve(out_dir=d)) == 1


def test_a_truncated_file_costs_its_own_rows_and_nothing_else(tmp_path):
    d = str(tmp_path)
    write(d, "outcomes_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])
    with open(os.path.join(d, "calendar_2025.csv"), "w", encoding="utf-8") as fh:
        fh.write('date,leaid,status\n"unterminated\n')
    res = O.resolve(out_dir=d)
    assert len(res) == 1
    assert res["source"].iloc[0] == "news"


def test_no_outcomes_at_all_is_an_empty_frame_not_a_raise(tmp_path):
    assert O.resolve(out_dir=str(tmp_path)).empty
    assert O.load_all(out_dir=str(tmp_path)).empty
    assert O.closures(out_dir=str(tmp_path)).empty


# --------------------------------------------------------------------------
# Coverage: the flattering bug
# --------------------------------------------------------------------------

def test_a_district_the_feed_does_not_carry_is_never_observed(tmp_path):
    """THE BUG THIS MODULE EXISTS FOR.

    The feed was checked, and it carried Binghamton. Johnson City is not in
    the viewing area, so its silence is not evidence of anything. Before the
    coverage gate it scored as a correct 'open' call, every morning, free.
    """
    d = str(tmp_path)
    write(d, "outcomes_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])
    checks = {"news": pd.DataFrame([{"date": "2026-01-15", "ok": True}])}

    seen = O.observed(coverage=coverage(bing_news="yes"),
                      checks_by_source=checks, out_dir=d)
    assert ("2026-01-15", "3604870") in seen
    assert ("2026-01-15", "3615900") not in seen


def test_unknown_news_coverage_does_not_count_as_covered(tmp_path):
    """We have never seen a payload, so we do not know who is in it. Unknown
    has to mean unknown or the gate is decorative."""
    checks = {"news": pd.DataFrame([{"date": "2026-01-15", "ok": True}])}
    seen = O.observed(coverage=coverage(), checks_by_source=checks,
                      out_dir=str(tmp_path))
    assert seen == set()


def test_a_manual_override_is_its_own_evidence(tmp_path):
    """Johnson City is covered by nothing. Typing the override is the coverage,
    which is the entire mechanism by which the two off-export districts ever
    get scored."""
    d = str(tmp_path)
    write(d, "overrides_2025.csv",
          [{"date": "2026-01-15", "leaid": "3615900", "status": "closed"}])
    seen = O.observed(coverage=coverage(), checks_by_source={}, out_dir=d)
    assert seen == {("2026-01-15", "3615900")}


def test_calendar_coverage_still_needs_the_reconciler_to_have_run(tmp_path):
    """Covering a district says the source CAN speak for it, not that it did."""
    cov = coverage()
    assert O.observed(coverage=cov, checks_by_source={},
                      out_dir=str(tmp_path)) == set()

    checks = {"calendar": pd.DataFrame([{"date": "2026-01-15", "ok": True}])}
    seen = O.observed(coverage=cov, checks_by_source=checks,
                      out_dir=str(tmp_path))
    assert seen == {("2026-01-15", "3604870")}


def test_a_failed_check_does_not_count_as_a_look(tmp_path):
    checks = {"calendar": pd.DataFrame([{"date": "2026-01-15", "ok": False}])}
    assert O.observed(coverage=coverage(), checks_by_source=checks,
                      out_dir=str(tmp_path)) == set()


def test_a_missing_coverage_file_observes_nothing(tmp_path):
    """The safe direction. A missing coverage table produces an empty accuracy
    record, never a flattering one."""
    empty = O.load_coverage(path=str(tmp_path / "nope.csv"))
    checks = {"news": pd.DataFrame([{"date": "2026-01-15", "ok": True}])}
    assert O.observed(coverage=empty, checks_by_source=checks,
                      out_dir=str(tmp_path)) == set()


# --------------------------------------------------------------------------
# Promotion and the shipped table
# --------------------------------------------------------------------------

def test_seen_in_news_reports_who_the_feed_has_actually_carried(tmp_path):
    d = str(tmp_path)
    write(d, "outcomes_2025.csv",
          [{"date": "2026-01-15", "leaid": "3604870", "status": "closed"}])
    write(d, "overrides_2025.csv",
          [{"date": "2026-01-15", "leaid": "3615900", "status": "closed"}])
    # Only the feed promotes feed coverage. An override says nothing about
    # whether WBNG carries that district.
    assert O.seen_in_news(out_dir=d) == {"3604870"}


def test_build_coverage_marks_the_off_export_districts(tmp_path):
    roster = pd.DataFrame({"leaid": ["3604870", "3615900"],
                           "district": ["Binghamton", "Johnson City"]})
    extras = pd.DataFrame({"leaid": ["3615900"]})
    table = O.build_coverage(roster, extras)
    by = table.set_index("leaid")
    assert by.loc["3604870", "calendar"] == "yes"
    assert by.loc["3615900", "calendar"] == "no"
    # Nobody starts news-covered. It is earned by evidence.
    assert set(table["news"]) == {"unknown"}


def test_the_shipped_coverage_table_matches_the_roster():
    """The table is generated, so the only way it drifts is by hand-editing it
    and forgetting a district. A district missing from it is invisible to the
    scorecard forever, which is a silent failure."""
    root = os.path.join(os.path.dirname(__file__), "..")
    roster = pd.read_csv(os.path.join(root, "reference", "roster.csv"),
                         dtype={"leaid": str})
    cov = O.load_coverage()
    assert set(cov["leaid"]) == set(roster["leaid"])
    assert set(cov["calendar"]) <= {"yes", "no", "unknown"}
    assert set(cov["news"]) <= {"yes", "no", "unknown"}


def test_closures_carries_only_full_closures(tmp_path):
    d = str(tmp_path)
    write(d, "outcomes_2025.csv", [
        {"date": "2026-01-15", "leaid": "3604870", "status": "closed"},
        {"date": "2026-01-16", "leaid": "3604870", "status": "delay"},
    ])
    got = O.closures(out_dir=d)
    assert len(got) == 1
    assert got["kind"].iloc[0] == "closed"
