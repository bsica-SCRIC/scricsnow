"""Tests for the hand-built scorer.

These pin the SHAPE of the judgement, not the exact numbers. Coefficients are
meant to be tuned; the orderings below are what must survive tuning, because
each encodes something true about ploughing a road.
"""

import json

import pandas as pd
import pytest

import heuristic as H


def row(**kw):
    base = {"snow_total_mm": 0.0, "ice_mm": 0.0, "depth_mm": 0.0,
            "temp_min_c": -3.0, "gust_kmh": 10.0, "elev_m": 300.0,
            "closures_used": 0}
    base.update(kw)
    return pd.DataFrame([base])


def score(**kw):
    return float(H.severity_score(row(**kw)).iloc[0])


def test_more_snow_scores_higher():
    assert score(snow_total_mm=25) < score(snow_total_mm=100) < score(snow_total_mm=250)


def test_ice_outweighs_the_same_depth_of_snow():
    """A quarter inch of glaze shuts a district that would shrug at the same
    amount of snow."""
    assert score(ice_mm=6.0) > score(snow_total_mm=6.0)


def test_extreme_cold_raises_the_score_without_any_snow():
    assert score(temp_min_c=-20.0) > score(temp_min_c=-5.0)


def test_higher_districts_are_more_snow_tolerant_at_equal_snowfall():
    """Backwards from the obvious guess, and the history is clear about it.

    On days with 50 mm or more, districts below 300 m closed 24% of the time
    and those above 450 m closed 15-18%. Hill districts get more snow but cope
    with it better, so once snowfall is held equal they close LESS. The first
    version of this coefficient had the sign the other way round.
    """
    assert score(snow_total_mm=80, elev_m=900) < score(snow_total_mm=80, elev_m=260)


def test_a_spent_bank_makes_a_district_more_reluctant():
    assert score(snow_total_mm=80, closures_used=5) < score(snow_total_mm=80,
                                                            closures_used=0)


def test_wind_only_matters_when_there_is_snow_to_move():
    """Drifting needs loose snow. A gale over bare pavement closes nothing."""
    bare = score(snow_total_mm=0, gust_kmh=60) - score(snow_total_mm=0, gust_kmh=10)
    snowy = score(snow_total_mm=60, gust_kmh=60) - score(snow_total_mm=60, gust_kmh=10)
    assert bare == pytest.approx(0.0)
    assert snowy > 0


def test_a_storm_ending_later_scores_higher():
    """The single most important thing the heuristic knows that the fitted
    model cannot: ploughs need clear hours before the buses roll."""
    early = score(snow_total_mm=100, snow_end_hour=21)   # 9pm, night before
    mid = score(snow_total_mm=100, snow_end_hour=26)     # 2am
    late = score(snow_total_mm=100, snow_end_hour=30)    # 6am
    assert early < mid < late


def test_still_snowing_at_nine_is_worse_than_having_stopped():
    stopped = score(snow_total_mm=100, snow_ongoing_at_9am=False)
    ongoing = score(snow_total_mm=100, snow_ongoing_at_9am=True)
    assert ongoing > stopped


def test_absent_timing_columns_do_not_shift_the_score():
    """History has no hourly detail. A missing column must add nothing rather
    than quietly biasing every historical row."""
    without = H.severity_score(row(snow_total_mm=100))
    with_null = H.severity_score(row(snow_total_mm=100).assign(
        snow_end_hour=None, snow_ongoing_at_9am=False))
    assert float(without.iloc[0]) == pytest.approx(float(with_null.iloc[0]))


def test_a_closure_is_never_likelier_than_a_disruption():
    """Ordinal coherence. A full closure is contained in delay-or-worse, so it
    cannot be the more probable of the two."""
    df = pd.concat([row(snow_total_mm=s) for s in
                    (0, 10, 25, 50, 100, 200, 400)], ignore_index=True)
    p = H.predict(df)
    assert (p["p_disrupted"] >= p["p_closed"]).all()


def test_probabilities_stay_in_range_on_absurd_input():
    p = H.predict(row(snow_total_mm=5000, ice_mm=200, elev_m=3000))
    assert 0.0 <= float(p["p_closed"].iloc[0]) <= 1.0


def test_a_calm_day_is_close_to_zero_not_merely_low():
    """Base rate is under 3%. A clear day must read as clear, or the public
    number is noise."""
    p = H.predict(row())
    assert float(p["p_closed"].iloc[0]) < 0.05


def test_explain_names_the_actual_drivers():
    reasons = " ".join(H.explain({"snow_total_mm": 150.0, "ice_mm": 0.0,
                                  "temp_min_c": -4.0, "depth_mm": 0.0,
                                  "gust_kmh": 10.0, "snow_end_hour": 30}))
    assert "in of snow" in reasons
    assert "morning" in reasons


def test_calibration_comes_from_the_artifact_when_present(tmp_path):
    """The stale-calibration guard.

    heuristic.py's score-to-probability constants used to be literals a human
    had to remember to refit. Rebuilding the training set silently left them
    describing a dataset that no longer existed, and the symptom was not an
    error, just probabilities that drifted away from real frequencies. The
    training run now writes them and this is what proves they get read.
    """
    path = tmp_path / "heuristic.json"
    path.write_text(json.dumps({
        "closed": {"slope": 2.0, "intercept": 0.0},
        "disrupted": {"slope": 2.0, "intercept": 0.0},
    }))
    assert H.load_calibration(str(path)) == {"closed": (2.0, 0.0),
                                             "disrupted": (2.0, 0.0)}


def test_a_missing_or_broken_artifact_falls_back_instead_of_raising(tmp_path):
    """A truncated calibration file should cost a little accuracy on a snowy
    morning, not take the page down."""
    absent = tmp_path / "nope.json"
    assert H.load_calibration(str(absent))["closed"] == (H.CLOSED_SLOPE,
                                                         H.CLOSED_INTERCEPT)

    truncated = tmp_path / "broken.json"
    truncated.write_text('{"closed": {"slope": 0.4')
    assert H.load_calibration(str(truncated))["closed"] == (H.CLOSED_SLOPE,
                                                            H.CLOSED_INTERCEPT)

    incomplete = tmp_path / "partial.json"
    incomplete.write_text('{"closed": {"slope": 1.0, "intercept": 0.0}}')
    assert H.load_calibration(str(incomplete))["closed"] == (H.CLOSED_SLOPE,
                                                             H.CLOSED_INTERCEPT)
