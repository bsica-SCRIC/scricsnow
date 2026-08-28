"""Tests for the serving path.

Serving runs unattended every night from October to April. A mistake here does
not crash, it publishes a wrong number, so the checks are about behaviour under
weather we cannot observe today: in August every real forecast is calm, and the
first genuine blizzard arrives months after this code has to be right.
"""

import json

import pandas as pd
import pytest

import features as F
import predict as P

ROSTER = pd.DataFrame({
    "leaid": ["3604870", "3615060"],
    "district": ["Binghamton City School District",
                 "Hunter-Tannersville Central School District"],
    "elev_m": [261.0, 903.0],
})


def forecast(**kw):
    base = {"target_date": "2026-01-16", "leaid": "3604870",
            "snow_mm": 0.0, "snow_mm_overnight": 0.0, "ice_mm": 0.0,
            "temp_min_c": -3.0, "gust_max_kmh": 10.0,
            "snow_end_hour": None, "snow_ongoing_at_9am": False}
    base.update(kw)
    return pd.DataFrame([base])


def test_artifact_matches_the_current_feature_definition():
    """The coefficient list is positional. If features.py drifts from the
    artifact, every coefficient lands on the wrong column, silently."""
    model = P.load()
    assert model["features"] == F.FEATURES


def test_load_refuses_a_mismatched_artifact(tmp_path):
    model = P.load()
    broken = dict(model)
    broken["features"] = ["snow_total_mm"]          # deliberately wrong
    path = tmp_path / "model.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="different feature set"):
        P.load(path)


def test_a_blizzard_scores_far_above_a_clear_night():
    """The case that matters and cannot be observed in August."""
    calm = P.predict(forecast(), ROSTER)
    storm = P.predict(forecast(snow_mm=250.0, snow_mm_overnight=200.0,
                               temp_min_c=-8.0, snow_end_hour=30,
                               snow_ongoing_at_9am=True), ROSTER)
    assert float(storm["p_closed"].iloc[0]) > float(calm["p_closed"].iloc[0]) * 10
    assert float(storm["p_closed"].iloc[0]) > 0.25


def test_more_snow_never_lowers_the_probability():
    probs = [float(P.predict(forecast(snow_mm=mm, snow_mm_overnight=mm * 0.7),
                             ROSTER)["p_closed"].iloc[0])
             for mm in (0, 25, 75, 150, 300)]
    assert probs == sorted(probs)


def test_a_closure_is_never_likelier_than_a_disruption():
    out = P.predict(forecast(snow_mm=120.0, snow_mm_overnight=90.0), ROSTER)
    assert (out["p_disrupted"] >= out["p_closed"]).all()


def test_probabilities_stay_in_range_on_absurd_input():
    out = P.predict(forecast(snow_mm=9999.0, snow_mm_overnight=9999.0,
                             ice_mm=500.0, temp_min_c=-60.0), ROSTER)
    p = float(out["p_closed"].iloc[0])
    assert 0.0 <= p <= 1.0


def test_standing_snow_depth_reaches_the_heuristic():
    """There is no forecast of snow depth, so callers pass what was observed.

    It moves the HEURISTIC number, not the model one. depth_mm is no longer a
    fitted feature: its coefficient swung from 0.02 to 0.31 across validation
    folds because it largely restated midwinter and cold, and dropping it
    improved both PR-AUC and Brier. The mechanism is kept in heuristic.py where
    the term is hand-set and cannot swing. See features.py for the argument.
    """
    without = P.predict(forecast(snow_mm=60.0), ROSTER)
    with_depth = P.predict(forecast(snow_mm=60.0), ROSTER,
                           depth_mm={"3604870": 200.0})
    assert (float(with_depth["p_closed_heuristic"].iloc[0])
            > float(without["p_closed_heuristic"].iloc[0]))
    # And it must NOT move the fitted model, which no longer sees the column.
    assert (float(with_depth["p_closed"].iloc[0])
            == pytest.approx(float(without["p_closed"].iloc[0])))


def test_an_unknown_depth_is_missing_rather_than_zero():
    """Absent depth must reach matrix() as NaN so the stored imputation value
    decides what it is worth, instead of this code asserting there is none."""
    feat = F.from_forecast(forecast(snow_mm=60.0), ROSTER)
    assert feat["depth_mm"].isna().all()


def test_a_january_storm_beats_the_same_storm_in_april():
    """The season term is folded around midwinter, not linear.

    The old linear days_into_season sloped upward through the spring, so it
    believed six inches on 25 April was a likelier closure than six inches on
    15 January. Closure rates by month say otherwise: 4.8% in January and 5.7%
    in February against 2.7% in April.
    """
    jan = P.predict(forecast(target_date="2026-01-15", snow_mm=150.0), ROSTER)
    apr = P.predict(forecast(target_date="2026-04-25", snow_mm=150.0), ROSTER)
    assert float(jan["p_closed"].iloc[0]) > float(apr["p_closed"].iloc[0])


def test_an_october_storm_also_scores_below_midwinter():
    """The fold has to cut both ways, or it is just a differently-signed line."""
    oct_ = P.predict(forecast(target_date="2025-10-30", snow_mm=150.0), ROSTER)
    jan = P.predict(forecast(target_date="2026-01-15", snow_mm=150.0), ROSTER)
    assert float(jan["p_closed"].iloc[0]) > float(oct_["p_closed"].iloc[0])


def test_output_carries_district_names_and_a_reason():
    out = P.predict(forecast(snow_mm=150.0, snow_mm_overnight=120.0), ROSTER)
    assert out["district"].iloc[0].startswith("Binghamton")
    assert "in of snow" in out["why"].iloc[0]


def test_all_districts_in_the_snapshot_come_back():
    fc = pd.concat([forecast(leaid="3604870", snow_mm=50.0),
                    forecast(leaid="3615060", snow_mm=50.0)],
                   ignore_index=True)
    out = P.predict(fc, ROSTER)
    assert len(out) == 2
    assert set(out["leaid"]) == {"3604870", "3615060"}


def test_heuristic_second_opinion_is_reported_alongside():
    """They see different things: the heuristic knows storm timing, the model
    does not. Both numbers travel so a disagreement is visible."""
    out = P.predict(forecast(snow_mm=100.0, snow_end_hour=31), ROSTER)
    assert "p_closed_heuristic" in out.columns
    assert 0.0 <= float(out["p_closed_heuristic"].iloc[0]) <= 1.0


def test_snow_enters_with_diminishing_returns():
    """The first inches carry the decision, the twelfth adds almost nothing.

    A linear term had to price both the same and split the difference, which
    read too high on dry midwinter mornings and too low in the middle of the
    range where the call is actually in doubt.
    """
    p = [float(P.predict(forecast(target_date="2026-01-15", snow_mm=mm),
                         ROSTER)["p_closed"].iloc[0])
         for mm in (0.0, 25.4, 76.2, 152.4, 304.8)]
    assert p == sorted(p)
    # A dry midwinter morning is close to nothing happening.
    assert p[0] < 0.02
    # The step from one inch to three has to be bigger than six inches to twelve.
    assert (p[2] - p[1]) > (p[4] - p[3])


def test_day_of_week_is_a_truthful_diagnostic_and_not_a_feature():
    """dow used to be a linear feature fitted only on Monday to Friday, so a
    Saturday extrapolated off the end of the trained range and scored like a
    super-Friday. It was clipped to hide that, which made the column lie in
    exports.

    It earned nothing (removing it moved out-of-fold AUC by 0.0000) and it is
    constant across districts on any given night, so it could never affect the
    ranking the page actually shows. Dropping it removed the hazard outright,
    and the column can go back to telling the truth.
    """
    assert "dow" not in F.FEATURES
    sat = F.add_calendar_context(
        F.from_forecast(forecast(target_date="2026-01-17"), ROSTER))
    assert int(sat["dow"].iloc[0]) == 5


def test_ice_is_scored_by_the_heuristic_and_not_by_the_model():
    """ice_mm means different things on the two sides: derived daily liquid
    precipitation in history, forecast glaze thickness in service. It is not a
    fitted feature for that reason, but freezing rain still has to move the
    published pair of numbers."""
    assert "ice_mm" not in F.FEATURES
    clear = P.predict(forecast(snow_mm=0.0), ROSTER)
    icy = P.predict(forecast(snow_mm=0.0, ice_mm=8.0), ROSTER)
    assert (float(icy["p_closed_heuristic"].iloc[0])
            > float(clear["p_closed_heuristic"].iloc[0]))
    assert (float(icy["p_closed"].iloc[0])
            == pytest.approx(float(clear["p_closed"].iloc[0])))


def test_the_snow_transform_is_shared_by_both_sources():
    """Training and serving must derive snow_sqrt from the same function, or
    the coefficient means one thing when fitted and another when applied."""
    obs = pd.DataFrame({"leaid": ["3604870"], "date": ["2026-01-16"],
                        "snow_mm": [100.0], "snow_prior_mm": [0.0],
                        "depth_mm": [0.0], "prcp_mm": [0.0], "tmin_c": [-5.0],
                        "tmax_c": [-1.0], "gust_ms": [3.0], "elev_m": [261.0]})
    from_obs = F.from_observations(obs)["snow_sqrt"].iloc[0]
    from_fc = F.from_forecast(forecast(snow_mm=100.0), ROSTER)["snow_sqrt"].iloc[0]
    assert from_obs == pytest.approx(from_fc)
    assert from_obs == pytest.approx(10.0)


def events(*pairs):
    """Closure history in the shape add_calendar_context expects."""
    return pd.DataFrame(
        [{"leaid": l, "date": pd.Timestamp(d), "kind": "closed"} for l, d in pairs],
        columns=["leaid", "date", "kind"])


NO_HISTORY = pd.DataFrame(columns=["leaid", "date", "kind"])


def test_a_recent_closure_cuts_the_odds_of_a_second_one_under_heavy_snow():
    """The reluctance effect, which is the whole reason closed_prev exists.

    A district that just spent a snow day does not spend another under the same
    storm unless it is genuinely worse. In the record, mornings with six inches
    or more closed 86.5% of the time when the district had not closed recently
    and 32.3% of the time when it had.
    """
    fresh = P.predict(forecast(target_date="2026-01-15", snow_mm=152.4),
                      ROSTER, events=NO_HISTORY)
    spent = P.predict(forecast(target_date="2026-01-15", snow_mm=152.4),
                      ROSTER, events=events(("3604870", "2026-01-14")))
    assert (float(spent["p_closed"].iloc[0])
            < float(fresh["p_closed"].iloc[0]) * 0.75)


def test_but_a_recent_closure_RAISES_the_odds_on_a_quiet_morning():
    """The effect reverses, and the reversal is the point.

    With little falling snow, having closed recently is not reluctance, it is
    evidence of a bad stretch: snow on the ground and roads not recovered. A
    single non-interacted term would have to average +4 and -54 into one
    coefficient, and measurably does worse than having no term at all.
    """
    fresh = P.predict(forecast(target_date="2026-01-15", snow_mm=0.0),
                      ROSTER, events=NO_HISTORY)
    spent = P.predict(forecast(target_date="2026-01-15", snow_mm=0.0),
                      ROSTER, events=events(("3604870", "2026-01-14")))
    assert (float(spent["p_closed"].iloc[0])
            > float(fresh["p_closed"].iloc[0]))


def test_the_lookback_reaches_back_over_a_weekend():
    """A Monday has to see the previous Friday, or every Monday reads as a
    district that has not closed in a week."""
    monday = "2026-01-19"
    friday = ("3604870", "2026-01-16")
    feat = F.add_calendar_context(
        F.from_forecast(forecast(target_date=monday, snow_mm=100.0), ROSTER),
        events=events(friday))
    assert float(feat["closed_prev"].iloc[0]) == 1.0

    # Four days back is outside the window and must not count.
    feat = F.add_calendar_context(
        F.from_forecast(forecast(target_date=monday, snow_mm=100.0), ROSTER),
        events=events(("3604870", "2026-01-15")))
    assert float(feat["closed_prev"].iloc[0]) == 0.0


def test_the_interaction_column_is_the_product_it_claims_to_be():
    feat = F.add_calendar_context(
        F.from_forecast(forecast(target_date="2026-01-15", snow_mm=100.0), ROSTER),
        events=events(("3604870", "2026-01-14")))
    assert (float(feat["closed_prev_snow"].iloc[0])
            == pytest.approx(float(feat["closed_prev"].iloc[0])
                             * float(feat["snow_sqrt"].iloc[0])))


def test_one_district_closing_does_not_affect_another():
    """closed_prev is a district's own history. Nothing in the model is allowed
    to know what the district next door did."""
    ev = events(("3615060", "2026-01-14"))
    feat = F.add_calendar_context(
        F.from_forecast(forecast(leaid="3604870", target_date="2026-01-15",
                                 snow_mm=100.0), ROSTER), events=ev)
    assert float(feat["closed_prev"].iloc[0]) == 0.0


def test_calendar_context_refuses_to_build_the_interaction_without_snow():
    bare = pd.DataFrame({"leaid": ["3604870"], "date": [pd.Timestamp("2026-01-15")]})
    with pytest.raises(ValueError, match="snow_sqrt"):
        F.add_calendar_context(bare, events=events(("3604870", "2026-01-14")))


def test_closure_history_reads_outcomes_and_keeps_only_closures(tmp_path):
    (tmp_path / "outcomes_2026.csv").write_text(
        "date,leaid,status\n"
        "2026-01-14,3604870,closed\n"
        "2026-01-14,3615060,delay\n"
        "2026-01-15,3615060,closed\n")
    hist = P.closure_history(str(tmp_path))
    assert set(hist["kind"]) == {"closed"}
    assert len(hist) == 2
    assert ("3615060", pd.Timestamp("2026-01-14")) not in set(
        zip(hist["leaid"], hist["date"]))


def test_closure_history_survives_a_missing_or_junk_directory(tmp_path):
    """A broken outcome file costs a day of history, never the evening's run."""
    assert P.closure_history(str(tmp_path / "nope")).empty
    (tmp_path / "outcomes_bad.csv").write_text("not,a,valid\noutcome\n")
    assert P.closure_history(str(tmp_path)).empty


def test_omitting_events_loads_history_rather_than_assuming_none(monkeypatch):
    """The trap this replaced.

    predict() used to take an optional events argument that none of its three
    callers passed, so closures_used was zero on every number the site ever
    published. Omitting history has to mean "go and get it", not "assert there
    is none", because the assertion inflates the number in exactly the
    situation it is most likely to be wrong about.
    """
    monkeypatch.setattr(P, "closure_history",
                        lambda *a, **k: events(("3604870", "2026-01-14")))
    loaded = P.predict(forecast(target_date="2026-01-15", snow_mm=152.4), ROSTER)
    asserted_empty = P.predict(forecast(target_date="2026-01-15", snow_mm=152.4),
                               ROSTER, events=NO_HISTORY)
    assert (float(loaded["p_closed"].iloc[0])
            < float(asserted_empty["p_closed"].iloc[0]))
