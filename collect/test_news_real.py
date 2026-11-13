"""The closings feed as it really is, from the first live payload (2026-11-13).

test_news.py was written before any payload existed and tests tolerance of
plausible shapes. This file pins the real one: where the records live, which
fields carry the name and status, and which names are ours.
"""

import os

import pandas as pd
import pytest

import news as N

ROSTER = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "reference", "roster.csv"),
    dtype={"leaid": str})


def rec(name, status, updated="2026-11-13", category="Schools",
        state="NEW YORK", comment=None):
    return {"forced_organization_name": name, "forced_status_name": status,
            "forced_category_name": category, "state": state,
            "stateName": state, "comments_line1": comment, "event": None,
            "expiration": f"{updated} 11:00:00", "updated": updated,
            "rec_id": "123456", "zipcode": "13815"}


def feed(records):
    """The real envelope: records under "record", state names under
    "locations"."""
    return [{"record": records, "locations": ["New York"],
             "num_closings": len(records), "export_type": "L1",
             "source": "GSync", "run_date": ""}]


def leaid_of(name):
    return N.match_roster([name], ROSTER).get(name)


def test_records_are_read_from_record_not_locations():
    df = N.parse(feed([rec("Afton School District", "Closed Friday")]), ROSTER)
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["name"], row["status"]) == ("Afton School District", "closed")
    assert row["leaid"] == "3602400"
    assert row["for_date"] == "2026-11-13"


@pytest.mark.parametrize("text,expected", [
    ("Closed Friday", "closed"),
    ("Delayed Friday 2 Hours", "delay"),
    ("Delayed Friday 1 Hour", "delay"),
    ("Delayed Friday", "delay"),
    ("Opening Late Friday 2 Hours", "delay"),
])
def test_real_status_wording(text, expected):
    assert N.normalize_status(text) == expected


# --- names: the ones the feed really used -----------------------------------

@pytest.mark.parametrize("feed_name,leaid", [
    ("Afton School District", "3602400"),
    ("Bainbridge-Guilford Schools", "3603810"),
    ("Delaware Academy CSD at Delhi", "3608910"),
    ("Gilbertsville-Mt. Upton Schools", "3604757"),
    ("Norwich School District", "3621330"),
    ("Oxford School District", "3622260"),
    ("Sherburne Earlville Schools", "3626700"),
    ("Unadilla Valley Schools", "3600019"),
    ("Unatego School District", "3629240"),       # alias: Otego-Unadilla
    ("Walton Central School", "3629820"),
    ("Worcester Central School", "3631740"),
])
def test_feed_names_reach_their_district(feed_name, leaid):
    assert leaid_of(feed_name) == leaid


@pytest.mark.parametrize("feed_name,leaid", [
    ("Delhi Schools", "3608910"),                 # fewer words, one candidate
    ("Cherry Valley Schools", "3600006"),
    ("Walton Public Schools", "3629820"),         # a harmless extra word
    ("Otselic Valley CSD", "3622100"),            # alias
])
def test_matching_is_a_little_greedy(feed_name, leaid):
    assert leaid_of(feed_name) == leaid


@pytest.mark.parametrize("feed_name", [
    "Norwich Head Start", "North Norwich Head Start", "Franklin Head Start",
    "Oxford Head Start", "Sidney Head Start", "Walton Head Start",
    "Greene Head Start", "Holy Family School- Norwich",
    "Otsego Christian Academy", "Valley Heights Christian Academy",
    "DCMO BOCES - Harrold Campus", "Bovina Center Montessori School",
    "Cincinnatus School District", "Marathon School District",
    "Vestal High School",
])
def test_a_shared_town_name_is_not_a_match(feed_name):
    """The reason matching is only a LITTLE greedy. A Head Start closing filed
    as a district snow day is invisible; a missed district is not."""
    assert leaid_of(feed_name) is None


def test_a_word_two_districts_share_matches_neither():
    assert leaid_of("Unadilla Schools") is None      # Unadilla Valley, Unatego
    assert leaid_of("Chenango Schools") is None      # Forks, Valley


def test_a_blank_alias_blocks_a_name(tmp_path, monkeypatch):
    path = tmp_path / "aliases.csv"
    path.write_text("feed_name,leaid\nDelhi Schools,\n")
    original = N.load_aliases
    monkeypatch.setattr(N, "load_aliases", lambda *a, **k: original(path))
    assert leaid_of("Delhi Schools") is None


def test_other_states_and_categories_are_not_matched():
    df = N.parse(feed([
        rec("Afton School District", "Closed Friday", state="PENNSYLVANIA"),
        rec("Afton", "Closed Friday", category="Businesses"),
    ]), ROSTER)
    assert df["leaid"].isna().all()


# --- which day a status is about ---------------------------------------------

def test_the_weekday_in_the_status_sets_the_date():
    assert N.closure_date("Closed Friday", "2026-11-13") == "2026-11-13"
    # Posted Sunday evening about Monday.
    assert N.closure_date("Closed Monday", "2026-11-15") == "2026-11-16"
    assert N.closure_date("Closed", "2026-11-13") is None


def test_a_closing_for_tomorrow_is_not_written_against_today(tmp_path,
                                                             monkeypatch):
    body = feed([rec("Afton School District", "Closed Monday",
                     updated="2026-11-15")])
    monkeypatch.setattr(N, "fetch", lambda *a, **k: (
        body, {"ok": True, "last_modified": None, "fetched_at": "t"}))
    out = tmp_path / "outcomes"
    assert N.main(["2026-11-15"], out_dir=str(out), roster=ROSTER) == 0
    assert not (out / "outcomes_2026.csv").exists()
    assert N.main(["2026-11-16"], out_dir=str(out), roster=ROSTER) == 0
    got = pd.read_csv(out / "outcomes_2026.csv", dtype={"leaid": str})
    assert list(got["date"]) == ["2026-11-16"]


# --- the season file ---------------------------------------------------------

def frame(rows):
    return pd.DataFrame(rows, columns=["leaid", "name", "status",
                                       "status_text"])


def test_a_district_that_dropped_off_the_feed_keeps_its_row(tmp_path):
    """Records expire mid-morning and the job runs after that."""
    N.write_outcomes(frame([("1", "A", "closed", "Closed Friday"),
                            ("2", "B", "delay", "Delayed Friday")]),
                     "2026-11-13", out_dir=str(tmp_path))
    # Later run: A has expired, B upgraded to closed.
    N.write_outcomes(frame([("2", "B", "closed", "Closed Friday")]),
                     "2026-11-13", out_dir=str(tmp_path))
    got = pd.read_csv(tmp_path / "outcomes_2026.csv", dtype={"leaid": str})
    assert dict(zip(got["leaid"], got["status"])) == {"1": "closed",
                                                      "2": "closed"}


def test_two_feed_names_for_one_district_write_one_row(tmp_path):
    N.write_outcomes(frame([("1", "A Schools", "delay", "Delayed Friday"),
                            ("1", "A CSD", "closed", "Closed Friday")]),
                     "2026-11-13", out_dir=str(tmp_path))
    got = pd.read_csv(tmp_path / "outcomes_2026.csv", dtype={"leaid": str})
    assert list(got["status"]) == ["closed"]
