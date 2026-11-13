"""Tests for the closings feed parser.

We have never seen a non-empty payload: in August `locations` is `[]`, and the
first real closing is months away. So these tests do not assert the true shape.
They assert that the parser survives SEVERAL plausible shapes, and that when it
cannot understand something it says so instead of dropping it.

The design contract being tested: a wrong guess about field names must cost
nothing that cannot be recovered from the archived raw JSON.
"""

import collections
import os
import re
import json

import pandas as pd
import pytest

import news as N

ROSTER = pd.DataFrame({
    "leaid": ["3604870", "3602400", "3615900"],
    "district": ["Binghamton City School District",
                 "Afton Central School District",
                 "Johnson City Central School District"],
})


def payload(locations):
    return [{"locations": locations, "num_closings": len(locations),
             "export_type": "L1", "source": "GSync", "run_date": ""}]


# --- shape tolerance --------------------------------------------------------

@pytest.mark.parametrize("record,expect_name", [
    ({"name": "Afton Central School District", "status": "Closed"}, True),
    ({"Name": "Afton Central School District", "Status": "Closed"}, True),
    ({"location": "Afton Central School District", "message": "Closed"}, True),
    ({"title": "Afton Central School District", "closing_type": "Closed"}, True),
    ({"organization": "Afton Central School District", "type": "Closed"}, True),
])
def test_common_field_namings_are_all_understood(record, expect_name):
    """Which of these the vendor actually uses is unknown, so accept them all."""
    df = N.parse(payload([record]), ROSTER)
    assert (df["name"].iloc[0] is not None) == expect_name
    assert df["status"].iloc[0] == "closed"


def test_an_unrecognized_record_is_kept_not_dropped():
    """The failure that must never happen quietly: a shape we did not predict
    still produces a row, with the original preserved for later re-parsing."""
    df = N.parse(payload([{"weird_field": "Afton CSD", "other": "Closed"}]),
                 ROSTER)
    assert len(df) == 1
    assert df["name"].iloc[0] is None      # flagged as unparsed
    assert "weird_field" in df["raw"].iloc[0]


def test_raw_is_preserved_on_every_row():
    rec = {"name": "Afton Central School District", "status": "Closed",
           "extra": {"nested": 1}}
    df = N.parse(payload([rec]), ROSTER)
    assert json.loads(df["raw"].iloc[0])["extra"] == {"nested": 1}


def test_empty_and_malformed_payloads_do_not_raise():
    assert len(N.parse(payload([]), ROSTER)) == 0
    assert len(N.parse(None, ROSTER)) == 0
    assert len(N.parse([{}], ROSTER)) == 0
    assert len(N.parse({"locations": []}, ROSTER)) == 0


def test_a_non_dict_record_still_produces_a_row():
    df = N.parse(payload(["Afton CSD - Closed"]), ROSTER)
    assert len(df) == 1
    assert "Afton" in df["raw"].iloc[0]


# --- status normalization ---------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Closed", "closed"),
    ("CLOSED TODAY", "closed"),
    ("Cancelled", "closed"),
    ("2 Hour Delay", "delay"),
    ("Two hour delay", "delay"),
    ("Delayed opening", "delay"),
    ("Late Start", "delay"),
    ("Early Dismissal", "early"),
    ("Remote Learning Day", "remote"),
    ("Open as scheduled", "open"),
])
def test_status_text_is_normalized(text, expected):
    assert N.normalize_status(text) == expected


def test_closed_wins_over_delay_when_both_words_appear():
    """Order matters: 'closed, no delay' is a closure, not a delay."""
    assert N.normalize_status("Closed - no delay") == "closed"


def test_unknown_status_text_returns_none_rather_than_guessing():
    assert N.normalize_status("Buses on snow routes") is None
    assert N.normalize_status("") is None
    assert N.normalize_status(None) is None


# --- roster matching --------------------------------------------------------

@pytest.mark.parametrize("feed_name", [
    "Binghamton City School District",
    "Binghamton City SD",
    "BINGHAMTON CITY SCHOOLS",
    "Binghamton City School Dist.",
])
def test_district_names_match_across_common_phrasings(feed_name):
    got = N.match_roster([feed_name], ROSTER)
    assert got.get(feed_name) == "3604870"


def test_unrelated_organizations_are_not_matched():
    """A closings feed carries every church, daycare and business in the
    viewing area. Loose matching would file a daycare closure against a
    district, so anything uncertain is left unmatched for review."""
    names = ["Binghamton Day Care Center", "First Baptist Church of Afton",
             "Broome County Offices", "Happy Kids Preschool"]
    got = N.match_roster(names, ROSTER)
    assert got == {}


def test_matching_is_reported_per_row():
    df = N.parse(payload([
        {"name": "Afton Central School District", "status": "Closed"},
        {"name": "Some Random Daycare", "status": "Closed"},
    ]), ROSTER)
    assert df["leaid"].iloc[0] == "3602400"
    assert pd.isna(df["leaid"].iloc[1])


def test_two_districts_reducing_to_the_same_key_match_neither():
    """Silently picking one would file closures against the wrong district."""
    ambiguous = pd.DataFrame({
        "leaid": ["1", "2"],
        "district": ["Springfield Central School District",
                     "Springfield Union Free School District"],
    })
    assert N.match_roster(["Springfield Schools"], ambiguous) == {}


def test_an_alias_file_overrides_token_matching(tmp_path, monkeypatch):
    """Escape hatch for names token matching cannot reach, to be filled in
    once the first real payload shows what the feed actually calls things."""
    path = tmp_path / "aliases.csv"
    path.write_text("feed_name,leaid\nDelaware Academy,3608910\n")
    original = N.load_aliases          # capture before patching, or it recurses
    monkeypatch.setattr(N, "load_aliases", lambda *a, **k: original(path))
    got = N.match_roster(["Delaware Academy"], ROSTER)
    assert got["Delaware Academy"] == "3608910"


def test_locations_of_tolerates_a_bare_dict():
    assert len(N.locations_of({"locations": [{"name": "x"}]})) == 1


def test_no_two_roster_districts_collide_on_the_match_key():
    """A pre-season check that costs nothing and cannot be run in January.

    match_roster drops any normalized key that two districts share, because
    guessing between them would file a closure against the wrong district. The
    consequence is that both become permanently unmatchable, and the symptom on
    a snowy morning is indistinguishable from "those districts stayed open".
    Catch it here instead.
    """
    roster = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "reference", "roster.csv"),
        dtype={"leaid": str})
    keys = collections.defaultdict(list)
    for name in roster["district"]:
        keys[N._norm_name(name)].append(name)

    collisions = {k: v for k, v in keys.items() if len(v) > 1}
    assert not collisions, f"districts share a match key: {collisions}"
    blank = [v for k, v in keys.items() if not k.strip()]
    assert not blank, f"districts reduce to an empty match key: {blank}"


def test_roster_names_survive_the_spellings_a_feed_actually_uses():
    """The WBNG feed will not print official NCES names.

    An unmatched district writes no outcome row, and absence is read as "did
    not close" by both the scorecard and the closed_prev feature. So a matching
    failure is silent in two places at once, and it is worth pinning the
    tolerance now rather than discovering it during the first storm.
    """
    roster = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "reference", "roster.csv"),
        dtype={"leaid": str})

    def variants(name):
        base = re.sub(r"\s+(Central School District|City School District|"
                      r"Central Schools?|School District)$", "", name)
        return [name, base, base + " CSD", base + " Central School District",
                (base + " schools").upper(), base.replace(" ", "-")]

    every = [v for d in roster["district"] for v in variants(d)]
    matched = N.match_roster(every, roster)
    for leaid, name in zip(roster["leaid"], roster["district"]):
        for v in variants(name):
            assert matched.get(v) == leaid, f"{name!r} missed as {v!r}"


# --- the alarm: news.py exits non-zero on a feed it cannot understand --------
#
# These matter in both directions. A missed alarm means a silent season; a false
# alarm in January means the real one gets ignored.

def _parsed(rows):
    """A parsed frame in the shape collect() returns."""
    return pd.DataFrame(rows, columns=["name", "status_text", "status", "raw",
                                       "leaid"])


def _rec(name="Chenango Forks", status="closed", leaid="3604870"):
    return {"name": name, "status_text": status, "status": status,
            "raw": "{}", "leaid": leaid}


def test_no_alarm_when_the_feed_is_empty():
    """The normal state for most of the year."""
    assert N.health(_parsed([]), records=0, matched_before=False) == []


def test_no_alarm_when_everything_parses_and_matches():
    df = _parsed([_rec(), _rec(name="Windsor", leaid="3635130")])
    assert N.health(df, records=2, matched_before=False) == []


def test_alarm_when_no_record_yields_a_name():
    """NAME_KEYS guessed wrong: every record fails, not just an odd one."""
    df = _parsed([_rec(name=None, leaid=None) for _ in range(10)])
    problems = N.health(df, records=10, matched_before=True)
    assert any("NAME_KEYS" in p for p in problems)


def test_alarm_when_no_record_yields_a_status():
    df = _parsed([_rec(status=None) for _ in range(10)])
    problems = N.health(df, records=10, matched_before=True)
    assert any("STATUS_KEYS" in p for p in problems)


def test_alarm_when_the_parser_produces_nothing_at_all():
    """locations_of() failed to find the list inside the payload."""
    problems = N.health(_parsed([]), records=40, matched_before=True)
    assert len(problems) == 1
    assert "no rows at all" in problems[0]


def test_one_odd_entry_is_not_an_alarm():
    """The false-alarm guard. A feed carrying a single malformed record is not
    a parser failure, and failing the job over it would train everyone to
    ignore the red X by February."""
    df = _parsed([_rec() for _ in range(9)] + [_rec(name=None, status=None,
                                                    leaid=None)])
    assert N.health(df, records=10, matched_before=True) == []


def test_a_morning_when_none_of_ours_closed_is_not_an_alarm():
    """The important false-alarm case.

    The WBNG viewing area is far larger than these 50 districts. Churches and
    daycares closing while no tracked district does is an ordinary morning, and
    once roster matching has been proven once it is never suspicious again.
    """
    df = _parsed([_rec(name="St Mary's Preschool", leaid=None)
                  for _ in range(30)])
    assert N.health(df, records=30, matched_before=True) == []


def test_but_the_very_first_unmatched_payload_does_raise_the_alarm():
    """Fires at most once in the project's life, on the first real payload.

    Until one district has matched, we have no evidence the roster matching
    works at all, and that is worth a human glance even though the innocent
    explanation is available.
    """
    df = _parsed([_rec(name="St Mary's Preschool", leaid=None)
                  for _ in range(30)])
    problems = N.health(df, records=30, matched_before=False)
    assert len(problems) == 1
    assert "district_aliases.csv" in problems[0]


def test_a_field_failure_outranks_the_first_payload_warning():
    """Do not tell someone to check the roster when the real problem is that
    no name was parsed. One clear cause beats two vague ones."""
    df = _parsed([_rec(name=None, leaid=None) for _ in range(10)])
    problems = N.health(df, records=10, matched_before=False)
    assert not any("district_aliases" in p for p in problems)


def test_ever_matched_reads_the_outcome_files(tmp_path):
    assert N.ever_matched(str(tmp_path)) is False
    (tmp_path / "outcomes_2026.csv").write_text("date,leaid,status\n")
    assert N.ever_matched(str(tmp_path)) is False
    (tmp_path / "outcomes_2026.csv").write_text(
        "date,leaid,status\n2026-01-14,3604870,closed\n")
    assert N.ever_matched(str(tmp_path)) is True


def test_ever_matched_survives_a_junk_outcome_file(tmp_path):
    (tmp_path / "outcomes_bad.csv").write_text("nonsense\n")
    assert N.ever_matched(str(tmp_path)) is False
    assert N.ever_matched(str(tmp_path / "missing")) is False


# --- the alarm, end to end through main() -----------------------------------

def _feed(records):
    """A payload in the shape locations_of() expects."""
    return {"locations": list(records)}


def _run(monkeypatch, tmp_path, payload, roster=None):
    """Run one full cycle against a temp directory. Returns (code, tmp_path)."""
    monkeypatch.setattr(N, "fetch", lambda *a, **k: (
        payload, {"ok": True, "fetched_at": "2026-01-15T09:00:00-05:00",
                  "last_modified": None, "error": None}))
    if roster is None:
        roster = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "reference",
                         "roster.csv"), dtype={"leaid": str})
    code = N.main(argv=["2026-01-15"], out_dir=str(tmp_path), roster=roster)
    return code


def test_main_returns_zero_on_an_empty_feed(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _feed([])) == 0


def test_main_returns_zero_when_a_real_district_closes(monkeypatch, tmp_path):
    payload = _feed([{"name": "Chenango Forks Central School District",
                      "status": "Closed"}])
    assert _run(monkeypatch, tmp_path, payload) == 0
    written = pd.read_csv(tmp_path / "outcomes_2025.csv", dtype={"leaid": str})
    assert written["status"].tolist() == ["closed"]


def test_main_fails_loudly_when_the_field_names_are_wrong(monkeypatch, tmp_path):
    """The failure this whole mechanism exists for.

    NAME_KEYS and STATUS_KEYS are guesses until the first real payload. If they
    are wrong the job used to archive the JSON, match nothing, print a warning
    into a log nobody reads, and exit zero, leaving a blank scorecard and a
    closed_prev of zero for every district on the second morning of the storm.
    """
    payload = _feed([{"orgName": f"Somewhere {i}", "orgStatus": "Closed"}
                     for i in range(12)])
    assert _run(monkeypatch, tmp_path, payload) == 1


def test_the_evidence_survives_the_failure(monkeypatch, tmp_path):
    """The workflow commits on !cancelled() precisely so this is kept.

    Failing before archiving would destroy the one artifact needed to fix the
    parser, which would make the alarm worse than useless.
    """
    payload = _feed([{"orgName": f"Somewhere {i}", "orgStatus": "Closed"}
                     for i in range(12)])
    assert _run(monkeypatch, tmp_path, payload) == 1
    assert list((tmp_path / "raw").glob("*.json")), "raw payload was not archived"
    assert (tmp_path / "checks.csv").exists(), "checks row was not written"


def test_main_does_not_cry_wolf_on_a_morning_that_is_not_ours(monkeypatch,
                                                              tmp_path):
    """Once matching has been proven, a feed full of churches and daycares with
    no tracked district in it is an ordinary morning."""
    (tmp_path / "outcomes_2025.csv").write_text(
        "date,leaid,status\n2026-01-02,3604870,closed\n")
    payload = _feed([{"name": f"St Mary's Preschool {i}", "status": "Closed"}
                     for i in range(20)])
    assert _run(monkeypatch, tmp_path, payload) == 0


def test_main_flags_the_very_first_payload_that_matches_nobody(monkeypatch,
                                                               tmp_path):
    payload = _feed([{"name": f"St Mary's Preschool {i}", "status": "Closed"}
                     for i in range(20)])
    assert _run(monkeypatch, tmp_path, payload) == 1


def test_main_writes_nothing_outside_the_directory_it_is_given(monkeypatch,
                                                               tmp_path):
    """Paths are injected rather than bound as defaults so a test run cannot
    reach the real data directory. This pins that."""
    payload = _feed([{"name": "Chenango Forks Central School District",
                      "status": "Closed"}])
    _run(monkeypatch, tmp_path, payload)
    real = os.path.join(os.path.dirname(__file__), "..", "data", "outcomes")
    assert not os.path.exists(os.path.join(real, "outcomes_2025.csv"))


def test_the_read_only_path_matches_collect_and_writes_nothing(tmp_path):
    """The app must never call collect().

    collect() archives the raw payload whenever it is non-empty. That is right
    for the once-a-morning job and wrong for a page that refreshes every five
    minutes on an ephemeral filesystem. fetch() and parse() are the read-only
    pair, and this pins that they are equivalent so nobody "simplifies" the app
    back onto collect().
    """
    raw = str(tmp_path / "raw")
    payload_json = payload([{"name": "Afton Central School District",
                             "status": "Closed"}])

    read_only = N.parse(payload_json, ROSTER)
    assert len(read_only) == 1
    assert read_only["status"].iloc[0] == "closed"
    assert not os.path.exists(raw)

    # And the archiving path, for contrast: same rows, plus a file on disk.
    archived = N.archive_raw(payload_json, {"ok": True}, raw_dir=raw)
    assert os.path.exists(archived)
