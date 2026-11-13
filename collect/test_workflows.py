"""Tests for the Actions workflows that drive these collectors.

A broken workflow file fails in the worst possible way: GitHub does not run it
and does not tell anybody. There is no exception, no red tick, no email -- the
schedule simply never fires, and the first sign is a gap in the data three
weeks later. The forecast snapshot cannot be backfilled, so that gap is
permanent.

This caught a real one: the closings window had its months in the day-of-week
field, where 10, 11 and 12 are not valid values.
"""

import glob
import os

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.join(os.path.dirname(__file__), "..")
WORKFLOW_DIR = os.path.join(ROOT, ".github", "workflows")

# minute, hour, day-of-month, month, day-of-week
CRON_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


def workflows():
    return sorted(glob.glob(os.path.join(WORKFLOW_DIR, "*.yml")))


def load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def triggers(doc):
    """The `on:` block.

    YAML 1.1 parses a bare `on` key as the boolean True, which is a genuine
    trap rather than a quirk worth working around in the file itself: GitHub
    wants the key spelled `on`, so the reader adapts, not the workflow.
    """
    return doc[True] if True in doc else doc["on"]


def crons(doc):
    return [e["cron"] for e in triggers(doc).get("schedule", [])]


def field_values(field):
    """Every literal number in one cron field, ignoring * and step syntax."""
    out = []
    for part in field.split(","):
        part = part.split("/")[0]
        if part == "*":
            continue
        for piece in part.split("-"):
            if piece.isdigit():
                out.append(int(piece))
    return out


def test_there_are_workflows_at_all():
    """They were documented in the runbook for weeks before they existed."""
    assert workflows(), "no workflow files found"


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_every_workflow_parses(path):
    doc = load(path)
    assert doc.get("jobs"), f"{path} defines no jobs"


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_every_cron_has_five_fields(path):
    for cron in crons(load(path)):
        assert len(cron.split()) == 5, f"{cron!r} is not a five-field cron"


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_every_cron_field_is_in_range(path):
    """THE BUG THIS EXISTS FOR. Months written into the day-of-week field look
    perfectly reasonable and never fire."""
    for cron in crons(load(path)):
        for field, (lo, hi) in zip(cron.split(), CRON_RANGES):
            for value in field_values(field):
                assert lo <= value <= hi, (
                    f"{os.path.basename(path)}: {cron!r} has {value} in a "
                    f"field that only accepts {lo}-{hi}")


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_every_workflow_can_be_run_by_hand(path):
    """workflow_dispatch is how a schedule gets exercised in September rather
    than discovered to be broken in January."""
    assert "workflow_dispatch" in triggers(load(path))


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_every_workflow_may_write(path):
    """Without contents: write the jobs run, produce data, and fail to commit
    it. The runbook lists this as a known symptom, so it is worth a test."""
    assert load(path).get("permissions", {}).get("contents") == "write"


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_data_is_committed_even_when_the_job_failed(path):
    """Every collector writes its output and THEN exits non-zero if the result
    looks wrong. Committing only on success would throw away both the data and
    the evidence needed to work out what happened."""
    doc = load(path)
    steps = [s for job in doc["jobs"].values() for s in job["steps"]]
    commits = [s for s in steps if "commit-data.sh" in str(s.get("run", ""))]
    assert commits, f"{os.path.basename(path)} never commits anything"
    for step in commits:
        assert "cancelled()" in str(step.get("if", "")), (
            "the commit step must run on !cancelled(), not only on success")


@pytest.mark.parametrize("path", workflows(), ids=os.path.basename)
def test_writes_go_to_the_data_branch_not_the_deployed_one(path):
    """The whole point of the branch split. A workflow that quietly checks out
    the default branch would restart the public app on every capture."""
    doc = load(path)
    steps = [s for job in doc["jobs"].values() for s in job["steps"]]
    checkouts = [s for s in steps if "actions/checkout" in str(s.get("uses"))]
    assert any(s.get("with", {}).get("ref") == "data" for s in checkouts), (
        f"{os.path.basename(path)} does not check out the data branch")


def test_the_commit_script_exists_and_is_executable():
    path = os.path.join(ROOT, ".github", "scripts", "commit-data.sh")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    # Three workflows push to one branch on overlapping schedules, so a
    # rejected push is routine rather than exceptional.
    assert "--rebase" in body, "no retry on a rejected push"
