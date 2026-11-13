"""Refuse to commit a file that is wrong. The last gate before git.

WHY THIS EXISTS, IN ONE SENTENCE: the app reads what these jobs write, so a
malformed CSV committed at 3am is a broken public page at 6am, and the job that
wrote it exited zero and told nobody.

The shape is copied from news.health(), which already worked this way and is
the only alerting channel this project has: a GitHub Actions failure emails the
repository owner, so a non-zero exit is the alarm. That makes false alarms
expensive. Cry wolf in January and the real one gets ignored in February. So
every check below is one that cannot fire on an ordinary night.

WHAT IS DELIBERATELY NOT CHECKED:

    "fewer districts than usual"     A partial night is worth far more than no
                                     night, and nws.snapshot is built to write
                                     one. Failing on it would throw away the
                                     rows it just saved.
    "no closures today"              That is most days.
    "the numbers look wrong"         Not knowable here, and a check that needs
                                     judgement is a check that fires on noise.

The order of operations in every caller has to be: write the file, THEN
validate, THEN exit non-zero. Validating first and refusing to write loses the
evidence needed to work out what went wrong.
"""

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
ROSTER_CSV = os.path.join(ROOT, "reference", "roster.csv")


def roster_ids(path=ROSTER_CSV):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, dtype={"leaid": str})["leaid"])


def check(df, name, required=(), leaid_col="leaid", date_cols=("date",),
          known_ids=None, min_rows=1):
    """Problems worth failing a job over. Pure, and returns a list of strings.

    Returns rather than raises so a caller can print every problem at once. A
    job that fails on the first of three faults gets fixed three times.
    """
    problems = []
    label = f"{name}:"

    if df is None:
        return [f"{label} nothing to write at all. The job produced no frame, "
                f"which is different from producing an empty one."]

    if len(df) < min_rows:
        problems.append(f"{label} {len(df)} row(s), expected at least "
                        f"{min_rows}.")
        return problems

    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"{label} missing column(s) {missing}. Got "
                        f"{list(df.columns)}.")
        # Every check below reads columns, so stop here rather than reporting
        # the same missing column five more times.
        return problems

    if leaid_col and leaid_col in df.columns:
        blank = int(df[leaid_col].isna().sum())
        if blank:
            problems.append(f"{label} {blank} row(s) with no {leaid_col}. A "
                            f"row nothing can be joined to is not data.")

        if known_ids:
            strays = sorted(set(df[leaid_col].dropna()) - set(known_ids))
            if strays:
                # Nearly always a leading zero eaten by a spreadsheet, which is
                # why the message says so rather than just listing them.
                problems.append(
                    f"{label} {len(strays)} leaid(s) not on the roster: "
                    f"{strays[:5]}. Check for a dropped leading zero.")

    for col in date_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            examples = df.loc[parsed.isna(), col].astype(str).unique()[:3]
            problems.append(f"{label} {bad} unparsable {col} value(s), e.g. "
                            f"{list(examples)}.")

    return problems


def report(problems, stream=None):
    """Print problems the way Actions renders them, and say whether to fail.

    ::error:: gets a red annotation in the Actions UI. The non-zero exit is
    what actually sends the mail, so callers must use the return value.
    """
    import sys

    stream = stream or sys.stderr
    if not problems:
        return 0
    for problem in problems:
        print(f"::error::{problem}")
        print(f"WILL NOT COMMIT: {problem}", file=stream)
    return 1


def gate(df, name, **kw):
    """check + report in one call. Returns an exit code, never raises.

    The convenience form, because a caller that has to remember two calls will
    eventually remember one.
    """
    return report(check(df, name, **kw))
