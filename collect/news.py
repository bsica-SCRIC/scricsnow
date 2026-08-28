"""Live school closings from the WBNG feed.

The feed is a static JSON file on S3, written by Gray Digital Media's closings
system. No auth, no cookies, no JavaScript, no cache-buster needed:

    https://s3.amazonaws.com/grayfilestore-wbng/closingsData/closings_WBNG.json

Two consumers, with opposite needs:

    the app     wants freshness. It calls fetch() on page load and shows what
                is closed right now. If this breaks, the banner disappears and
                nothing else is affected.
    the archive wants persistence. One run a morning records the settled
                outcome. For the two districts with no history behind them it
                is the only source of outcomes there is.

WE HAVE NEVER SEEN A NON-EMPTY PAYLOAD. In August `locations` is `[]`, so the
record shape is a guess until the first real closing, probably in December.
Everything below is built around that:

  - the raw JSON is archived verbatim whenever it is non-empty, BEFORE any
    parsing, so a wrong guess costs nothing that cannot be recovered
  - field names are probed from a list of candidates rather than assumed
  - every parsed row keeps the original record in a `raw` column
  - unparsed records are reported loudly, not skipped quietly

When the first closing arrives, read data/outcomes/raw/, correct the candidate
lists below, and re-parse the archive. Nothing is lost in the meantime.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

FEED = ("https://s3.amazonaws.com/grayfilestore-wbng/closingsData/"
        "closings_WBNG.json")
LOCAL_TZ = ZoneInfo("America/New_York")

ROOT = os.path.join(os.path.dirname(__file__), "..")
ROSTER_CSV = os.path.join(ROOT, "reference", "roster.csv")
OUT_DIR = os.path.join(ROOT, "data", "outcomes")
RAW_DIR = os.path.join(OUT_DIR, "raw")

HEADERS = {"User-Agent": "scricsnow/0.1 (snow day predictor; bsica@btboces.org)",
           "Accept": "application/json"}

# Candidate field names, most likely first. Extend from the first real payload.
NAME_KEYS = ["name", "location", "location_name", "title", "organization",
             "school", "district", "institution", "displayName"]
STATUS_KEYS = ["status", "closing_status", "closing_type", "type", "message",
               "statusText", "condition", "note"]

# Text patterns to normalize a status into the vocabulary the model uses.
STATUS_PATTERNS = [
    (r"\bclos(ed|ing|ure)\b|\bcancel", "closed"),
    (r"\b(\d+\s*(hour|hr)|two|one)\b.*\bdelay|delay|late\s*(start|arrival)|"
     r"\bdelayed\b", "delay"),
    (r"early\s*(dismissal|release|out)", "early"),
    (r"\bremote\b|\bvirtual\b|\bonline\b", "remote"),
    (r"\bopen\b|\bnormal\b|\bas\s*scheduled\b", "open"),
]


def fetch(url=FEED, timeout=30, retries=3):
    """Return (payload, meta). Never raises for the caller's convenience: the
    app treats a failure as "no banner", not as an error page."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json(), {
                "ok": True,
                "last_modified": r.headers.get("last-modified"),
                "fetched_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
            }
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(2 ** attempt)
    return None, {"ok": False, "error": str(last)[:200],
                  "fetched_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds")}


def locations_of(payload):
    """Pull the record list out, tolerating a couple of plausible shapes."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    out = []
    for block in payload:
        if isinstance(block, dict):
            out.extend(block.get("locations") or [])
    return out


def _first_key(record, candidates):
    lowered = {str(k).lower(): k for k in record}
    for cand in candidates:
        if cand.lower() in lowered:
            value = record[lowered[cand.lower()]]
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def normalize_status(text):
    """Map free text onto closed / delay / early / remote / open, or None.

    Order matters: "closed" is checked before "delay" so that a record reading
    "closed, no delay" is not filed as a delay.
    """
    if not text:
        return None
    low = text.lower()
    for pattern, label in STATUS_PATTERNS:
        if re.search(pattern, low):
            return label
    return None


# Words that carry no identity: every district has some of them, and feeds
# abbreviate them inconsistently ("Binghamton City School District",
# "Binghamton City SD", "BINGHAMTON CITY SCHOOLS" are one district).
GENERIC_WORDS = {
    "school", "schools", "district", "districts", "dist", "central",
    "city", "union", "free", "common", "csd", "ufsd", "sd", "cs",
    "the", "of", "at",
}


def _norm_name(text):
    """Reduce a name to its identifying tokens, as a stable sorted key.

    Token comparison rather than string comparison, because the generic words
    appear in different orders and abbreviations. Note "city" is generic even
    though Johnson City is a place: dropping it leaves {johnson}, which is
    still unique in this roster, while keeping it would break the match on
    "Johnson City CSD".
    """
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    tokens = [t for t in cleaned.split() if t and t not in GENERIC_WORDS]
    return " ".join(sorted(tokens))


def match_roster(names, roster):
    """Map feed names to LEAIDs. Exact-ish on a normalized key only.

    Deliberately conservative. A feed carries every school, church and daycare
    in the viewing area, so loose matching would attach a daycare closure to a
    district. Unmatched names are returned for review rather than guessed at.
    """
    lookup = {}
    collisions = set()
    for leaid, district in zip(roster["leaid"], roster["district"]):
        key = _norm_name(district)
        if key in lookup and lookup[key] != leaid:
            collisions.add(key)
        lookup[key] = leaid
    # Two districts reducing to the same key cannot be told apart, so neither
    # is matched. Silently picking one would attach closures to the wrong
    # district, which is worse than reporting nothing.
    for key in collisions:
        lookup.pop(key, None)

    aliases = load_aliases()
    out = {}
    for name in names:
        if name in aliases:
            out[name] = aliases[name]
            continue
        key = _norm_name(name)
        if key in lookup:
            out[name] = lookup[key]
    return out


def load_aliases(path=None):
    """Hand-maintained feed-name overrides, for names token matching misses.

    Expected to stay empty until the first real payload shows which names the
    feed actually uses. Two columns: feed_name, leaid.
    """
    path = path or os.path.join(ROOT, "reference", "district_aliases.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["feed_name"], df["leaid"]))


def parse(payload, roster=None):
    """Records to a tidy frame. Keeps the original dict on every row."""
    records = locations_of(payload)
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            rows.append({"name": None, "status_text": None, "status": None,
                         "raw": json.dumps(rec)})
            continue
        name = _first_key(rec, NAME_KEYS)
        status_text = _first_key(rec, STATUS_KEYS)
        rows.append({
            "name": name,
            "status_text": status_text,
            "status": normalize_status(status_text),
            "raw": json.dumps(rec, sort_keys=True),
        })

    df = pd.DataFrame(rows, columns=["name", "status_text", "status", "raw"])
    if roster is not None and len(df):
        matched = match_roster([n for n in df["name"] if n], roster)
        df["leaid"] = df["name"].map(matched)
    else:
        df["leaid"] = None
    return df


# Fraction of records that must parse before we believe our field guesses.
# If NAME_KEYS or STATUS_KEYS are wrong, EVERY record fails, so the real signal
# is near-total failure. A threshold at half separates that cleanly from one odd
# entry in an otherwise fine payload, and never fires on noise.
PARSE_FLOOR = 0.5


def ever_matched(out_dir=OUT_DIR):
    """Have we ever successfully written an outcome for a tracked district?

    Read from the outcome files rather than tracked in state, because the whole
    point is to survive a fresh checkout on a GitHub runner.
    """
    if not os.path.isdir(out_dir):
        return False
    for name in os.listdir(out_dir):
        if not (name.startswith("outcomes_") and name.endswith(".csv")):
            continue
        try:
            df = pd.read_csv(os.path.join(out_dir, name), dtype={"leaid": str})
        except (OSError, ValueError):
            continue
        if len(df) and df["leaid"].notna().any():
            return True
    return False


def health(df, records, matched_before):
    """Problems worth FAILING the job over. Pure, and deliberately narrow.

    This exists because the interesting failure here is silent. We have never
    seen a non-empty payload, so NAME_KEYS and STATUS_KEYS are educated guesses.
    If they are wrong, the first snowy morning archives the raw JSON, matches
    nothing, writes no outcomes, prints a warning into a log nobody reads, and
    exits zero. The scorecard stays blank and, since the model now reads
    closed_prev from these files, every district also reads as "has not closed
    recently" on the second morning of the storm. Two failures, no alarm.

    A GitHub Actions failure emails the repo owner, so a non-zero exit is the
    only alerting channel this project has. That makes false alarms expensive:
    cry wolf in January and the real one gets ignored. So the checks below are
    only the unambiguous ones.

    NOT an error, deliberately: a payload where everything parses and none of it
    is ours. The WBNG viewing area is far larger than these 50 districts, and a
    morning where some churches and daycares close and no tracked district does
    is an ordinary morning, not a bug.
    """
    problems = []
    if not records:
        return problems

    if not len(df):
        problems.append(
            f"feed returned {records} record(s) and the parser produced no "
            f"rows at all. locations_of() is not finding the list. Read the "
            f"archived raw JSON in data/outcomes/raw/.")
        return problems

    unnamed = int(df["name"].isna().sum())
    unstatused = int(df["status"].isna().sum())
    if unnamed > len(df) * PARSE_FLOOR:
        problems.append(
            f"no name found on {unnamed}/{len(df)} records. NAME_KEYS does not "
            f"match this feed. Read data/outcomes/raw/ and extend it.")
    if unstatused > len(df) * PARSE_FLOOR:
        problems.append(
            f"no status found on {unstatused}/{len(df)} records. STATUS_KEYS "
            f"or STATUS_PATTERNS does not match this feed. Read "
            f"data/outcomes/raw/ and extend them.")

    # Fires at most once in the project's life: on the first payload that has
    # real content and still matches none of our districts. After one success
    # the roster matching is proven and this never speaks again.
    matched = int(df["leaid"].notna().sum())
    if matched == 0 and not matched_before and not problems:
        problems.append(
            f"{len(df)} record(s) parsed and NONE matched the roster, and no "
            f"tracked district has ever been matched. This may simply be a "
            f"morning when none of our 50 closed, in which case rerun to "
            f"clear it. But it is the first live payload, so check the names "
            f"in data/outcomes/raw/ against reference/roster.csv and add any "
            f"overrides to reference/district_aliases.csv.")
    return problems


def archive_raw(payload, meta, raw_dir=RAW_DIR):
    """Store the payload verbatim before anything interprets it."""
    os.makedirs(raw_dir, exist_ok=True)
    stamp = datetime.now(LOCAL_TZ).strftime("%Y-%m-%dT%H%M")
    path = os.path.join(raw_dir, f"closings_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "payload": payload}, fh, indent=2)
    return path


def log_check(meta, target_date, out_dir=OUT_DIR):
    """Record that we looked, whether or not anything was closed.

    Without this, "no closures on file for 12 January" is ambiguous between
    "nobody closed" and "the job did not run". The scorecard can only score
    mornings we actually observed, so it needs to know which those were.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "checks.csv")
    row = pd.DataFrame([{
        "date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
        "checked_at": meta.get("fetched_at"),
        "ok": bool(meta.get("ok")),
        "records": int(meta.get("records", 0)),
        "last_modified": meta.get("last_modified"),
    }])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["date"] != row["date"].iloc[0]]
        row = pd.concat([old, row], ignore_index=True)
    row.sort_values("date").to_csv(path, index=False)
    return path


def write_outcomes(df, target_date, out_dir=OUT_DIR):
    """Append matched closures to the season file, replacing the same day."""
    if df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)
    d = pd.Timestamp(target_date)
    season = d.year if d.month >= 7 else d.year - 1
    path = os.path.join(out_dir, f"outcomes_{season}.csv")

    df = df.copy()
    df["date"] = d.strftime("%Y-%m-%d")
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"leaid": str})
        old = old[old["date"] != df["date"].iloc[0]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def collect(url=FEED, roster=None, verbose=True, raw_dir=None):
    """One archive-and-parse cycle. Returns (frame, meta).

    raw_dir is threaded through rather than left to archive_raw's default so a
    test can point the whole run at a temporary directory. Binding it as a
    default deep in the call stack is what made the alarm below untestable, and
    an untested alarm is not much better than no alarm.
    """
    if roster is None:
        roster = pd.read_csv(ROSTER_CSV, dtype={"leaid": str})

    payload, meta = fetch(url)
    if not meta["ok"]:
        if verbose:
            print(f"feed unavailable: {meta['error']}")
        return pd.DataFrame(), meta

    records = locations_of(payload)
    meta["records"] = len(records)

    # Archive only non-empty payloads. An empty feed is the normal state for
    # most of the year and archiving it daily would bury the useful ones.
    if records:
        meta["raw_path"] = archive_raw(payload, meta,
                                       raw_dir=raw_dir or RAW_DIR)

    df = parse(payload, roster)
    if verbose:
        print(f"feed: {len(records)} record(s), last modified "
              f"{meta.get('last_modified')}")
        if len(df):
            unnamed = int(df["name"].isna().sum())
            unstatused = int(df["status"].isna().sum())
            matched = int(df["leaid"].notna().sum())
            print(f"  parsed name on {len(df) - unnamed}/{len(df)}, "
                  f"status on {len(df) - unstatused}/{len(df)}, "
                  f"matched to roster: {matched}")
            if unnamed or unstatused:
                print("  SOME FIELDS UNRECOGNIZED. Read the archived raw JSON "
                      "and extend NAME_KEYS / STATUS_KEYS in collect/news.py.")
    return df, meta


def main(argv=None, out_dir=None, raw_dir=None, url=FEED, roster=None):
    """One run. RETURNS an exit code rather than calling sys.exit.

    A function with injectable paths, not a bare __main__ block, so the alarm
    can be tested. The whole value of exiting non-zero is that somebody gets an
    email on a snowy morning; shipping that untested would be its own kind of
    silent failure.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = out_dir or OUT_DIR
    raw_dir = raw_dir or os.path.join(out_dir, "raw")

    # Whether we have ever matched anybody must be read BEFORE this run writes
    # its own outcomes, or a successful first match would mask itself.
    matched_before = ever_matched(out_dir)

    df, meta = collect(url=url, roster=roster, raw_dir=raw_dir)
    target = (argv[0] if argv and not argv[0].startswith("--")
              else datetime.now(LOCAL_TZ).date().isoformat())
    log_check(meta, target, out_dir=out_dir)

    if len(df):
        keep = df[df["leaid"].notna()].copy()
        if len(keep):
            path = write_outcomes(
                keep[["leaid", "name", "status", "status_text"]], target,
                out_dir=out_dir)
            print(f"wrote {path} ({len(keep)} tracked district(s))")
        else:
            print("no tracked districts in the feed right now")
        print()
        print(df.head(20).to_string(index=False))
    elif meta.get("ok"):
        print("feed is empty; nothing closed right now")

    # Everything above has already been written: the raw JSON, the checks row,
    # and any outcomes we could match. Failing now costs no data and is purely
    # a way to get somebody's attention. The workflow's commit step runs on
    # !cancelled() so the evidence is still committed.
    problems = health(df, int(meta.get("records", 0)), matched_before)
    if not problems:
        return 0
    print()
    for problem in problems:
        # ::error:: renders as a red annotation in the Actions UI. The non-zero
        # return is what actually sends the mail.
        print(f"::error::{problem}")
        print(f"FEED PROBLEM: {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
