"""What actually happened, from every source that can say so.

Three sources answer "did this district close", they arrive at different times
with different authority, and they cover different districts. This module is
the one place that knows how to merge them.

    manual    data/outcomes/overrides_{season}.csv    typed by a human
    calendar  data/outcomes/calendar_{season}.csv     reconciled offline
    news      data/outcomes/outcomes_{season}.csv     the WBNG feed

Precedence is manual, then calendar, then news. A human who took the trouble to
type it outranks a reconciliation, which outranks a scrape.

THE SOURCES ARE NEVER MERGED ON DISK, only in memory. Keeping the files apart
is what lets us ask, next June, how often the feed was actually right. Fold
them together at write time and that question stops having an answer.

UNKNOWN IS NOT "OPEN", AND THAT IS THE WHOLE POINT OF THE COVERAGE TABLE.

The WBNG viewing area is not the SCRIC region. It reaches well into
Pennsylvania and it does not reach every district we forecast, so there will be
districts the feed simply never carries. Before this module existed, the
scorecard read "no closure on file" as "the district was open", which handed
the tool a free correct call for every uncovered district on every morning, and
scored a real closure at one of them as a miss. Fifty districts times a hundred
mornings of flattery, in the direction nobody would ever notice.

So a district-morning counts as OBSERVED only when a source that actually
covers that district reported that day:

    manual    an override row exists for that district and date
    calendar  the district is calendar-covered and the reconciler ran that day
    news      the district is news-covered and the feed was checked that day

`news` coverage starts as "unknown" for everybody, because we have never seen a
non-empty payload and cannot know who is in it. Unknown does not count as
covered. It is promoted to "yes" per district by evidence, once the feed has
actually carried that district; `seen_in_news()` reports who qualifies.

The consequence is deliberate: until the reconciler is running, the scorecard
scores nothing. An empty record is the correct answer to "how accurate have we
been" when we have not yet observed a morning we can vouch for.
"""

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTCOME_DIR = os.path.join(ROOT, "data", "outcomes")
COVERAGE_CSV = os.path.join(ROOT, "reference", "district_coverage.csv")

# Most authoritative first. Everything in this module iterates in this order.
SOURCES = ("manual", "calendar", "news")

# The file prefix each source writes. Deliberately distinct rather than
# `outcomes_calendar_`: the readers scan a directory by prefix, and a calendar
# file whose name starts with `outcomes_` would be swallowed by the news reader
# and silently double-counted.
PREFIX = {"manual": "overrides_", "calendar": "calendar_", "news": "outcomes_"}

# Per-source log of "we looked on this date", which is what makes the absence
# of a closure meaningful. manual has none: typing an override IS the evidence.
CHECKS = {"calendar": "calendar_checks.csv", "news": "checks.csv"}

COLUMNS = ["date", "leaid", "status", "source"]


def _season_of(date):
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def _read_prefixed(directory, prefix):
    """Every CSV in a directory starting with prefix, concatenated.

    A truncated file costs its own rows and nothing else. On a snowy morning
    the alternative is that one bad file takes down the whole evening.
    """
    if not os.path.isdir(directory):
        return pd.DataFrame()
    frames = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith(prefix) and name.endswith(".csv")):
            continue
        try:
            frames.append(pd.read_csv(os.path.join(directory, name),
                                      dtype={"leaid": str}))
        except (OSError, ValueError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_source(source, out_dir=OUTCOME_DIR):
    """One source's rows, normalized to date / leaid / status / source."""
    df = _read_prefixed(out_dir, PREFIX[source])
    if df.empty or "leaid" not in df.columns or "status" not in df.columns:
        return pd.DataFrame(columns=COLUMNS)

    out = df[["date", "leaid", "status"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].notna() & out["leaid"].notna()]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["source"] = source
    # The feed is archived several times a morning and restates itself, so the
    # same district-day arrives more than once. Counting rows rather than
    # district-days is what burns the snow-day bank down too fast.
    return out.drop_duplicates(subset=["date", "leaid", "source"])


def load_all(out_dir=OUTCOME_DIR):
    """Every source, stacked, still tagged. Not yet resolved."""
    frames = [load_source(s, out_dir) for s in SOURCES]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def resolve(stacked=None, out_dir=OUTCOME_DIR):
    """One row per district-morning, the most authoritative source winning."""
    df = load_all(out_dir) if stacked is None else stacked
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    rank = {s: i for i, s in enumerate(SOURCES)}
    df = df.copy()
    df["_rank"] = df["source"].map(rank)
    # A source outside SOURCES cannot be ranked, and guessing where to put it
    # would be worse than dropping it loudly on read.
    df = df[df["_rank"].notna()]
    df = (df.sort_values(["date", "leaid", "_rank"])
            .drop_duplicates(subset=["date", "leaid"], keep="first"))
    return df[COLUMNS].sort_values(["date", "leaid"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Coverage: which sources can speak for which district
# --------------------------------------------------------------------------

def load_coverage(path=COVERAGE_CSV):
    """The coverage table, or an empty one if it has not been built.

    An empty table means nothing is covered, so the scorecard scores nothing.
    That is the safe direction: the failure mode of a missing coverage file is
    an empty accuracy record, not a flattering one.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["leaid", "district", "news", "calendar"])
    df = pd.read_csv(path, dtype=str).fillna("unknown")
    for col in ("news", "calendar"):
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].str.strip().str.lower()
    return df


def covered_by(coverage, source):
    """LEAIDs a source can speak for. Only an explicit "yes" counts."""
    if coverage is None or coverage.empty or source not in coverage.columns:
        return set()
    return set(coverage.loc[coverage[source] == "yes", "leaid"])


def load_checks(source, out_dir=OUTCOME_DIR):
    """Dates a source actually reported on. Empty frame if it never has."""
    name = CHECKS.get(source)
    if name is None:
        return pd.DataFrame(columns=["date", "ok"])
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "ok"])
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=["date", "ok"])


def _ok_dates(checks):
    if checks is None or checks.empty or "date" not in checks.columns:
        return set()
    if "ok" in checks.columns:
        checks = checks[checks["ok"].astype(bool)]
    return set(checks["date"].astype(str))


def observed(coverage=None, checks_by_source=None, stacked=None,
             out_dir=OUTCOME_DIR):
    """The (date, leaid) pairs we can honestly score.

    Everything else is unknown, and unknown is not "open". See the module
    docstring for why this distinction is the whole point of the file.

    checks_by_source lets a test supply the per-source logs directly; left out,
    they are read from disk.
    """
    coverage = load_coverage() if coverage is None else coverage
    stacked = load_all(out_dir) if stacked is None else stacked

    keys = set()

    # manual: the override row is its own evidence. No coverage test and no
    # check log, because a human typed this district and this date.
    if len(stacked):
        manual = stacked[stacked["source"] == "manual"]
        keys |= set(zip(manual["date"].astype(str), manual["leaid"]))

    for source in ("calendar", "news"):
        districts = covered_by(coverage, source)
        if not districts:
            continue
        checks = (checks_by_source or {}).get(source)
        if checks is None:
            checks = load_checks(source, out_dir)
        for date in _ok_dates(checks):
            keys |= {(date, leaid) for leaid in districts}

    return keys


def seen_in_news(out_dir=OUTCOME_DIR):
    """Districts the feed has actually carried at least once.

    The promotion list for the coverage table. A district only appears in the
    feed when it announces something, so "never seen" stays ambiguous between
    "not covered" and "covered and never closed" -- which is exactly why
    coverage is promoted by evidence and never assumed.
    """
    news = load_source("news", out_dir)
    return set(news["leaid"]) if len(news) else set()


def closures(stacked=None, out_dir=OUTCOME_DIR, statuses=("closed",)):
    """Resolved closures only, in the leaid/date/kind shape features expects."""
    res = resolve(stacked, out_dir)
    if res.empty:
        return pd.DataFrame(columns=["leaid", "date", "kind"])
    shut = res[res["status"].isin(statuses)].copy()
    if shut.empty:
        return pd.DataFrame(columns=["leaid", "date", "kind"])
    shut["date"] = pd.to_datetime(shut["date"])
    shut["kind"] = "closed"
    return shut[["leaid", "date", "kind"]].drop_duplicates()


def build_coverage(roster, extras=None):
    """The starting coverage table, derived rather than typed.

    calendar covers everyone the annual export reaches, which is everyone not
    listed in extra_districts.csv. news starts unknown for all fifty, because
    we have never seen a payload and will not pretend to know who is in it.
    """
    out = roster[["leaid", "district"]].copy()
    off_export = set(extras["leaid"]) if extras is not None and len(extras) else set()
    out["calendar"] = ["no" if l in off_export else "yes" for l in out["leaid"]]
    out["news"] = "unknown"
    out["note"] = ["not covered by the calendar export" if l in off_export else ""
                   for l in out["leaid"]]
    return out.sort_values("district").reset_index(drop=True)


if __name__ == "__main__":
    roster = pd.read_csv(os.path.join(ROOT, "reference", "roster.csv"),
                         dtype={"leaid": str})
    extras_path = os.path.join(ROOT, "reference", "extra_districts.csv")
    extras = (pd.read_csv(extras_path, dtype={"leaid": str})
              if os.path.exists(extras_path) else None)

    table = build_coverage(roster, extras)
    table.to_csv(COVERAGE_CSV, index=False)
    print(f"wrote {COVERAGE_CSV} ({len(table)} districts)")
    print(f"  calendar: {(table['calendar'] == 'yes').sum()} yes, "
          f"{(table['calendar'] == 'no').sum()} no")
    print(f"  news:     {(table['news'] == 'unknown').sum()} unknown")

    seen = seen_in_news()
    if seen:
        print(f"\nthe feed has carried {len(seen)} district(s). Promote them "
              f"to news=yes:")
        for leaid in sorted(seen):
            hit = table[table["leaid"] == leaid]
            name = hit["district"].iloc[0] if len(hit) else "(not on roster)"
            print(f"  {leaid}  {name}")
