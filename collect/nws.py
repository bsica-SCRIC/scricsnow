"""NWS gridpoint forecasts, and the nightly snapshot that feeds the model.

THIS IS THE JOB THAT CANNOT BE MISSED. api.weather.gov serves no forecast
history: once tomorrow becomes today, tonight's forecast is gone permanently.
Closure outcomes can be reconstructed after the fact, so every other collector
can fail and be repaired. This one cannot. Robustness here is worth more than
features.

Consequences for the design:
  - one district failing never aborts the run
  - a partial night is written, not discarded
  - grid coordinates are cached permanently, so a /points outage cannot stop a
    snapshot
  - re-running the same night overwrites rather than duplicating

The forecast window is 18:00 the evening before through 09:00 on the target
morning, local time. That covers the overnight accumulation a superintendent is
looking at when making a roughly 5am call.
"""

import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API = "https://api.weather.gov"
LOCAL_TZ = ZoneInfo("America/New_York")

REF = os.path.join(os.path.dirname(__file__), "..", "reference")
ROSTER_CSV = os.path.join(REF, "roster.csv")
GRIDS_CSV = os.path.join(REF, "district_grids.csv")
FORECAST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "forecasts")
HOURLY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hourly")

# api.weather.gov asks for a User-Agent identifying the application and a
# contact. Requests without one get throttled or refused.
HEADERS = {
    "User-Agent": "scricsnow/0.1 (snow day predictor; bsica@btboces.org)",
    "Accept": "application/geo+json",
}

# Hourly snowfall at or above this counts as "it is snowing", for deciding when
# a storm starts and ends. Below this is flurries that plough nothing.
SNOWING_MM = 0.5

# Two captures per night, kept as separate rows rather than one overwriting the
# other. They answer different questions:
#
#   evening  ~20:00, what a parent sees at bedtime. This is the row the model
#            trains on, because it is the moment the prediction is actually
#            wanted and it holds lead time constant across nights.
#   morning  ~05:00, hours before the superintendent decides. Far more
#            accurate, and useless as a product on its own since the real
#            answer arrives at the same time. Its value is measuring how much
#            certainty the last nine hours buy, which tells us how much the
#            bedtime number should hedge.
#
# Rows are keyed on (target_date, leaid, capture_slot), so the two coexist.
EVENING, MORNING = "evening", "morning"
MORNING_BEFORE_HOUR = 12

_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?)?$")


def parse_duration_hours(text):
    """ISO-8601 duration to hours. NWS uses PT1H, PT3H, P1DT6H and similar."""
    m = _DURATION.match(text)
    if not m:
        raise ValueError(f"unparsable duration: {text}")
    d = {k: int(v) for k, v in m.groupdict(default="0").items()}
    return d["days"] * 24 + d["hours"] + d["minutes"] / 60.0


def get(url, timeout=60, retries=3, backoff=2):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"NWS request failed after {retries} tries: {url}: {last}")


def lookup_grid(lat, lon):
    """Map a point to its NWS forecast grid cell. Stable, so cache it."""
    j = get(f"{API}/points/{lat:.4f},{lon:.4f}")
    p = j["properties"]
    return {"office": p["gridId"], "grid_x": p["gridX"], "grid_y": p["gridY"]}


def ensure_grids(roster, sleep=0.3):
    """Return roster with office/grid_x/grid_y, fetching only what is missing.

    Cached permanently: NWS grids do not move, and a /points outage on a snowy
    night must never be the reason a snapshot is lost.
    """
    if os.path.exists(GRIDS_CSV):
        cache = pd.read_csv(GRIDS_CSV, dtype={"leaid": str})
    else:
        cache = pd.DataFrame(columns=["leaid", "office", "grid_x", "grid_y"])

    missing = roster[~roster["leaid"].isin(set(cache["leaid"]))]
    rows = []
    for _, d in missing.iterrows():
        try:
            g = lookup_grid(d["lat"], d["lon"])
            rows.append({"leaid": d["leaid"], **g})
        except RuntimeError as e:
            print(f"  grid lookup failed for {d['district']}: {e}")
        time.sleep(sleep)

    if rows:
        new = pd.DataFrame(rows)
        cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
        cache.to_csv(GRIDS_CSV, index=False)

    return roster.merge(cache, on="leaid", how="left")


def expand_series(prop, is_accumulation=False):
    """Flatten one gridpoint property into hourly rows.

    NWS gives each value a validTime like "2026-01-15T18:00:00+00:00/PT3H",
    meaning the value covers a span. Expanding to hours makes windows and
    storm-end timing straightforward.

    is_accumulation matters. Snowfall, ice and QPF are TOTALS for their block,
    so a 3-hour block holding 6 mm must become three hours of 2 mm; copying 6
    into each hour would triple the storm. Temperature and wind are states, so
    they are copied unchanged. Getting this backwards silently inflates every
    multi-hour forecast block, which is most of them beyond about 36 hours out.
    """
    if not prop or "values" not in prop:
        return pd.DataFrame(columns=["time", "value"])

    rows = []
    for item in prop["values"]:
        stamp, _, dur = item["validTime"].partition("/")
        start = datetime.fromisoformat(stamp).astimezone(LOCAL_TZ)
        hours = max(1, int(round(parse_duration_hours(dur))))
        value = item["value"]
        if is_accumulation and value is not None:
            value = value / hours
        for h in range(hours):
            rows.append({"time": start + timedelta(hours=h), "value": value})
    return pd.DataFrame(rows)


def _hourly(props, name, is_accumulation=False):
    return expand_series(props.get(name), is_accumulation=is_accumulation)


def window_bounds(target_date):
    """18:00 the evening before through 09:00 on the target morning, local."""
    end = datetime.combine(target_date, datetime.min.time(),
                           tzinfo=LOCAL_TZ) + timedelta(hours=9)
    start = end - timedelta(hours=15)
    return start, end


def _sum_in(df, start, end):
    if df.empty:
        return 0.0
    m = (df["time"] >= start) & (df["time"] < end)
    return float(df.loc[m, "value"].fillna(0).sum())


def _at(df, when):
    if df.empty:
        return None
    m = df["time"] == when
    if not m.any():
        return None
    v = df.loc[m, "value"].iloc[0]
    return None if pd.isna(v) else float(v)


def _extreme_in(df, start, end, how="max"):
    if df.empty:
        return None
    m = (df["time"] >= start) & (df["time"] < end)
    vals = df.loc[m, "value"].dropna()
    if not len(vals):
        return None
    return float(vals.max() if how == "max" else vals.min())


def _max_in(df, start, end):
    return _extreme_in(df, start, end, "max")


def _min_in(df, start, end):
    return _extreme_in(df, start, end, "min")


def features(props, target_date):
    """Reduce a gridpoint forecast to one row of overnight features."""
    start, end = window_bounds(target_date)

    snow = _hourly(props, "snowfallAmount", is_accumulation=True)
    ice = _hourly(props, "iceAccumulation", is_accumulation=True)
    qpf = _hourly(props, "quantitativePrecipitation", is_accumulation=True)
    temp = _hourly(props, "temperature")
    wind = _hourly(props, "windSpeed")
    gust = _hourly(props, "windGust")
    chill = _hourly(props, "windChill")
    pop = _hourly(props, "probabilityOfPrecipitation")

    midnight = end - timedelta(hours=9)
    six_am = end - timedelta(hours=3)

    # When the snow stops is arguably the most important single feature: the
    # same storm total gets a delay if it ends at 4am and a closure if it is
    # still falling at 7am, because ploughs need a few clear hours.
    snowing = snow[(snow["time"] >= start) & (snow["time"] < end)
                   & (snow["value"].fillna(0) >= SNOWING_MM)]
    if len(snowing):
        snow_start = snowing["time"].min()
        snow_end = snowing["time"].max()
        start_hour = snow_start.hour + (24 if snow_start.date() == target_date else 0)
        end_hour = snow_end.hour + (24 if snow_end.date() == target_date else 0)
        ongoing = bool(snow_end >= end - timedelta(hours=1))
    else:
        start_hour = end_hour = None
        ongoing = False

    return {
        "snow_mm": round(_sum_in(snow, start, end), 2),
        "snow_mm_overnight": round(_sum_in(snow, midnight, end), 2),
        "ice_mm": round(_sum_in(ice, start, end), 3),
        "qpf_mm": round(_sum_in(qpf, start, end), 2),
        "pop_max": _max_in(pop, start, end),
        "temp_6am_c": _at(temp, six_am),
        "temp_min_c": _min_in(temp, start, end),
        "windchill_6am_c": _at(chill, six_am),
        "wind_max_kmh": _max_in(wind, start, end),
        "gust_max_kmh": _max_in(gust, start, end),
        # Hours run on an 18-33 scale: 18-23 is the evening before, 24-33 is
        # the target morning. The +24 goes on the MORNING so that later always
        # means larger; putting it on the evening inverts the ordering and
        # makes a storm ending at 4am look later than one ending at 10pm.
        "snow_start_hour": start_hour,
        "snow_end_hour": end_hour,
        "snow_ongoing_at_9am": ongoing,
    }


# The hourly series kept alongside the feature row, and the reason it exists.
#
# features() above reduces about fifteen hourly series to eighteen numbers and
# discards the rest. Three of those numbers -- snow_start_hour, snow_end_hour,
# snow_ongoing_at_9am -- are among the strongest signals we have for telling a
# delay from a closure, and the FITTED model cannot use any of them, because it
# is trained on GHCN daily observations which carry no hourly detail. Only the
# hand-built heuristic scores them.
#
# There are two ways out of that and they are not competitors. NOAA ISD gives
# hourly observations back fourteen seasons and is backfillable next summer
# from the endpoint ghcn.py already calls, so it needs no live job. What is NOT
# recoverable is our own forecast series: api.weather.gov serves no history, so
# every hour not written down tonight is gone. Training on the archived
# forecast also avoids the mismatch features.py spends two pages on, because
# the column then means the same thing in training as it does in service.
#
# Kept flat rather than as raw JSON. The full gridpoint payload is about 7 GB a
# season and does not belong in git; these columns are the ones anybody would
# go back for, and they come to roughly 30 MB.
HOURLY_COLUMNS = [
    "target_date", "leaid", "capture_slot", "captured_at",
    "time_local", "hour",
    "snow_mm", "ice_mm", "qpf_mm", "temp_c", "wind_kmh", "gust_kmh", "pop",
]

# Which series are per-block TOTALS rather than states, and how many decimals
# each is worth keeping. Same accumulation distinction expand_series() makes,
# restated here so the two cannot drift apart.
#
# THE DECIMALS ARE NOT COSMETIC. NWS serves temperature in Celsius converted
# from whole degrees Fahrenheit, so an unrounded value arrives as
# 25.555555555555557: seventeen characters of which two are information. Across
# fifty districts, three slots and two hundred nights that fake precision is
# the difference between roughly 58 MB a season and 30. Every figure below is
# still finer than the forecast it came from.
HOURLY_SERIES = {
    "snow_mm": ("snowfallAmount", True, 3),
    "ice_mm": ("iceAccumulation", True, 3),
    "qpf_mm": ("quantitativePrecipitation", True, 3),
    "temp_c": ("temperature", False, 2),
    "wind_kmh": ("windSpeed", False, 2),
    "gust_kmh": ("windGust", False, 2),
    "pop": ("probabilityOfPrecipitation", False, 0),
}


def hourly(props, target_date):
    """The forecast window hour by hour, one row per hour.

    Accumulations are PER HOUR, not running totals: a three-hour block holding
    6 mm becomes three rows of 2 mm, which is what expand_series already does
    and what makes the rows sum back to the feature row's totals.

    `hour` uses the same 18-33 scale as features(): 18-23 is the evening
    before, 24 onward is the target morning, so later always means larger.
    `time_local` carries the same instant unambiguously, because this file is
    written to be read next summer by somebody who has forgotten the scale.
    """
    start, end = window_bounds(target_date)

    series = {name: _hourly(props, prop, is_accumulation=acc)
              for name, (prop, acc, _) in HOURLY_SERIES.items()}
    at = {}
    for name, frame in series.items():
        if frame.empty:
            at[name] = {}
            continue
        clipped = frame[(frame["time"] >= start) & (frame["time"] < end)]
        # Last value wins on a duplicated hour. NWS blocks do not normally
        # overlap, but a re-issued grid can repeat one and a silent doubling
        # would be worse than preferring the newer figure.
        at[name] = {t: v for t, v in zip(clipped["time"], clipped["value"])}

    rows = []
    when = start
    while when < end:
        row = {"time_local": when.isoformat(timespec="minutes"),
               "hour": when.hour + (24 if when.date() == target_date else 0)}
        for name, (_, _, decimals) in HOURLY_SERIES.items():
            value = at[name].get(when)
            row[name] = (None if value is None or pd.isna(value)
                         else round(float(value), decimals))
        rows.append(row)
        when += timedelta(hours=1)
    return rows


def hourly_path(target_date, hourly_dir=HOURLY_DIR):
    """One gzipped file per target night.

    Not one file per season. A day file is written at most three times, once
    per capture slot, and never touched again; appending a season would rewrite
    a growing blob six hundred times a winter and pack badly.

    GZIPPED, unlike every other CSV here, and the reason the rule differs is
    that these files are write-once. `forecast_{season}.csv` is committed
    nightly and re-read by a human, so a readable diff is worth real bytes. A
    day file is never diffed, only added, so there is nothing to read in a
    diff and the bytes buy nothing.

    Measured on a full night of 50 districts x 3 slots x 15 hours: 259 KB
    plain, 43 KB gzipped, which is 53 MB a season against 9. pandas infers the
    compression from the extension on both read and write, including over
    HTTPS, so nothing else has to know.

    The human-readable version of the same night has not gone anywhere: the
    feature row per district is in data/forecasts as plain CSV. This file is
    the archive underneath it.
    """
    stamp = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    return os.path.join(hourly_dir, f"{stamp}.csv.gz")


def write_hourly(df, hourly_dir=HOURLY_DIR):
    """Write the night's hourly rows, replacing this slot's if it re-runs."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    df = pd.DataFrame(df)
    if df.empty:
        return None
    os.makedirs(hourly_dir, exist_ok=True)

    path = hourly_path(df["target_date"].iloc[0], hourly_dir)
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"leaid": str})
        replacing = set(zip(df["target_date"], df["leaid"], df["capture_slot"]))
        keep = [(t, l, s) not in replacing for t, l, s
                in zip(old["target_date"], old["leaid"], old["capture_slot"])]
        df = pd.concat([old[keep], df], ignore_index=True)

    df = df[HOURLY_COLUMNS].sort_values(
        ["target_date", "capture_slot", "leaid", "hour"])
    df.to_csv(path, index=False)
    return path


def current_slot(now=None):
    """Which capture this run is. Derived from the clock, not passed in, so a
    delayed cron firing still lands in the right slot."""
    now = now or datetime.now(LOCAL_TZ)
    return MORNING if now.hour < MORNING_BEFORE_HOUR else EVENING


def season_path(target_date, forecast_dir=FORECAST_DIR):
    target = pd.Timestamp(target_date)
    season = target.year if target.month >= 7 else target.year - 1
    return os.path.join(forecast_dir, f"forecast_{season}.csv")


def already_captured(target_date, slot=EVENING, forecast_dir=FORECAST_DIR):
    """LEAIDs already stored for this target night IN THIS SLOT."""
    path = season_path(target_date, forecast_dir)
    if not os.path.exists(path):
        return set()
    old = pd.read_csv(path, dtype={"leaid": str})
    if "capture_slot" not in old.columns:
        old["capture_slot"] = EVENING
    m = ((old["target_date"] == pd.Timestamp(target_date).strftime("%Y-%m-%d"))
         & (old["capture_slot"] == slot))
    return set(old.loc[m, "leaid"])


def snapshot(roster=None, target_date=None, sleep=0.4, verbose=True,
             only_missing=False, slot=None, hourly_out=None):
    """Fetch one overnight forecast row per district.

    Never raises for a single district. A night with 47 of 50 districts is
    worth far more than no night at all, and the failures are reported so a
    systematic problem is visible rather than silent.

    hourly_out, if given a list, is extended with the hour-by-hour rows for
    every district fetched. An out-parameter rather than a third return value
    on purpose: one gridpoint response has to serve both outputs, and widening
    the return tuple would break track.py, which wants only the feature rows
    and captures far more often than the hourly file should be written.
    """
    if roster is None:
        roster = pd.read_csv(ROSTER_CSV, dtype={"leaid": str})
    now = datetime.now(LOCAL_TZ)
    slot = slot or current_slot(now)
    if target_date is None:
        # A morning run is forecasting TODAY; an evening run forecasts tomorrow.
        target_date = (now.date() if slot == MORNING
                       else (now + timedelta(days=1)).date())

    roster = ensure_grids(roster)

    # only_missing backs up the primary run. GitHub Actions cron is
    # best-effort and occasionally skips a firing entirely; a skipped night is
    # gone permanently. A later run with this flag fills gaps without
    # overwriting rows captured at the intended time, so lead time stays
    # comparable across nights instead of drifting with whenever cron fired.
    if only_missing:
        have = already_captured(target_date, slot)
        roster = roster[~roster["leaid"].isin(have)]
        if verbose:
            print(f"only_missing: {len(roster)} district(s) still to capture")
        if roster.empty:
            return pd.DataFrame(), []

    captured_at = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")

    rows, failures = [], []
    for _, d in roster.iterrows():
        if pd.isna(d.get("office")):
            failures.append((d["district"], "no grid cell cached"))
            continue
        try:
            url = (f"{API}/gridpoints/{d['office']}"
                   f"/{int(d['grid_x'])},{int(d['grid_y'])}")
            props = get(url)["properties"]
            rows.append({
                "target_date": target_date.isoformat(),
                "leaid": d["leaid"],
                "district": d["district"],
                "capture_slot": slot,
                "captured_at": captured_at,
                **features(props, target_date),
            })
            if hourly_out is not None:
                # Same response, second output. A district that failed above
                # contributes to neither, which keeps the two files consistent
                # with each other on a partial night.
                hourly_out.extend(
                    {"target_date": target_date.isoformat(),
                     "leaid": d["leaid"], "capture_slot": slot,
                     "captured_at": captured_at, **hour}
                    for hour in hourly(props, target_date))
        except Exception as e:                      # noqa: BLE001
            failures.append((d["district"], str(e)[:120]))
        time.sleep(sleep)

    if verbose:
        print(f"target {target_date} [{slot}]: captured {len(rows)}/{len(roster)}")
        for name, why in failures:
            print(f"  FAILED {name}: {why}")

    return pd.DataFrame(rows), failures


def write_snapshot(df, forecast_dir=FORECAST_DIR, append_only=False):
    """Append to the season file, replacing any existing rows for that night.

    CSV rather than Parquet on purpose: this file is committed once per night,
    and a text format gives readable diffs and clean merges. A season is about
    50 districts x 120 nights, which is trivial either way.
    """
    if df.empty:
        return None
    os.makedirs(forecast_dir, exist_ok=True)

    path = season_path(df["target_date"].iloc[0], forecast_dir)
    slot = df["capture_slot"].iloc[0]

    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"leaid": str})
        if "capture_slot" not in old.columns:
            old["capture_slot"] = EVENING
        if append_only:
            # Gap-fill: keep what is there, add only districts still missing.
            have = set(zip(old["target_date"], old["leaid"], old["capture_slot"]))
            df = df[[(t, l, s) not in have for t, l, s in
                     zip(df["target_date"], df["leaid"], df["capture_slot"])]]
        else:
            # Idempotent per slot: re-running the evening run replaces only the
            # evening rows and leaves the morning ones alone.
            drop = (old["target_date"].isin(set(df["target_date"]))
                    & (old["capture_slot"] == slot))
            old = old[~drop]
        df = pd.concat([old, df], ignore_index=True)

    df = df.sort_values(["target_date", "capture_slot", "district"])
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import sys

    import validate as V

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only_missing = "--if-missing" in sys.argv

    target = datetime.fromisoformat(args[0]).date() if args else None

    slot = None
    for name in (EVENING, MORNING):
        if f"--{name}" in sys.argv:
            slot = name

    hours = []
    df, failures = snapshot(target_date=target, only_missing=only_missing,
                            slot=slot, hourly_out=hours)
    if df.empty and only_missing:
        print("nothing missing; night already captured")
        raise SystemExit(0)
    path = write_snapshot(df, append_only=only_missing)
    if path:
        print(f"wrote {path} ({len(df)} rows)")

    # Written after the feature rows, deliberately. If this fails, the night's
    # training row is already safe on disk; the hourly detail is the bonus, not
    # the thing that cannot be missed.
    hourly_written = write_hourly(pd.DataFrame(hours, columns=HOURLY_COLUMNS)
                                  if hours else None)
    if hourly_written:
        print(f"wrote {hourly_written} ({len(hours)} hourly rows)")

    if failures:
        print(f"{len(failures)} district(s) failed; the night was still saved")

    # Validated AFTER writing, never before. This is the job that cannot be
    # missed, so the rows go to disk first and the exit code is only about
    # whether anybody needs to be told. Refusing to write would throw away a
    # night that cannot be recollected in order to report that it was odd.
    raise SystemExit(V.gate(
        df, "forecast", required=["target_date", "leaid", "capture_slot",
                                  "captured_at", "snow_mm"],
        date_cols=("target_date",), known_ids=V.roster_ids()))
