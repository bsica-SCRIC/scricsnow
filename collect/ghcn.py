"""NOAA GHCN daily observations.

This is the backfillable half of the weather data. Unlike NWS forecasts, which
only exist going forward, observed daily weather is available for decades. That
is what makes it the natural partner for historical closure outcomes.

Station default is Greater Binghamton Airport. Verify with resolve_station()
before trusting a new one.
"""

import io
import time

import pandas as pd
import requests

NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"
NCEI_SEARCH = "https://www.ncei.noaa.gov/access/services/search/v1/data"

# Greater Binghamton Airport (BGM).
BGM = "USW00004725"

# PRCP/SNOW/SNWD are tenths of mm / mm / mm; TMAX/TMIN tenths of degC;
# AWND/WSF2 tenths of m/s. The API returns them already scaled to real units
# when units=metric, so we do not rescale here.
DATATYPES = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND", "WSF2"]

USER_AGENT = "scricsnow/0.1 (snow day predictor; bsica@btboces.org)"


def fetch_daily(station=BGM, start="1990-01-01", end=None, datatypes=None,
                timeout=120, retries=3):
    """Fetch daily summaries for one station as a tidy DataFrame.

    Returns columns: date, station, and one column per requested datatype.
    Metric units: SNOW/SNWD/PRCP in mm, TMAX/TMIN in degC, wind in m/s.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if datatypes is None:
        datatypes = DATATYPES

    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start,
        "endDate": end,
        "dataTypes": ",".join(datatypes),
        "format": "csv",
        "units": "metric",
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(NCEI_DATA, params=params,
                             headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"NCEI fetch failed after {retries} tries: {last_err}")

    if not r.text.strip():
        raise RuntimeError(f"NCEI returned no data for {station} {start}..{end}")

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.upper() for c in df.columns]
    df = df.rename(columns={"DATE": "date", "STATION": "station"})
    df["date"] = pd.to_datetime(df["date"])

    for col in datatypes:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["date", "station"] + list(datatypes)].sort_values("date")


def resolve_station(text="Binghamton", limit=10):
    """Look up candidate station IDs by name. Use to verify BGM, not at runtime."""
    r = requests.get(
        NCEI_SEARCH,
        params={"dataset": "daily-summaries", "text": text, "limit": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for item in r.json().get("results", []):
        out.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "start": (item.get("startDate") or "")[:10],
            "end": (item.get("endDate") or "")[:10],
        })
    return out


def winter_season(date):
    """Label the school year a date belongs to. 2025-12-15 and 2026-02-03 are
    both season 2025. July 1 is the boundary."""
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def add_derived(df):
    """Features that only need observed data, computed per station in date order."""
    df = df.sort_values("date").copy()
    df["season"] = df["date"].map(winter_season)
    df["dow"] = df["date"].dt.dayofweek
    # Snow that fell overnight into this morning is better proxied by today's
    # SNOW total plus yesterday evening's, but GHCN days end at midnight local,
    # so the previous day carries most of an overnight event.
    df["snow_prev"] = df["SNOW"].shift(1)
    df["snow_2day"] = df["SNOW"].fillna(0) + df["SNOW"].shift(1).fillna(0)
    df["depth_prev"] = df["SNWD"].shift(1)
    df["tmin_c"] = df["TMIN"]
    df["fresh_snow_on_ground"] = (df["SNWD"] > 25) & (df["snow_2day"] > 25)
    return df



def fetch_many(station_ids, start="2012-07-01", end=None, batch=20,
               datatypes=None, verbose=True):
    """Daily observations for many stations.

    NCEI accepts a comma-separated station list, so this batches rather than
    making one request per station. A batch that fails is reported and skipped
    instead of aborting the run: a training build over 50 stations should not
    be lost because one of them 404s.
    """
    ids = sorted(set(station_ids))
    frames, failed = [], []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        try:
            frames.append(fetch_daily(",".join(chunk), start=start, end=end,
                                      datatypes=datatypes))
        except RuntimeError as e:
            failed.append((chunk, str(e)[:100]))
        time.sleep(0.5)

    if verbose and failed:
        for chunk, why in failed:
            print(f"  station batch failed ({len(chunk)} stations): {why}")

    if not frames:
        raise RuntimeError("no station data fetched at all")
    return pd.concat(frames, ignore_index=True).sort_values(["station", "date"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "resolve":
        for s in resolve_station(sys.argv[2] if len(sys.argv) > 2 else "Binghamton"):
            print(s)
    else:
        d = fetch_daily(start="2015-07-01")
        d = add_derived(d)
        print(d.tail(10).to_string())
        print(f"\nrows={len(d)}  {d.date.min().date()} .. {d.date.max().date()}")
        snowy = d[d["SNOW"] > 50]
        print(f"days with >50mm (2in) snow: {len(snowy)}")
