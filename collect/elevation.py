"""District elevation, from the USGS 3DEP point service.

Elevation matters here because the districts span 858 ft at Binghamton to
nearly 3,000 ft in the Catskills. A valley weather station badly misrepresents
a hilltop district, so the model needs both the district's elevation and the
gap between it and whichever station is reporting its weather.

Results are cached to reference/district_elevation.csv and only missing LEAIDs
are fetched, so this costs one request per district ever rather than per run.
"""

import os
import time

import pandas as pd
import requests

EPQS = "https://epqs.nationalmap.gov/v1/json"
CACHE = os.path.join(os.path.dirname(__file__), "..", "reference",
                     "district_elevation.csv")
USER_AGENT = "scricsnow/0.1 (snow day predictor; bsica@btboces.org)"


def fetch_one(lat, lon, timeout=60, retries=3):
    """Elevation in metres at a point, or None if the service will not say."""
    for attempt in range(retries):
        try:
            r = requests.get(EPQS, params={"x": lon, "y": lat, "units": "Meters",
                                           "wkid": 4326, "includeDate": "false"},
                             headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            value = r.json().get("value")
            if value is None:
                return None
            elev = float(value)
            # The service returns a large negative sentinel off-raster.
            return None if elev < -1000 else elev
        except (requests.RequestException, ValueError, TypeError):
            time.sleep(2 ** attempt)
    return None


def load_cache():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, dtype={"leaid": str})
    return pd.DataFrame(columns=["leaid", "elev_m"])


def ensure(districts, sleep=0.2):
    """Return districts with elev_m filled, fetching only what is missing.

    `districts` needs columns leaid, lat, lon.
    """
    cache = load_cache()
    known = set(cache["leaid"])

    missing = districts[~districts["leaid"].isin(known)]
    if len(missing):
        rows = []
        for _, r in missing.iterrows():
            rows.append({"leaid": r["leaid"],
                         "elev_m": fetch_one(r["lat"], r["lon"])})
            time.sleep(sleep)
        new = pd.DataFrame(rows)
        cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
        cache.to_csv(CACHE, index=False)

    # Idempotent: re-running on a frame that already carries elevation must not
    # produce elev_m_x / elev_m_y.
    districts = districts.drop(columns=["elev_m"], errors="ignore")
    return districts.merge(cache, on="leaid", how="left")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "reference", "roster.csv")
    d = pd.read_csv(path, dtype={"leaid": str})
    out = ensure(d)
    missing = out["elev_m"].isna().sum()
    print(f"districts: {len(out)}   elevation missing: {missing}")
    print(f"elevation range: {out.elev_m.min():.0f} - {out.elev_m.max():.0f} m "
          f"({out.elev_m.min()*3.28:.0f} - {out.elev_m.max()*3.28:.0f} ft)")
