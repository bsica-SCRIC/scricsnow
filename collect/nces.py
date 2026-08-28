"""The New York school district reference table.

Everything here comes from Census TIGERweb and the NCES Common Core of Data, so
the repository can rebuild its own reference data from scratch without leaning
on anything it does not ship.

Three identifiers appear, and they are not interchangeable:

  NCES LEAID   7 digits, e.g. 3604870 for Binghamton City SD. This is also the
               Census TIGER GEOID for the district polygon, so it is the only
               one that joins directly to map geometry.
  NY BEDS      12 digits, e.g. 030200010000. New York's own code, published by
               NYSED and used throughout state reporting. It does NOT appear in
               TIGER, so it needs this crosswalk to reach geometry.
  Name         Ambiguous, and never used for joining. 'Unatego' is the common
               name for what TIGER calls 'Otego-Unadilla Central School
               District', and no string algorithm bridges that.
"""

import os

import pandas as pd
import requests

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference")
GEO_CSV = os.path.join(REF_DIR, "ny_school_districts.csv")
BEDS_CSV = os.path.join(REF_DIR, "ny_leaid_beds_crosswalk.csv")
NY_DISTRICTS_CSV = os.path.join(REF_DIR, "ny_districts.csv")

TIGERWEB = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
            "School/MapServer/5/query")
CCD = "https://educationdata.urban.org/api/v1/school-districts/ccd/directory/2022/"
NY_FIPS = "36"
USER_AGENT = "scricsnow/0.1 (snow day predictor; bsica@btboces.org)"


def fetch_tiger(timeout=180):
    """Unified school district polygons for NY: GEOID, name, centroid."""
    rows, offset = [], 0
    while True:
        r = requests.get(TIGERWEB, params={
            "where": f"STATE='{NY_FIPS}'",
            "outFields": "GEOID,NAME,LOGRADE,HIGRADE,CENTLAT,CENTLON,AREALAND",
            "returnGeometry": "false", "f": "json",
            "resultOffset": offset, "resultRecordCount": 1000,
        }, headers={"User-Agent": USER_AGENT}, timeout=timeout).json()
        feats = r.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < 1000:
            break
        offset += 1000

    df = pd.DataFrame(rows)
    df["CENTLAT"] = df["CENTLAT"].astype(float)
    df["CENTLON"] = df["CENTLON"].astype(float)
    return df.sort_values("NAME")


def fetch_ccd(timeout=180):
    """NCES directory for NY, carrying the state-assigned (BEDS) id."""
    rows, url = [], f"{CCD}?fips={NY_FIPS}"
    while url:
        j = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=timeout).json()
        rows += j.get("results", [])
        url = j.get("next")
    return pd.DataFrame(rows)[["leaid", "state_leaid", "lea_name",
                               "county_name", "agency_type"]]


def refresh():
    """Re-download both sources and rewrite the cached reference CSVs."""
    os.makedirs(REF_DIR, exist_ok=True)
    fetch_tiger().to_csv(GEO_CSV, index=False)
    fetch_ccd().to_csv(BEDS_CSV, index=False)
    return build()


def build():
    """Merge cached TIGER geography with the NCES/BEDS crosswalk."""
    geo = pd.read_csv(GEO_CSV, dtype={"GEOID": str})
    beds = pd.read_csv(BEDS_CSV, dtype={"leaid": str, "state_leaid": str})

    beds = beds.rename(columns={"leaid": "GEOID"})
    # Urban Institute prefixes the state, e.g. NY-030200010000.
    beds["beds"] = beds["state_leaid"].str.replace(r"^NY-", "", regex=True)

    df = geo.merge(beds[["GEOID", "beds", "county_name", "agency_type"]],
                   on="GEOID", how="left")
    df = df.rename(columns={"GEOID": "leaid", "NAME": "name",
                            "CENTLAT": "lat", "CENTLON": "lon",
                            "county_name": "county"})
    return df[["leaid", "beds", "name", "county", "lat", "lon",
               "LOGRADE", "HIGRADE", "AREALAND"]]


def normalize_beds(value):
    """Coerce a BEDS code back to 12-digit text.

    BEDS codes have a leading zero (Binghamton is 030200010000). Excel, most
    CSV exporters, and pandas all happily read that as the integer 30200010000
    and drop it, which silently breaks any join. Re-padding is cheap and repairs
    the damage after the fact, so do it on every read.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.endswith(".0"):          # float round-trip, e.g. 30200010000.0
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(12)


def to_district_beds(code):
    """Reduce a BEDS code to its district-level form.

    A BEDS code is an 8-digit district prefix plus a 4-digit building suffix,
    where 0000 means the district itself. Small single-building districts store
    a BUILDING code like ...0001, which will not match the district-level
    ...0000 in the NCES reference. Keying both sides at district level fixes
    those without weakening anything: the reference holds only district-level
    codes, so this can never collapse two real districts.
    """
    if code is None:
        return None
    return code[:8] + "0000"


if __name__ == "__main__":
    import sys

    df = refresh() if "--refresh" in sys.argv else build()
    df.to_csv(NY_DISTRICTS_CSV, index=False)
    print(f"NY districts: {len(df)}   without BEDS: {df.beds.isna().sum()}")
    print(f"wrote {NY_DISTRICTS_CSV}")
