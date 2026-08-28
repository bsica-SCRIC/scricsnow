"""Assign each district the weather stations that actually represent it.

The districts span roughly 110 miles and 858 to 2,965 ft of elevation, from the
Susquehanna valley to the Catskills. One airport station cannot speak for all of
them: Binghamton airport says nothing useful about Hunter-Tannersville, 90 miles
away and 1,260 ft higher.

So each district gets its own nearest snowfall-reporting stations, blended by
inverse distance, plus the elevation gap between the district and the stations
reporting for it. That gap is the correction term. A valley station under-reports
snow for a hilltop district in a predictable direction, and a model given the gap
can learn that; a model given only the raw snowfall cannot.

This is also what lets the model serve districts with no closure history of their
own. Predictions come from weather and terrain, not from a district's own past,
so a district joining today gets a real forecast on day one.

Inputs:
    reference/roster.csv                leaid, district, lat, lon, elev_m
    reference/ghcn_snow_stations.csv    id, name, lat, lon, elev_m
"""

import os

import numpy as np
import pandas as pd

REF = os.path.join(os.path.dirname(__file__), "..", "reference")
STATIONS_CSV = os.path.join(REF, "ghcn_snow_stations.csv")
OUT_CSV = os.path.join(REF, "district_stations.csv")

# How many stations to blend per district. One station is a single point of
# failure: co-op observers miss days, and a missing day reads as zero snow,
# which is a dangerous error in this direction. Three lets the blend survive one
# silent observer.
N_STATIONS = 3

# Beyond this, a station is describing different weather. Districts that cannot
# find N within it simply get fewer, which is reported rather than hidden.
MAX_MILES = 25.0


def miles(lat1, lon1, lat2, lon2):
    """Flat-earth miles. Fine at this latitude over these distances."""
    return np.sqrt(((lat1 - lat2) * 69.0) ** 2 + ((lon1 - lon2) * 51.5) ** 2)


def assign(districts, stations=None, n=N_STATIONS, max_miles=MAX_MILES):
    """One row per district-station pair, with distance, elevation gap, weight."""
    if stations is None:
        stations = pd.read_csv(STATIONS_CSV)

    rows = []
    for _, d in districts.iterrows():
        dist = miles(d["lat"], d["lon"], stations["lat"], stations["lon"])
        near = stations.assign(miles=dist)
        near = near[near["miles"] <= max_miles].nsmallest(n, "miles")

        for rank, (_, s) in enumerate(near.iterrows(), start=1):
            rows.append({
                "leaid": d["leaid"],
                "district": d["district"],
                "rank": rank,
                "station_id": s["id"],
                "station_name": s["name"],
                "miles": round(float(s["miles"]), 1),
                # Positive means the district sits ABOVE its station, so the
                # station under-reports snow for it.
                "elev_gap_m": (round(float(d["elev_m"] - s["elev_m"]), 1)
                               if pd.notna(d.get("elev_m")) else None),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Inverse-distance weights, normalized within each district. The +1 keeps a
    # station sitting almost on top of a district centroid from taking all the
    # weight and reintroducing the single-station problem.
    out["w"] = 1.0 / (out["miles"] + 1.0)
    out["weight"] = out["w"] / out.groupby("leaid")["w"].transform("sum")
    return out.drop(columns="w").sort_values(["district", "rank"])


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    districts = pd.read_csv(os.path.join(root, "reference", "roster.csv"),
                            dtype={"leaid": str})
    pairs = assign(districts)
    pairs.to_csv(OUT_CSV, index=False)

    per = pairs.groupby("leaid").size()
    thin = per[per < N_STATIONS]
    nearest = pairs[pairs["rank"] == 1]

    print(f"districts: {len(districts)}   pairs: {len(pairs)}")
    print(f"nearest station: median {nearest.miles.median():.1f} mi, "
          f"worst {nearest.miles.max():.1f} mi")
    print(f"elevation gap to nearest: min {nearest.elev_gap_m.min():.0f} m, "
          f"max {nearest.elev_gap_m.max():.0f} m")
    print(f"districts with fewer than {N_STATIONS} stations: {len(thin)}")
    if len(thin):
        names = districts.set_index("leaid")["district"]
        for leaid, cnt in thin.items():
            print(f"    {names.get(leaid, leaid)}: {cnt}")
    print(f"\nwrote {OUT_CSV}")
