"""District boundaries for the map.

Pulls polygons for the tracked districts from Census TIGERweb as GeoJSON and
caches them in reference/district_shapes.geojson.

NO GEO LIBRARY REQUIRED, which is the point. TIGER polygons are far too
detailed to ship to a phone: Binghamton alone is 558 vertices and 11.5 KB. But
the ArcGIS REST service will generalize server-side via maxAllowableOffset, so
we ask for simplified geometry rather than simplifying it here. At 0.001
degrees, roughly 110 m, Binghamton drops to 78 vertices and 920 bytes, and 50
districts fit in well under 100 KB. That keeps geopandas and shapely off the
dependency list entirely, and Streamlit already bundles pydeck to draw it.

Refetch only when the roster changes. The boundaries themselves move about once
a decade.
"""

import json
import os

import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
ROSTER_CSV = os.path.join(ROOT, "reference", "roster.csv")
SHAPES_JSON = os.path.join(ROOT, "reference", "district_shapes.geojson")

URL = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
       "School/MapServer/5/query")
HEADERS = {"User-Agent": "scricsnow/0.1 (snow day predictor; bsica@btboces.org)"}

# Degrees of tolerance for server-side generalization. 0.001 is about 110 m,
# which is invisible at the zoom a region map uses and cuts the payload by more
# than 90 percent.
OFFSET_DEGREES = 0.001

# ArcGIS caps how long a where clause can be, so ask in batches.
BATCH = 20

# Drop polygon parts smaller than this share of their district's main body.
#
# 15 of the 50 districts come back as multipolygons, and every one of the 22
# extra parts is under 1% of the district it belongs to -- the largest is
# 0.894%. These are not exclaves, they are slivers, almost certainly split off
# where server-side generalization pinched a narrow neck in two. At map scale
# they render as unexplained specks floating beside a district, which reads as
# a bug rather than as geography. Filtering at 5% removes all 22 and leaves
# exactly one polygon per district.
MIN_PART_SHARE = 0.05


def fetch(leaids, offset=OFFSET_DEGREES, timeout=180):
    """GeoJSON FeatureCollection for the given LEAIDs, already simplified."""
    features, missing = [], set(leaids)
    ids = sorted(set(leaids))

    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        quoted = ",".join(f"'{x}'" for x in chunk)
        r = requests.get(URL, params={
            "where": f"GEOID IN ({quoted})",
            "outFields": "GEOID,NAME",
            "returnGeometry": "true",
            "maxAllowableOffset": offset,
            "outSR": "4326",
            "f": "geojson",
        }, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        got = r.json().get("features", [])
        features.extend(got)
        missing -= {f["properties"]["GEOID"] for f in got}

    return {"type": "FeatureCollection", "features": features}, sorted(missing)


def load(path=SHAPES_JSON):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ring_area(ring):
    """Shoelace area in square degrees. Only ever used for comparing parts of
    the same district to each other, so the units do not matter."""
    total = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def to_polygons(geojson, min_part_share=MIN_PART_SHARE):
    """Flatten to one row per drawable ring, with slivers removed.

    MultiPolygons are split into separate rows rather than nested. Parts far
    smaller than their district's main body are dropped; see MIN_PART_SHARE.
    """
    rows = []
    for feat in (geojson or {}).get("features", []):
        leaid = feat["properties"]["GEOID"]
        name = feat["properties"].get("NAME")
        geom = feat["geometry"]
        parts = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])

        rings = [part[0] for part in parts]
        areas = [ring_area(r) for r in rings]
        biggest = max(areas) if areas else 0.0
        for ring, area in zip(rings, areas):
            if biggest and area / biggest < min_part_share:
                continue
            rows.append({"leaid": leaid, "name": name, "polygon": ring})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    roster = pd.read_csv(ROSTER_CSV, dtype={"leaid": str})
    geo, missing = fetch(roster["leaid"].tolist())

    with open(SHAPES_JSON, "w", encoding="utf-8") as fh:
        json.dump(geo, fh)

    size = os.path.getsize(SHAPES_JSON)
    polys = to_polygons(geo)
    print(f"districts requested: {len(roster)}   returned: "
          f"{len(geo['features'])}   polygons: {len(polys)}")
    print(f"file size: {size / 1024:.0f} KB")
    if missing:
        print(f"NO BOUNDARY RETURNED for {len(missing)}: {missing}")
        print("Those districts will be absent from the map.")
    print(f"wrote {SHAPES_JSON}")
