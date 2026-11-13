"""Tests for the map drawing, mostly about not silently destroying geometry.

Simplification is the kind of change that looks fine in a benchmark and ruins
the picture: a ring that collapses to a triangle still renders, still has a
tooltip, and is simply the wrong shape. Nobody would get an exception. So the
properties that matter are pinned here rather than eyeballed once.
"""

import math

import pandas as pd
import pytest

import regionmap as RM
import shapes as SHP


def circle(n=64, r=0.5, cx=0.0, cy=42.0):
    """A closed ring with no redundant points, so simplification has to work
    rather than just dropping duplicates."""
    pts = [(cx + r * math.cos(2 * math.pi * i / n),
            cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return pts + [pts[0]]


def square(size=1.0, cx=0.0, cy=42.0):
    half = size / 2
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half), (cx - half, cy - half)]


def area(ring):
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


# --------------------------------------------------------------------------
# Simplification
# --------------------------------------------------------------------------

def test_a_closed_ring_stays_closed():
    """The failure that made the first attempt useless. Douglas-Peucker anchors
    on first and last, which on a ring are the same point, so the whole
    district measures as zero distance from a zero-length segment and
    collapses. Splitting at the farthest vertex is what stops that."""
    out = RM.simplify([circle()])[0]
    assert out[0] == out[-1]
    assert len(out) >= 4


def test_a_district_never_collapses_to_a_sliver():
    for ring in (circle(), circle(n=8), square()):
        out = RM.simplify([ring])[0]
        assert len(out) >= 4, "a ring simplified into something undrawable"


def test_the_shape_is_preserved_not_just_the_point_count():
    """Area is the cheap proxy for 'still looks like the same district'."""
    ring = circle(n=256)
    out = RM.simplify([ring])[0]
    assert area(out) == pytest.approx(area(ring), rel=0.02)


def test_simplified_points_are_a_subset_of_the_originals():
    """Douglas-Peucker selects vertices, it does not invent them. A new point
    would mean the projection round-trip has drifted."""
    ring = circle(n=128)
    out = RM.simplify([ring])[0]
    originals = {(round(x, 9), round(y, 9)) for x, y in ring}
    for x, y in out:
        assert (round(x, 9), round(y, 9)) in originals


def test_a_straight_edge_loses_its_redundant_points():
    """The whole point. A square drawn with a hundred points per side is still
    a square, and should ship as four corners."""
    dense = []
    for (x1, y1), (x2, y2) in zip(square(), square()[1:]):
        dense += [(x1 + (x2 - x1) * i / 25, y1 + (y2 - y1) * i / 25)
                  for i in range(25)]
    dense.append(dense[0])
    out = RM.simplify([dense])[0]
    assert len(out) < 12
    assert area(out) == pytest.approx(area(square()), rel=0.01)


def test_tiny_rings_are_passed_through_untouched():
    tiny = [(0.0, 42.0), (0.1, 42.0), (0.0, 42.1)]
    assert RM.simplify([tiny])[0] == tiny


def test_simplify_on_nothing_does_not_raise():
    assert RM.simplify([]) == []
    assert RM.simplify([[]]) == [[]]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_render_is_byte_stable_for_the_same_inputs():
    """Streamlit can only skip resending an element whose payload is
    unchanged, and the render is by far the largest thing on the page. A
    nondeterministic render would quietly cost every viewer 61 KB per refresh.
    """
    polys = [("1", "A", circle()), ("2", "B", square(cx=2.0))]
    fills = {"1": "#111111", "2": "#222222"}
    first = RM.render(polys, fills, selected="1")
    second = RM.render(polys, fills, selected="1")
    assert first == second


def test_coordinates_are_not_written_at_five_decimals():
    """Precision here is paid for by every viewer on every refresh."""
    svg = RM.render([("1", "A", square())], {"1": "#111111"})
    for token in svg.split('d="')[1].split('"')[0].replace("M", " ").replace(
            "L", " ").replace("Z", " ").split():
        for part in token.split(","):
            assert len(part.split(".")[-1]) <= RM.COORD_DECIMALS


def test_the_real_region_shrinks_and_stays_whole():
    """Against the shipped geojson, not a fixture. The saving is the reason
    the code exists and the district count is what must not change."""
    frame = SHP.to_polygons(SHP.load())
    fills = {leaid: "#123456" for leaid in frame["leaid"]}

    before = RM.render(RM.polygons_from(frame), fills)
    thinned = frame.copy()
    thinned["polygon"] = RM.simplify(list(frame["polygon"]))
    after = RM.render(RM.polygons_from(thinned), fills)

    assert len(after) < len(before) * 0.75
    assert thinned["leaid"].nunique() == frame["leaid"].nunique()
    assert all(len(r) >= 4 for r in thinned["polygon"])


def test_every_district_keeps_a_tooltip_and_a_fill():
    frame = SHP.to_polygons(SHP.load())
    fills = {leaid: "#123456" for leaid in frame["leaid"]}
    tips = {leaid: f"{name} - 40%" for leaid, name
            in zip(frame["leaid"], frame["name"])}
    svg = RM.render(RM.polygons_from(frame), fills, tooltips=tips)
    assert svg.count("<title>") == len(frame)
    assert "#cccccc" not in svg, "a district rendered with the fallback fill"


def test_an_empty_region_renders_nothing_rather_than_raising():
    assert RM.render([], {}) == ""
    assert RM.render(pd.DataFrame().to_dict("records"), {}) == ""
