"""The region as inline SVG. No map library, no tiles, no fixed zoom.

WHY SVG RATHER THAN A MAP WIDGET. deck.gl needs a view state -- latitude,
longitude, zoom -- and zoom is fixed in pixels per degree. A zoom that frames
this region on a monitor shows a third of it on a phone, and Streamlit cannot
reliably tell Python how wide the container is, so it cannot be computed per
device either. An SVG viewBox solves it by construction: the browser scales the
drawing to whatever space it is given, on any screen, including rotation.

The region projects to about 1.76:1, a comfortable landscape band at every size
we care about -- roughly 390x222 on a phone, 700x398 in a desktop card.

Two details that are easy to get wrong and ruin it on phones:

  * strokes need vector-effect="non-scaling-stroke", or district borders scale
    down with the drawing and vanish into hairlines on a small screen
  * NO TEXT inside the SVG. Text scales with the viewBox too, so anything
    legible on a monitor is unreadable at 390px. District names live in the
    caption and in <title> elements, which browsers show as native tooltips.

Render with st.markdown(unsafe_allow_html=True), NOT st.html. st.markdown does
no sanitizing at all, while st.html runs DOMPurify with USE_PROFILES html-only,
which silently strips every SVG element.
"""

import math
from html import escape

# Padding around the bounding box, as a fraction of its size, so districts on
# the edge are not clipped against the frame.
PAD = 0.02

# COORDINATE PRECISION, AND WHY IT IS NOT FIVE DECIMALS.
#
# The path coordinates are in SVG user units, and the whole region is about
# 1.83 units wide. At the ~700 px this renders at in a desktop card, one CSS
# pixel is 0.0026 units, so five decimal places resolved about four hundred
# times finer than anything anyone can see. It also resolved seventy times
# finer than the source data: shapes.py asks TIGER to generalize at 0.001
# degrees, which lands near 0.0007 units.
#
# That is not a rounding nicety. render() emits one string per rerun and
# Streamlit ships it over the websocket every time, so those digits are paid
# for by every viewer on every refresh, on the night when the most people are
# refreshing. Three decimals is still sub-pixel and cuts the payload by a
# quarter on its own.
COORD_DECIMALS = 3

# How far a simplified boundary may stray from the real one, in CSS pixels at
# REFERENCE_WIDTH_PX. One pixel is two device pixels on a phone, which for a
# region 110 miles across works out near 0.16 miles of boundary error -- far
# below anything that changes which district a reader thinks they are looking
# at, and still coarser than the TIGER generalization we started from.
#
# Measured on the 50-district geojson: 139 KB and 8,238 points unsimplified,
# 61 KB and 4,459 points at one pixel.
SIMPLIFY_PX = 1.0
REFERENCE_WIDTH_PX = 700.0

# No city dots. They were tried as orientation anchors and read as
# unexplained specks floating inside districts, because there is no text in
# this SVG to label them. An unlabelled dot is noise. If orientation turns out
# to be needed, it belongs in the caption, not the drawing.


def _bounds(polygons):
    xs = [x for poly in polygons for x, _ in poly]
    ys = [y for poly in polygons for _, y in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _projector(lon0, lat0, lon1, lat1):
    """Equirectangular with a cosine correction, which at 110 miles is visually
    identical to Mercator and far simpler to reason about."""
    k = math.cos(math.radians((lat0 + lat1) / 2))

    def project(lon, lat):
        # y is flipped because SVG counts downward from the top.
        return (lon - lon0) * k, (lat1 - lat)

    return project, (lon1 - lon0) * k, (lat1 - lat0)


def _path(points, project, decimals=COORD_DECIMALS):
    """One ring as an SVG path.

    Points that collapse onto each other once rounded are dropped. Without
    that, reducing the precision would keep every duplicate and give back a
    chunk of what the rounding just saved.
    """
    d, last = [], None
    for lon, lat in points:
        x, y = project(lon, lat)
        here = f"{x:.{decimals}f},{y:.{decimals}f}"
        if here == last:
            continue
        d.append(("M" if not d else "L") + here)
        last = here
    return "".join(d) + "Z"


def _perpendicular(point, start, end):
    (x, y), (x1, y1), (x2, y2) = point, start, end
    dx, dy = x2 - x1, y2 - y1
    span = math.hypot(dx, dy)
    if not span:
        return math.hypot(x - x1, y - y1)
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / span


def _douglas_peucker(points, tolerance):
    """Keep the points that carry the shape. Iterative, not recursive.

    A district boundary can run to several hundred vertices, and a recursive
    implementation on a pathological ring is a stack overflow inside a page
    render. An explicit stack costs nothing and cannot do that.
    """
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        worst, at = -1.0, lo
        for i in range(lo + 1, hi):
            gap = _perpendicular(points[i], points[lo], points[hi])
            if gap > worst:
                worst, at = gap, i
        if worst > tolerance:
            keep[at] = True
            stack.append((lo, at))
            stack.append((at, hi))
    return [p for p, k in zip(points, keep) if k]


def _simplify_ring(ring, tolerance):
    """Douglas-Peucker on a CLOSED ring.

    Closed rings need a wrinkle. Douglas-Peucker anchors on the first and last
    point, and on a ring those are the same point, so the segment between them
    has zero length and every vertex measures as zero distance from it: run
    naively, a whole district collapses to a single point. Splitting the ring
    at its farthest vertex first gives two open polylines with real endpoints.
    """
    closed = len(ring) > 1 and ring[0] == ring[-1]
    pts = list(ring[:-1]) if closed else list(ring)
    if len(pts) < 4:
        return list(ring)

    x0, y0 = pts[0]
    far = max(range(len(pts)),
              key=lambda i: (pts[i][0] - x0) ** 2 + (pts[i][1] - y0) ** 2)
    head = _douglas_peucker(pts[:far + 1], tolerance)
    tail = _douglas_peucker(pts[far:] + [pts[0]], tolerance)
    out = head[:-1] + tail

    # A ring that simplified below a drawable triangle is not a simplification,
    # it is a deletion. Give the original back rather than a sliver.
    if len(out) < 4:
        return list(ring)
    return out + [out[0]] if closed else out


def simplify(rings, px=SIMPLIFY_PX, reference_px=REFERENCE_WIDTH_PX):
    """Thin every ring to what survives at display size.

    Tolerance is derived from the bounding box of the whole set, so it means
    the same thing for a large district and a small one: both are allowed to
    stray by the same number of pixels on the finished map.

    Called once behind the app's cached shape loader, never per rerun.
    """
    rings = [list(r) for r in rings]
    drawable = [r for r in rings if len(r) >= 2]
    if not drawable:
        return rings

    lon0, lat0, lon1, lat1 = _bounds(drawable)
    k = math.cos(math.radians((lat0 + lat1) / 2))
    width = (lon1 - lon0) * k
    if width <= 0:
        return rings
    tolerance = (width / reference_px) * px

    out = []
    for ring in rings:
        if len(ring) < 4:
            out.append(ring)
            continue
        # Simplify in projected space, so the tolerance is a distance on the
        # drawing rather than in degrees, which are not the same length in x
        # and y at this latitude.
        projected = [((lon - lon0) * k, lat1 - lat) for lon, lat in ring]
        thinned = _simplify_ring(projected, tolerance)
        out.append([(x / k + lon0, lat1 - y) for x, y in thinned])
    return out


def render(polygons, fills, selected=None, line="#28405899",
           selected_line="#081834", background="none", tooltips=None,
           aria="Map of the region"):
    """Build the SVG.

    polygons  list of (leaid, name, ring) where ring is [(lon, lat), ...]
    fills     leaid -> CSS colour
    selected  leaid to outline on top, or None
    tooltips  leaid -> text for the native hover tooltip
    """
    rings = [ring for _, _, ring in polygons]
    if not rings:
        return ""

    lon0, lat0, lon1, lat1 = _bounds(rings)
    project, width, height = _projector(lon0, lat0, lon1, lat1)
    padx, pady = width * PAD, height * PAD

    parts = []
    picked = []
    for leaid, name, ring in polygons:
        d = _path(ring, project)
        fill = fills.get(leaid, "#cccccc")
        tip = escape(str((tooltips or {}).get(leaid, name or leaid)))
        parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{line}" stroke-width="1" '
            f'vector-effect="non-scaling-stroke"><title>{tip}</title></path>')
        if selected and leaid == selected:
            picked.append(d)

    # Drawn last so a neighbouring district can never paint over the outline
    # of the one the reader actually cares about.
    for d in picked:
        parts.append(
            f'<path d="{d}" fill="none" stroke="{selected_line}" '
            f'stroke-width="2.5" vector-effect="non-scaling-stroke" '
            f'stroke-linejoin="round" pointer-events="none"/>')

    # Four numbers, so precision here costs nothing and a tight viewBox is
    # worth more than a short one.
    view = (f"{-padx:.5f} {-pady:.5f} "
            f"{width + 2 * padx:.5f} {height + 2 * pady:.5f}")
    return (
        f'<svg viewBox="{view}" role="img" aria-label="{escape(aria)}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%;height:auto;display:block;background:{background};'
        f'border-radius:14px">'
        + "".join(parts) + "</svg>")


def polygons_from(frame):
    """(leaid, name, ring) tuples from the shapes frame."""
    return [(r["leaid"], r.get("name"), r["polygon"])
            for _, r in frame.iterrows()]
