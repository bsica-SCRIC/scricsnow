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


def _path(points, project):
    d = []
    for i, (lon, lat) in enumerate(points):
        x, y = project(lon, lat)
        d.append(f"{'M' if i == 0 else 'L'}{x:.5f},{y:.5f}")
    return "".join(d) + "Z"


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
