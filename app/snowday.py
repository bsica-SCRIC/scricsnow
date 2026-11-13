"""Snow Day Predictor - the public page.

Design notes, since several choices here are deliberate rather than decorative:

  * TWO NUMBERS, NOT ONE. The 8pm Call is frozen and is what the accuracy
    record scores. The live number moves as NWS updates. Showing only the
    frozen one is useless to a kid watching it snow harder at 11pm; showing
    only the moving one destroys the record, because "which number were you
    scoring?" stops having an answer. A market has a price and a settlement,
    and both are honest as long as they are labelled.

  * The number is read from the hourly capture rather than fetched per viewer.
    Everyone then sees the same figure, which matters for something kids
    compare at the bus stop. A live fetch is the fallback when the stored
    reading has gone stale.

  * If the district has already announced, the announcement replaces the
    prediction. A percentage next to a district that has said "closed" is worse
    than useless, so truth always outranks the model.

  * The accuracy record is on the page, not buried. A tool that never publishes
    its receipts is asking to be believed on trust.
"""

import functools
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "collect"))

import bank as B           # noqa: E402
import datasync            # noqa: E402
import heuristic as H      # noqa: E402
import news as NEWS        # noqa: E402
import nws                 # noqa: E402
import predict as P        # noqa: E402
import regionmap as RM     # noqa: E402
import scorecard as SC     # noqa: E402
import schoolday as SD     # noqa: E402
import shapes as SHP       # noqa: E402
import track as TRACK      # noqa: E402
import verdicts as V       # noqa: E402

st.set_page_config(page_title="Snow Day Predictor", page_icon="❄️",
                   layout="centered")

# A stored reading older than this is not worth showing; fetch fresh instead.
STALE_AFTER = timedelta(hours=2)

PALETTE = {
    "light": {"bg": "#eaf4fc", "card": "#ffffff", "ink": "#0d2440",
              "muted": "#4a6785", "accent": "#1c72c4", "edge": "#c5ddf2",
              "flake": "#ffffff", "shadow": "rgba(13,36,64,.10)",
              "up": "#c2410c", "down": "#0e7490", "map": "light"},
    "dark": {"bg": "#0b1727", "card": "#132539", "ink": "#e8f3ff",
             "muted": "#93b4d4", "accent": "#5cb3ff", "edge": "#22415f",
             "flake": "#dceeff", "shadow": "rgba(0,0,0,.35)",
             "up": "#fb923c", "down": "#67e8f9", "map": "dark"},
}

# Map colours, per theme, because a single ramp cannot work on both basemaps.
#
# The rule in both: MORE PROBABILITY MEANS MORE VISUALLY PROMINENT. On a light
# basemap that means going darker and more saturated; on a dark one it means
# going brighter. The first version used one ramp for both, running pale to
# deep navy, which was invisible at the low end in light mode and would have
# gone invisible at the HIGH end in dark mode once a real storm arrived.
#
# The low end is deliberately not near-white: a calm district still has to read
# as a district, or the map loses its shape entirely on a quiet night.
MAP_COLOURS = {
    "light": {
        "low": (188, 214, 236),      # pale steel blue, still visible on white
        "high": (16, 60, 140),       # deep navy
        "alpha": 205,
        "line": "#28405899",         # dark slate, NOT white
        "selected": "#081834",
        "word": "Deeper blue",
    },
    "dark": {
        "low": (38, 64, 94),         # dim slate, present but quiet
        "high": (104, 204, 255),     # bright sky blue, pops off the dark map
        "alpha": 210,
        "line": "#c8e2fa55",
        "selected": "#ffffff",
        "word": "Brighter blue",
    },
}

# What a district actually did this morning, on the map.
#
# WCAG 2.1, the two criteria a coloured map can fail:
#   1.4.11 non-text contrast: every fill here is at least 3:1 against the page
#          background and against a quiet district, in its own theme. One
#          colour cannot do that on both a white and a navy page, hence two
#          sets. (The old single gold was 1.5:1 on the light theme.)
#   1.4.1  use of colour: colour is never the only carrier. The legend names
#          each colour, the tooltip names the status, and the same districts
#          are listed in words under the map.
STATUS_FILLS = {
    "light": {"closed": "#92400e", "delay": "#be185d", "early": "#5b21b6"},
    "dark": {"closed": "#facc15", "delay": "#fb7185", "early": "#d8b4fe"},
}
STATUS_LABELS = {"closed": "Closed", "delay": "Delayed",
                 "early": "Early dismissal"}


def status_bucket(status):
    """Which of the three map colours a recorded status gets.

    Remote and emergency days are days without school in the building, so
    they sit with closed. Anything unrecognized does too: a district that
    announced something is better shown as announced than as a percentage.
    """
    return status if status in ("delay", "early") else "closed"


def short_name(district):
    for tail in (" Central School District at Delhi", " and Central School District",
                 " Central School District", " City School District"):
        if district.endswith(tail):
            return district[: -len(tail)]
    return district


def style(theme, snowing):
    c = PALETTE[theme]
    flakes = ""
    if snowing:
        # Hidden unless snow is actually forecast, so the page does not cry
        # wolf on a clear night.
        spans = "".join(
            f'<span style="left:{(i * 8.3) % 100:.1f}%;'
            f'animation-delay:{(i * 1.7) % 12:.1f}s;'
            f'animation-duration:{9 + (i % 5) * 2}s;'
            f'font-size:{12 + (i % 4) * 6}px">❄</span>'
            for i in range(14))
        flakes = f'<div class="snow" aria-hidden="true">{spans}</div>'

    st.markdown(f"""
    <style>
      .stApp {{ background: {c['bg']}; color: {c['ink']}; }}
      .block-container {{ padding-top: 2rem; max-width: 48rem; }}
      h1, h2, h3, p, li, label, span, td, th {{ color: {c['ink']}; }}
      .card {{ background: {c['card']}; border: 1px solid {c['edge']};
        border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        box-shadow: 0 6px 18px {c['shadow']}; position: relative; z-index: 1; }}
      .big {{ font-size: clamp(3.4rem, 15vw, 5.5rem); font-weight: 800; line-height: 1;
              color: {c['accent']}; letter-spacing: -3px; }}
      .verdict {{ font-size: clamp(1.15rem, 4.6vw, 1.55rem); font-weight: 700; margin-top: .2rem; }}
      .muted {{ color: {c['muted']}; font-size: .93rem; }}
      .pill {{ display:inline-block; padding:.25rem .7rem; border-radius:999px;
               background:{c['accent']}22; color:{c['accent']};
               font-size:.8rem; font-weight:600; margin:0 .35rem .35rem 0; }}
      .up {{ color:{c['up']}; font-weight:700; }}
      .down {{ color:{c['down']}; font-weight:700; }}
      .meter {{ height: 16px; border-radius: 999px; background: {c['edge']};
                overflow: hidden; margin: .6rem 0 .2rem; }}
      .meter > div {{ height: 100%; border-radius: 999px;
                      background: linear-gradient(90deg,#7cc4ff,{c['accent']}); }}
      .snow {{ position: fixed; inset: 0; pointer-events: none; z-index: 0; }}
      .snow span {{ position: absolute; top: -6%; color: {c['flake']};
                    opacity: .7; animation-name: fall;
                    animation-timing-function: linear;
                    animation-iteration-count: infinite; }}
      @keyframes fall {{ 0% {{ transform: translateY(-5vh) rotate(0); }}
                         100% {{ transform: translateY(105vh) rotate(360deg); }} }}
      @media (prefers-reduced-motion: reduce) {{ .snow {{ display: none; }} }}
    </style>{flakes}
    """, unsafe_allow_html=True)


def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def script_frame(html):
    """A zero-height frame whose script can reach the real page.

    st.markdown and st.html both drop <script>, by design. A component frame is
    the only place Streamlit will run one, and it is same-origin, so it can see
    the parent's storage and address bar.

    st.iframe is the current name; st.components.v1.html is the same thing under
    the old one and is on its way out. Preferring the new name keeps the console
    clean without raising the streamlit floor in requirements.txt.
    """
    # One pixel, not zero: st.iframe rejects a height of 0 outright, and the
    # frame only has to exist, not be seen.
    if hasattr(st, "iframe"):
        st.iframe(html, height=1)
    else:
        components.html(html, height=1)


# --------------------------------------------------------------------------
# Data
#
# NO LOADER MAY RAISE. Streamlit answers an uncaught exception with a red
# traceback where the forecast should be, and every loader below reads a file
# that a bot wrote unattended at three in the morning. A malformed CSV is then
# not a missing card, it is a broken public page on the night the most people
# are looking at it.
#
# So each one degrades to something the page can render: an empty frame, an
# empty scorecard, no map. The page is already written to handle those, because
# they are also what August looks like.
#
# resilient() sits INSIDE the cache deliberately, so a failure is cached for
# the same TTL as a success. Retrying a broken file on every rerun of every
# session would turn one bad commit into a load problem as well as a display
# one.
# --------------------------------------------------------------------------

def resilient(fallback, what):
    """Wrap a loader so a bad file costs one card, not the page."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:                    # noqa: BLE001
                # Printed, not shown. The reader cannot act on it and a
                # stack trace next to a snow day forecast helps nobody; the
                # Actions log and the container log are where this belongs.
                print(f"[snowday] {what} unavailable: "
                      f"{type(exc).__name__}: {exc}")
                return fallback() if callable(fallback) else fallback
        return inner
    return wrap


def _empty(*columns):
    return lambda: pd.DataFrame(columns=list(columns))


@st.cache_data(ttl=3600)
@resilient(_empty("leaid", "district", "lat", "lon", "elev_m"), "the roster")
def load_roster():
    return pd.read_csv(os.path.join(ROOT, "reference", "roster.csv"),
                       dtype={"leaid": str}).sort_values("district")


@st.cache_data(ttl=3600)
@resilient(_empty("leaid", "district", "days_built_in"), "the snow-day bank")
def load_bank():
    return B.load()


@st.cache_data(ttl=300)
@resilient(_empty("leaid", "date", "status"), "recorded closings")
def load_outcomes(season):
    return B.load_outcomes(season)


# {"bands": []} rather than {}: an empty dict is falsy, so verdicts.load()
# would simply be called again on the very file that just failed. The
# built-in FALLBACK_BANDS take over from here.
@st.cache_data(ttl=3600)
@resilient(lambda: {"bands": []}, "the verdict wording")
def load_verdicts():
    return V.load()


# An empty calendar means no day is known to be a holiday, so the page keeps
# forecasting. The opposite default would blank the site over Christmas week
# and look identical to working correctly.
@st.cache_data(ttl=3600)
@resilient(_empty("date", "reason"), "the school calendar")
def load_calendar():
    return SD.load()


@st.cache_data(ttl=3600)
@resilient(_empty("leaid", "name", "polygon"), "the district map")
def load_shapes():
    """District rings, thinned to what survives at display size.

    Simplified HERE, inside the cache, rather than in render(): it costs about
    15 ms and the result is reused by every viewer for an hour, whereas doing
    it per render would pay it on every rerun of every session.
    """
    frame = SHP.to_polygons(SHP.load())
    if len(frame):
        frame = frame.copy()
        frame["polygon"] = RM.simplify(list(frame["polygon"]))
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
@resilient("", "the map drawing")
def render_map(fill_items, tip_items, selected, theme, aria):
    """The SVG, cached on everything that can change it.

    THE MAP IS THE PAGE'S DOMINANT COST, and it is bandwidth rather than CPU:
    building it takes about 5 ms, but it is 61 KB of markup that Streamlit
    ships over the websocket on every rerun. On a storm night with a few
    thousand people refreshing, that one element is most of the traffic.

    The arguments are tuples of items rather than dicts because a cache key has
    to be hashable. What actually varies is small: fills and tooltips change
    when a new capture lands, roughly hourly, and selected and theme change
    only when a reader acts. So a few dozen strings serve everybody, and
    identical output between reruns also lets Streamlit's own forward-message
    cache recognise the element instead of resending it.
    """
    polys = load_shapes()
    colours = MAP_COLOURS[theme]
    return RM.render(
        RM.polygons_from(polys), dict(fill_items), selected=selected,
        line=colours["line"], selected_line=colours["selected"],
        background="none", tooltips=dict(tip_items), aria=aria)


@st.cache_data(ttl=300)
def load_closings():
    """What is announced right now. READ ONLY -- never NEWS.collect().

    collect() archives the raw payload to disk whenever it is non-empty, which
    is right for the once-a-morning job and wrong here. On the first real snow
    day this function runs every five minutes for as long as the app is up, so
    calling collect() would write a JSON file every five minutes into a
    container with an ephemeral filesystem and about a gigabyte of memory, to
    archive a payload the Actions job has already archived properly.

    fetch() and parse() have no side effects, which is the whole reason they
    are separate functions from collect().
    """
    try:
        payload, meta = NEWS.fetch()
        if not meta.get("ok"):
            return pd.DataFrame(), meta
        return NEWS.parse(payload, load_roster()), meta
    except Exception:                                   # noqa: BLE001
        return pd.DataFrame(), {"ok": False}


@st.cache_data(ttl=600)
@resilient(lambda: (pd.DataFrame(), None, "none"), "tonight's numbers")
def region_now(target_iso):
    """Latest probability for every district, newest source that has one.

    Order matters: the hourly tracking file is the live picture, the frozen
    call is tonight's official number, and the nightly forecast archive is the
    long stop so the map is never blank.
    """
    hist = TRACK.load_history(target_iso)
    if not hist.empty:
        newest = hist["captured_at"].max()
        latest = hist[hist["captured_at"] == newest]
        return latest[["leaid", "p_closed", "snow_in"]].copy(), newest, "live"

    call = TRACK.load_call(target_iso)
    if not call.empty:
        return call[["leaid", "p_closed", "snow_in"]].copy(), None, "call"

    fdir = os.path.join(ROOT, "data", "forecasts")
    # Prefix + extension, matching every other directory scan in the project.
    # An unfiltered listing would hand read_csv whatever sorted last, so a
    # stray file committed into data/forecasts could take the page down.
    files = sorted(f for f in os.listdir(fdir)
                   if f.startswith("forecast_") and f.endswith(".csv")
                   ) if os.path.isdir(fdir) else []
    if files:
        fc = pd.read_csv(os.path.join(fdir, files[-1]), dtype={"leaid": str})
        fc = fc[fc["target_date"] == fc["target_date"].max()]
        if len(fc):
            out = P.predict(fc, load_roster())
            return out[["leaid", "p_closed", "snow_in"]], None, "archive"
    return pd.DataFrame(), None, "none"


@st.cache_data(ttl=3600)
@resilient(lambda: (pd.DataFrame(), {"district_mornings": 0}), "the accuracy record")
def load_scorecard():
    return SC.refresh()


def fill_for(p, theme):
    """District fill for a probability. Reads as 'more snow', not 'more danger'.

    Interpolates between the theme's low and high anchors, so the direction of
    travel is always toward more prominence against that theme's basemap.
    """
    c = MAP_COLOURS[theme]
    p = max(0.0, min(1.0, float(p)))
    lo, hi = c["low"], c["high"]
    rgb = [int(lo[i] + (hi[i] - lo[i]) * p) for i in range(3)]
    return "#%02x%02x%02x" % tuple(rgb)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

# Fresh numbers from the `data` branch, at most every five minutes, for the
# whole server rather than per viewer. See app/datasync.py. The loaders above
# keep their own TTLs, so new numbers show within one of those after landing.
# Only on Streamlit Cloud (served from /mount/src) unless SCRICSNOW_SYNC=1:
# locally it would overwrite the tracked floor copy in data/ and dirty main.
@st.cache_data(ttl=300, show_spinner=False)
@resilient(0, "the data branch")
def sync_data():
    return datasync.pull()


if (os.path.abspath(ROOT).startswith("/mount/src")
        or os.environ.get("SCRICSNOW_SYNC") == "1"):
    sync_data()

roster = load_roster()
verdict_data = load_verdicts()

# The one file the page cannot do without. Every other loader degrades to a
# missing card; without the roster there is no district to pick, so the code
# below would index an empty frame and put a traceback on the page instead.
#
# Says nothing about why. A reader can do nothing with a filename, and the
# detail is already in the container log where somebody can act on it.
if roster.empty:
    st.markdown("# ❄️ Snow Day Predictor")
    card('<div class="verdict">🔧 Back shortly</div>'
         '<p class="muted">The forecast is briefly unavailable. '
         "Check with your district or the news for anything official.</p>")
    st.stop()

with st.sidebar:
    st.markdown("### ❄️ Settings")
    # Night mode is the default: almost everyone reads this between 8pm and
    # 6am, and a white screen at that hour is unpleasant. Light mode stays
    # available and has to keep working, because people do check at 4pm.
    theme = "dark" if st.toggle("Night mode", value=True) else "light"

# The stylesheet has to sit at the top of the page but cannot be written until
# we know whether snow is forecast, since that decides the falling flakes. A
# placeholder reserves the position and is filled once the data is in.
style_slot = st.empty()

st.markdown("# ❄️ Snow Day Predictor")
st.caption("This app uses machine learning to predict when the school districts "
            "in the South Central Reginoal Information Center (SCRIC) "
            "will close for snow tomorrow morning. This is not an official source; "
            "check your district or the news to hear officially.")

# The district picker lives HERE, in the main column, not in the sidebar. On a
# phone the sidebar collapses behind a hamburger, and this is the single
# control the entire page exists to serve. Buried there, most readers would
# take Binghamton's number for their own.
names = roster["district"].tolist()

# REMEMBERING THE PICK. Binghamton is only the default because something has to
# be, and a parent in Windsor should not have to find their district again every
# night. Two layers, because neither is sufficient alone:
#
#   * The query string (?d=<leaid>) is the only part of this that Python can
#     read. It survives a refresh, it makes a bookmark carry the right district,
#     and it makes the link shareable, which is how one parent tells another.
#     leaid rather than the name, because districts get renamed and the federal
#     id does not.
#
#   * localStorage is what makes a bare visit to the plain URL work, which is
#     how most people actually arrive. Streamlit cannot read it from Python, so
#     a script mirrors the query string into it and, on a visit with no query
#     string at all, puts the saved district back and reloads once.
#
# Nothing is stored server-side. The choice lives in that reader's browser, so
# there is no district-to-person record here to keep or to leak.
DISTRICT_PARAM = "d"

incoming = st.query_params.get(DISTRICT_PARAM)
seed = roster[roster["leaid"] == incoming] if incoming else roster.iloc[0:0]
if len(seed) and seed["district"].iloc[0] in names:
    default = names.index(seed["district"].iloc[0])
else:
    default = next((i for i, n in enumerate(names) if "Binghamton City" in n), 0)


def remember_district():
    """Put the new pick in the URL, so a refresh keeps it."""
    hit = roster.loc[roster["district"] == st.session_state["district_pick"],
                     "leaid"]
    if len(hit):
        st.query_params[DISTRICT_PARAM] = hit.iloc[0]


chosen = st.selectbox("Your school district", names, index=default,
                      key="district_pick", on_change=remember_district,
                      label_visibility="collapsed")
row = roster[roster["district"] == chosen].iloc[0]

# The mirror, in a zero-height component frame. The frame is same-origin, so it
# can reach the real page's storage and address bar.
#
# The query string is read again HERE, after the widget, because on_change has
# by then written this run's pick into it. Read before the widget, a first-ever
# visitor's very first choice would look like no choice at all and never get
# saved.
#
# Only a deliberate pick is stored: with no query string the script restores and
# does not write, so nobody who simply opened the page gets Binghamton recorded
# in their browser as though they had asked for it. The reload needs both an
# absent query string and a saved value, so it happens at most once per visit
# and cannot loop.
_settled = st.query_params.get(DISTRICT_PARAM)
script_frame(f"""
<script>
(function () {{
  try {{
    var KEY = "snowday.leaid";
    var store = window.parent.localStorage;
    if ({json.dumps(_settled is None)}) {{
      var saved = store.getItem(KEY);
      if (saved) {{
        var url = new URL(window.parent.location.href);
        url.searchParams.set({json.dumps(DISTRICT_PARAM)}, saved);
        window.parent.location.replace(url.toString());
      }}
    }} else {{
      store.setItem(KEY, {json.dumps(str(row["leaid"]))});
    }}
  }} catch (e) {{
    /* Private mode, or storage switched off. The default still works. */
  }}
}})();
</script>
""")

now = datetime.now(nws.LOCAL_TZ)
calendar = load_calendar()

# What day to talk about, and whether to put a number on it. Predicting a snow
# day for Christmas is worse than saying nothing, and we only call it the night
# before, because that is the lead time the accuracy record is measured at.
target, predictable, reason_off, lead = SD.target_for(now, calendar)
target_iso = target.isoformat()

region, captured_at, source = (region_now(target_iso) if predictable
                               else (pd.DataFrame(), None, "none"))
mine = region[region["leaid"] == row["leaid"]] if len(region) else pd.DataFrame()

p_now, snow_in = None, 0.0
if len(mine):
    p_now = float(mine["p_closed"].iloc[0])
    snow_in = float(mine["snow_in"].iloc[0])

freshness = (V.reading_age(captured_at, source, nws.LOCAL_TZ, STALE_AFTER)
             if predictable else "")

with style_slot:
    style(theme, snowing=snow_in >= 0.5)
when = target.strftime("%A, %B %d")
st.caption(f"For the morning of **{when}**" if predictable
           else f"Next school day is **{when}**")

closings, meta = load_closings()
# Only announcements about the morning on the page. After 4pm the page has
# moved on to the next school day, and this morning's "Closed Friday" is still
# in the feed.
if len(closings) and "for_date" in closings.columns:
    closings = closings[closings["for_date"].isna()
                        | (closings["for_date"] == target_iso)]

# The feed drops its records mid-morning (11:00 on the first real day), hours
# before the page moves on at 4pm. What was already recorded for this morning
# fills in behind it, so a district that closed keeps reading as closed all
# afternoon. The live feed wins where it still has the district: it is newer.
saved = load_outcomes(B.season_of(target))
if len(saved):
    saved = saved[(saved["date"].astype(str) == target_iso)
                  & saved["status"].notna() & (saved["status"] != "open")]
    live_ids = set(closings["leaid"].dropna()) if len(closings) else set()
    saved = saved[~saved["leaid"].isin(live_ids)]
    if len(saved):
        closings = pd.concat(
            [closings, saved[["leaid", "status"]].assign(status_text="")],
            ignore_index=True)
announced = None
if len(closings):
    hit = closings[closings["leaid"] == row["leaid"]]
    if len(hit):
        announced = hit.iloc[0]

call = TRACK.load_call(target_iso)
call_row = call[call["leaid"] == row["leaid"]] if len(call) else pd.DataFrame()
p_call = float(call_row["p_closed"].iloc[0]) if len(call_row) else None

# --- headline ---------------------------------------------------------------
if not predictable:
    if lead > SD.MAX_LEAD_DAYS and reason_off is not None:
        # School IS on eventually, just not soon enough for us to call it.
        msg, emoji = V.no_school(verdict_data)
        far, _ = V.too_far(verdict_data)
        card(f'<div class="verdict">{emoji} {msg}</div>'
             f'<p class="muted">{SD.describe(reason_off, target)}</p>'
             f'<p class="muted">{far}</p>')
    else:
        msg, emoji = V.no_school(verdict_data)
        card(f'<div class="verdict">{emoji} {msg}</div>'
             f'<p class="muted">{SD.describe(reason_off, target)}</p>')
elif announced is not None:
    status = (announced.get("status") or "closed")
    msg, emoji = V.for_announcement(status, verdict_data)
    card(f'<div class="verdict">{emoji} {msg}</div>'
         f'<p class="muted">{chosen} has announced. '
         f'{announced.get("status_text") or ""}</p>')
elif p_now is None:
    msg, emoji = V.unavailable(verdict_data)
    card(f'<div class="verdict">{emoji} {msg}</div>')
else:
    msg, emoji = V.for_probability(p_now, verdict_data)
    pct = p_now * 100
    sub = ""
    if p_call is not None:
        delta = (p_now - p_call) * 100
        arrow = ("<span class='up'>▲ {:.0f} pts</span>".format(delta)
                 if delta >= 0.5 else
                 "<span class='down'>▼ {:.0f} pts</span>".format(abs(delta))
                 if delta <= -0.5 else "<span class='muted'>unchanged</span>")
        sub = (f'<p class="muted">The 8pm Call was <b>{p_call * 100:.0f}%</b> '
               f"&nbsp;{arrow} since then</p>")
    else:
        sub = ('<p class="muted">The 8pm Call has not been made yet tonight. '
               "This number changes until then.</p>")
    if freshness:
        sub += f'<p class="muted">{freshness}</p>'

    card(f'<div class="big">{pct:.0f}%</div>'
         f'<div class="verdict">{emoji} {msg}</div>'
         f'<div class="meter"><div style="width:{max(2, min(100, pct)):.0f}%"></div></div>'
         f'<p class="muted">chance {chosen} closes</p>{sub}')

# --- why --------------------------------------------------------------------
if predictable:
    reasons = ["nothing much in the forecast"]
    if snow_in > 0 or p_now:
        reasons = H.explain({"snow_total_mm": snow_in * H.MM_PER_INCH})
    pills = "".join(f'<span class="pill">{r}</span>' for r in reasons)
    card(f"<h3>Why?</h3>{pills}"
         f'<p class="muted" style="margin-top:.6rem">'
         f"Forecast snow {snow_in:.1f} in &middot; elevation "
         f'{row["elev_m"] * 3.28084:.0f} ft &middot; '
         f'{now.strftime("%I:%M %p").lstrip("0")}</p>')

# --- the region -------------------------------------------------------------
polys = load_shapes()
if len(polys):
    st.markdown("### Our Districts")
    colours = MAP_COLOURS[theme]
    # leaid -> closed / delay / early, for every district that has announced.
    did = ({leaid: status_bucket(status) for leaid, status in
            zip(closings["leaid"], closings["status"]) if isinstance(leaid, str)}
           if len(closings) else {})
    status_fills = STATUS_FILLS[theme]

    # The map is worth showing on a Sunday in July as well as a Tuesday in
    # January. On a day with no school it carries no probabilities -- every
    # district gets the same quiet fill and the tooltip is just a name -- so it
    # is the region and your place in it, not a prediction wearing a disguise.
    # It costs nothing visually: on a calm night the ramp is flat anyway.
    scored = bool(predictable and len(region))
    probs = dict(zip(region["leaid"], region["p_closed"])) if scored else {}

    fills, tips = {}, {}
    for leaid, name in zip(polys["leaid"], polys["name"]):
        if leaid in did:
            fills[leaid] = status_fills[did[leaid]]
            tips[leaid] = f"{name} - {STATUS_LABELS[did[leaid]].lower()}"
        elif scored:
            p = float(probs.get(leaid, 0.0))
            fills[leaid] = fill_for(p, theme)
            tips[leaid] = f"{name} - {p * 100:.0f}%"
        else:
            fills[leaid] = fill_for(0.0, theme)
            tips[leaid] = name

    svg = render_map(
        tuple(sorted(fills.items())), tuple(sorted(tips.items())),
        row["leaid"], theme,
        aria=(f"Closure chance across {polys['leaid'].nunique()} districts"
              if scored else
              f"The {polys['leaid'].nunique()} districts covered"))

    # st.markdown, never st.html: markdown passes raw HTML straight through,
    # while st.html runs DOMPurify with an HTML-only profile that silently
    # strips every SVG element and leaves a blank space.
    st.markdown(svg, unsafe_allow_html=True)

    if scored:
        note = (f"{colours['word']} means a higher chance. "
                "Your district is outlined.")
    else:
        note = (f"The {polys['leaid'].nunique()} districts covered. "
                "Yours is outlined. No chances shown, because there is no "
                "school to close.")
    st.caption(note)

    # What the region actually did. A legend with words on it, then the same
    # districts as a list, so nothing here depends on telling colours apart.
    if did:
        names = dict(zip(polys["leaid"], polys["name"]))
        swatches, lines = [], []
        for key, label in STATUS_LABELS.items():
            who = sorted(short_name(names[l]) for l, s in did.items()
                         if s == key and l in names)
            if not who:
                continue
            swatches.append(
                f'<span style="white-space:nowrap;margin-right:1.1em">'
                f'<span aria-hidden="true" style="display:inline-block;'
                f'width:.85em;height:.85em;border-radius:2px;'
                f'vertical-align:-.08em;margin-right:.35em;'
                f'background:{status_fills[key]}"></span>{label}</span>')
            lines.append(f"**{label} ({len(who)}):** {', '.join(who)}")
        if lines:
            st.markdown(f'<p class="muted">{"".join(swatches)}</p>',
                        unsafe_allow_html=True)
            st.markdown("  \n".join(lines))

# --- how the number moved ---------------------------------------------------
hist = TRACK.load_history(target_iso)
mine_hist = hist[hist["leaid"] == row["leaid"]] if len(hist) else pd.DataFrame()
if len(mine_hist) > 1:
    st.markdown("### How we did tonight")
    line = mine_hist.set_index("captured_at")[["p_closed"]] * 100
    line.columns = ["chance %"]
    st.line_chart(line, height=220)
    st.caption("Every point is a new forecast. The 8pm Call is the one we "
               "score ourselves on.")

# --- snow day bank ----------------------------------------------------------
bank = load_bank()
if len(bank):
    b = bank[bank["leaid"] == row["leaid"]]
    if len(b):
        built = int(b["days_built_in"].iloc[0])
        used = B.spent(load_outcomes(B.season_of(target)), row["leaid"])
        unspent, gone, makeup = B.pictogram(built, used)

        # Spent days stay on the row, faded, so the bank's size is still
        # readable once most of it is gone. Days past the bank cannot be drawn
        # as negative snowflakes, so they become the thing they actually are:
        # calendar days handed back in June.
        marks = (f'{"❄️" * unspent}'
                 f'<span style="opacity:.3">{"❄️" * gone}</span>'
                 f'<span style="opacity:.75">{"📅" * makeup}</span>')
        card(f"<h3>Snow days in the bank</h3>"
             f'<div style="font-size:2rem;letter-spacing:.35rem">{marks}</div>'
             f'<p class="muted">{B.caption(built, used)}</p>')

# --- accuracy ---------------------------------------------------------------
card_df, summary = load_scorecard()
if summary.get("district_mornings", 0) > 0:
    body = (f"<h3>How accurate have we been?</h3>"
            f'<p>Across <b>{summary["scored_days"]}</b> mornings we called '
            f'<b>{summary["closures_called"]}</b> closures right, missed '
            f'<b>{summary["closures_missed"]}</b>, and cried wolf '
            f'<b>{summary["false_alarms"]}</b> times.</p>')
    skill = summary.get("skill_vs_always_open")
    if skill is not None:
        body += (f'<p class="muted">That is {skill * 100:.0f}% better than '
                 "always guessing that school is on.</p>")
    card(body)
else:
    card("<h3>How accurate have we been?</h3>"
         '<p class="muted">The scoreboard starts at 8pm EST, once it snows.'
         "Call gets recorded here.</p>")

st.markdown(
    '<p class="muted" style="text-align:center;margin-top:2rem">'
    "For entertainment purposes only. Do not use this information as a substitute for official announcements. "
    "Check with your district or an official source.</p>",
    unsafe_allow_html=True)
