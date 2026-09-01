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

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "collect"))

import bank as B           # noqa: E402
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

ANNOUNCED_FILL = "#facc15"               # gold, legible in either theme


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


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_roster():
    return pd.read_csv(os.path.join(ROOT, "reference", "roster.csv"),
                       dtype={"leaid": str}).sort_values("district")


@st.cache_data(ttl=3600)
def load_bank():
    return B.load()


@st.cache_data(ttl=300)
def load_outcomes(season):
    return B.load_outcomes(season)


@st.cache_data(ttl=3600)
def load_verdicts():
    return V.load()


@st.cache_data(ttl=3600)
def load_calendar():
    return SD.load()


@st.cache_data(ttl=3600)
def load_shapes():
    return SHP.to_polygons(SHP.load())


@st.cache_data(ttl=300)
def load_closings():
    try:
        return NEWS.collect(verbose=False)
    except Exception:                                   # noqa: BLE001
        return pd.DataFrame(), {"ok": False}


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=900)
def live_one(leaid, lat, lon, district, target_iso):
    """One gridpoint call, for when the stored reading has gone stale."""
    row = pd.DataFrame([{"leaid": leaid, "district": district,
                         "lat": lat, "lon": lon}])
    graded = nws.ensure_grids(row)
    if graded["office"].isna().all():
        return None
    g = graded.iloc[0]
    props = nws.get(f"{nws.API}/gridpoints/{g['office']}"
                    f"/{int(g['grid_x'])},{int(g['grid_y'])}")["properties"]
    target = datetime.fromisoformat(target_iso).date()
    return pd.DataFrame([{"target_date": target_iso, "leaid": leaid,
                          "district": district,
                          **nws.features(props, target)}])


@st.cache_data(ttl=3600)
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

roster = load_roster()
verdict_data = load_verdicts()

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
st.caption("This app uses machine learning to predict when the school districts"
            "in the South Central Reginoal Information Center (SCRIC)"
            "will close for snow tomorrow morning. This is not an official source;"
            "check your district or the news to hear officially.")

# The district picker lives HERE, in the main column, not in the sidebar. On a
# phone the sidebar collapses behind a hamburger, and this is the single
# control the entire page exists to serve. Buried there, most readers would
# take Binghamton's number for their own.
names = roster["district"].tolist()
default = next((i for i, n in enumerate(names) if "Binghamton City" in n), 0)
chosen = st.selectbox("Your school district", names, index=default,
                      label_visibility="collapsed")
row = roster[roster["district"] == chosen].iloc[0]

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

stale = (captured_at is None
         or (datetime.now(timezone.utc) - captured_at.to_pydatetime()) > STALE_AFTER)
if predictable and (p_now is None or (stale and source != "archive")):
    fc = live_one(row["leaid"], float(row["lat"]), float(row["lon"]),
                  row["district"], target_iso)
    if fc is not None and len(fc):
        out = P.predict(fc, roster)
        p_now = float(out["p_closed"].iloc[0])
        snow_in = float(out["snow_in"].iloc[0])
        source = "live"

with style_slot:
    style(theme, snowing=snow_in >= 0.5)
when = target.strftime("%A, %B %d")
st.caption(f"For the morning of **{when}**" if predictable
           else f"Next school day is **{when}**")

closings, meta = load_closings()
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
    closed_ids = (set(closings.loc[closings["leaid"].notna(), "leaid"])
                  if len(closings) else set())

    # The map is worth showing on a Sunday in July as well as a Tuesday in
    # January. On a day with no school it carries no probabilities -- every
    # district gets the same quiet fill and the tooltip is just a name -- so it
    # is the region and your place in it, not a prediction wearing a disguise.
    # It costs nothing visually: on a calm night the ramp is flat anyway.
    scored = bool(predictable and len(region))
    probs = dict(zip(region["leaid"], region["p_closed"])) if scored else {}

    fills, tips = {}, {}
    for leaid, name in zip(polys["leaid"], polys["name"]):
        if leaid in closed_ids:
            fills[leaid] = ANNOUNCED_FILL
            tips[leaid] = f"{name} - announced"
        elif scored:
            p = float(probs.get(leaid, 0.0))
            fills[leaid] = fill_for(p, theme)
            tips[leaid] = f"{name} - {p * 100:.0f}%"
        else:
            fills[leaid] = fill_for(0.0, theme)
            tips[leaid] = name

    svg = RM.render(
        RM.polygons_from(polys), fills, selected=row["leaid"],
        line=colours["line"], selected_line=colours["selected"],
        background="none", tooltips=tips,
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
        if closed_ids:
            note = f"Gold means already announced. {note}"
    else:
        note = (f"The {polys['leaid'].nunique()} districts covered. "
                "Yours is outlined. No chances shown, because there is no "
                "school to close.")
    st.caption(note)

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
