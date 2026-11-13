"""Plain-language verdicts, loaded from reference/verdicts.yaml.

Kept out of the app code so the wording can be edited without touching Python,
and out of the model so nobody mistakes it for something that affects a
prediction. It is presentation only.

Bands are validated on load rather than trusted. A hand-edited YAML file is
going to get a typo eventually, and the failure mode without checks is a page
that renders no verdict at all for a range of probabilities, which nobody would
notice until it happened to land there on a storm night.
"""

import os
import random
from datetime import datetime, timezone

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "reference",
                            "verdicts.yaml")


# The wording of last resort, for when verdicts.yaml cannot be read at all.
#
# load() validates and raises rather than serving a file it does not trust,
# which is right: a band gap means some probability has nothing to say, and
# that is invisible until the night it lands there. But the page still has to
# render. These five lines are flat and unfunny on purpose -- they are what a
# reader sees when the file is broken, and dull is better than absent.
FALLBACK_BANDS = [
    {"max": 0.05, "messages": ["School is on."], "emoji": "🎒"},
    {"max": 0.20, "messages": ["Probably school."], "emoji": "🌨️"},
    {"max": 0.50, "messages": ["It could go either way."], "emoji": "🤔"},
    {"max": 0.80, "messages": ["A snow day looks likely."], "emoji": "❄️"},
    {"max": 1.00, "messages": ["A snow day looks very likely."], "emoji": "⛄"},
]


class VerdictError(ValueError):
    """Raised when the YAML cannot be trusted to cover every probability."""


def load(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    validate(data)
    return data


def validate(data):
    bands = data.get("bands") or []
    if not bands:
        raise VerdictError("verdicts.yaml has no bands")

    last = 0.0
    for i, band in enumerate(bands):
        if "max" not in band:
            raise VerdictError(f"band {i} ({band.get('name')}) has no max")
        if not band.get("messages"):
            raise VerdictError(f"band {band.get('name')} has no messages")
        if band["max"] <= last and i > 0:
            raise VerdictError(
                f"band {band.get('name')} has max {band['max']}, which is not "
                f"above the previous band's {last}. Bands must ascend.")
        last = band["max"]

    # The gap that would otherwise only show up on a storm night.
    if abs(last - 1.0) > 1e-9:
        raise VerdictError(
            f"the last band ends at {last}, so probabilities above it get no "
            "verdict. The final band must have max: 1.0")
    return True


def _pick(band, rng):
    emoji = band.get("emoji") or ""
    if isinstance(emoji, list):
        emoji = rng.choice(emoji) if emoji else ""
    return rng.choice(band["messages"]), emoji


def for_probability(p, data=None, rng=None):
    """Return (message, emoji) for a probability in [0, 1]."""
    data = data or load()
    rng = rng or random
    p = min(max(float(p), 0.0), 1.0)
    # .get, not [], so a caller that handed us a broken or empty file still
    # gets a sentence rather than a KeyError on the public page.
    bands = data.get("bands") or FALLBACK_BANDS
    for band in bands:
        if p <= band["max"]:
            return _pick(band, rng)
    return _pick(bands[-1], rng)


def for_announcement(status, data=None, rng=None):
    """Wording for a district that has already announced.

    Falls back to the closure wording for an unrecognized status: if the feed
    says something we do not have a phrase for, saying nothing is worse than
    saying the district has announced.
    """
    data = data or load()
    rng = rng or random
    section = (data.get("announced") or {})
    band = section.get(status) or section.get("closed")
    if not band or not band.get("messages"):
        return "The district has announced.", "📢"
    return _pick(band, rng)


def _section(name, fallback, data, rng):
    data = data or load()
    rng = rng or random
    band = data.get(name)
    if not band or not band.get("messages"):
        return fallback, ""
    return _pick(band, rng)


def unavailable(data=None, rng=None):
    return _section("unavailable", "No forecast available right now.",
                    data, rng)


def no_school(data=None, rng=None):
    """Wording for a day school is out anyway."""
    return _section("no_school", "No school tomorrow anyway.", data, rng)


def too_far(data=None, rng=None):
    """Wording for a school day further out than the night before."""
    return _section("too_far", "We only call it the night before.", data, rng)


def reading_age(captured_at, source, tz, stale_after, now=None):
    """How old the displayed number is, said plainly.

    THIS REPLACED A PER-VIEWER LIVE FETCH, and the reason is worth keeping.

    The page used to call the NWS gridpoint API itself whenever the stored
    reading was missing or older than stale_after. That is the wrong shape for
    a public page: it turns one broken capture job into thousands of blocking
    one-to-three-second requests, at exactly the moment the app is busiest, and
    it aims them all at a free API from a single address.

    It also fired on ordinary nights rather than only on broken ones. On a
    quiet night nothing is worth recording, so no history row is written,
    region_now falls through to the frozen call, and the call carries no
    capture time -- which read as stale, so every viewer fetched.

    Saying the number is from 8:42 is better than a scramble to hide that it
    is. If the capture job is down, that is worth showing, not papering over.
    """
    if captured_at is not None:
        local = captured_at.tz_convert(tz)
        stamp = local.strftime("%I:%M %p").lstrip("0")
        now = now or datetime.now(timezone.utc)
        if (now - captured_at.to_pydatetime()) > stale_after:
            return f"Last updated {stamp}. Nothing newer has come in."
        return f"Updated {stamp}"
    return {"call": "This is tonight's 8pm Call.",
            "archive": "From the last stored forecast."}.get(source, "")
