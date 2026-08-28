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

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "reference",
                            "verdicts.yaml")


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
    for band in data["bands"]:
        if p <= band["max"]:
            return _pick(band, rng)
    return _pick(data["bands"][-1], rng)


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
