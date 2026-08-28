"""Hand-built snow day scorer. The season-one predictor and the model's baseline.

Why a heuristic at all, when we have 700 labelled closures? Two reasons.

First, it ships without training and is legible: every term below can be
defended to a parent in one sentence, which matters for a public tool.

Second, it uses something the fitted model cannot. Storm end hour is probably
the strongest single predictor of delay-versus-closure, and NWS forecasts give
it to us hourly. GHCN daily observations cannot, so a model trained on history
has no column for it. Until we have a season of paired forecasts, the heuristic
is the only thing here that knows a storm ending at 4am is a different animal
from the same storm still falling at 8am.

The structure is an ordinal one: a single severity score, then two thresholds.
Delay and closure are the same judgement at different intensities, so scoring
them separately would allow the incoherent result of a closure being likelier
than a delay.

Coefficients are on a scale of inches-of-snow-equivalent, so +1.0 means "as
serious as one more inch". They are hand-set: each says how much a condition
matters RELATIVE to snowfall, which is a judgement about roads and buses.
Turning the resulting score into a probability is a separate step and is
fitted, not judged; see the calibration note further down.

Measured out of fold across the full history, grouped by season. The current
numbers live in artifacts/metrics.json rather than in this docstring, because a
number written in prose goes stale silently and one written by the training run
cannot.
"""

import json
import os

import numpy as np
import pandas as pd

MM_PER_INCH = 25.4

# Snowfall is the spine. Everything else is priced against it.
SNOW_PER_INCH = 1.00
BASE = -3.00

# Ice closes schools at amounts that would be trivial as snow. A quarter inch
# of glaze (about 6 mm) shuts a district that would shrug at two inches of snow.
ICE_PER_MM = 0.50

# Cold closes schools on its own, mostly over bus-stop exposure.
COLD_C = -18.0
COLD_BONUS = 1.50

# Snow already down means the ploughs start behind and the shoulders are gone.
DEEP_MM = 100.0
DEEP_BONUS = 0.30

# Wind matters only with loose snow to move: drifting reopens a cleared road.
DRIFT_GUST_KMH = 45.0
DRIFT_MIN_SNOW_MM = 25.0
DRIFT_BONUS = 0.50

# Elevation, and this one is BACKWARDS from the obvious guess.
#
# I first set this positive, reasoning that hill districts get more snow and
# steeper roads. The history says the opposite once snowfall is controlled for:
# on days with 50 mm or more, districts below 300 m closed 24% of the time
# while those above 450 m closed 15-18%. High districts are simply better at
# snow -- more ploughs per road mile, and a community that expects it.
#
# The two effects cancel unconditionally (closure rates are flat at 2.6-2.9%
# across elevation bands), because higher districts also get more snow. Since
# snowfall is already the spine of this score, what is left here is the
# tolerance, which is negative.
ELEV_REFERENCE_M = 300.0
ELEV_PER_M = -1.0 / 400.0

# A district that has spent its emergency days gets noticeably more reluctant.
BANK_PRESSURE = 0.15

# Storm end hour, on the 18-33 scale: 18-23 is the evening before, 24-33 the
# target morning. Ploughs need roughly three clear hours before the buses roll,
# so what matters is not how much fell but whether it stopped in time.
TIMING = [
    (24, -0.80),   # finished before midnight: roads clear by dawn
    (28, -0.30),   # midnight to 3am: tight but workable
    (31, 0.50),    # 4am to 6am: the call gets made mid-cleanup
]
STILL_SNOWING_BONUS = 1.00

# Turning the score into a probability.
#
# The score is a severity ranking; it is not a probability and the conversion
# cannot be eyeballed. My first attempt set even odds at three inches, on the
# grounds that the median closure in the history fell on about three inches.
# That is a base-rate error: three inches is the typical snowfall GIVEN a
# closure, P(snow | closed), and it was being used as P(closed | snow). Most
# three-inch days do not close anybody. Uncorrected it read 94% on days that
# actually close 27% of the time.
#
# So the two pairs below are fitted on the history rather than chosen: a
# one-dimensional logistic of outcome on score, which leaves every hand-set
# coefficient above untouched and only maps the result onto real frequencies.
#
# THEY ARE NO LONGER MAINTAINED BY HAND. train.py, in the private repo, refits
# them on every training run and writes artifacts/heuristic.json, which is what
# load_calibration() below returns. These literals are the fallback for a checkout
# with no artifact yet, and they are the values from the run that produced the
# first published model. If the artifact is present it always wins, so
# rebuilding the training set can no longer leave the heuristic quietly
# calibrated against a dataset that no longer exists.
CLOSED_SLOPE, CLOSED_INTERCEPT = 0.4338, -2.7950
DELAY_SLOPE, DELAY_INTERCEPT = 0.4549, -2.4349

CALIBRATION_JSON = os.path.join(os.path.dirname(__file__), "artifacts",
                                "heuristic.json")
_CACHE = {}


def load_calibration(path=CALIBRATION_JSON):
    """Fitted (slope, intercept) per threshold, from the artifact if it exists.

    Cached on path, because predict() is called per row group on a page render
    and re-reading a 200 byte file each time is pointless. Any failure to read
    or parse falls back to the literals above rather than raising: a missing or
    truncated calibration file should degrade the numbers slightly, not take
    the site down on a snowy morning.
    """
    if path in _CACHE:
        return _CACHE[path]
    fallback = {"closed": (CLOSED_SLOPE, CLOSED_INTERCEPT),
                "disrupted": (DELAY_SLOPE, DELAY_INTERCEPT)}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        cal = {k: (float(raw[k]["slope"]), float(raw[k]["intercept"]))
               for k in ("closed", "disrupted")}
    except (OSError, KeyError, ValueError, TypeError):
        cal = fallback
    _CACHE[path] = cal
    return cal


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def severity_score(df):
    """The single ordinal score. Higher means a more disruptive morning."""
    d = df
    snow_in = d["snow_total_mm"].fillna(0) / MM_PER_INCH
    score = BASE + SNOW_PER_INCH * snow_in

    score = score + ICE_PER_MM * d["ice_mm"].fillna(0)
    score = score + np.where(d["temp_min_c"].fillna(99) <= COLD_C,
                             COLD_BONUS, 0.0)
    score = score + np.where(d["depth_mm"].fillna(0) >= DEEP_MM,
                             DEEP_BONUS, 0.0)
    score = score + np.where(
        (d["gust_kmh"].fillna(0) >= DRIFT_GUST_KMH)
        & (d["snow_total_mm"].fillna(0) >= DRIFT_MIN_SNOW_MM),
        DRIFT_BONUS, 0.0)

    elev = d["elev_m"].fillna(ELEV_REFERENCE_M)
    score = score + (elev - ELEV_REFERENCE_M) * ELEV_PER_M

    if "closures_used" in d:
        score = score - BANK_PRESSURE * d["closures_used"].fillna(0)

    # Timing is forecast-only. Absent it, add nothing rather than guessing:
    # a missing column must not quietly shift every prediction.
    if "snow_end_hour" in d:
        end = d["snow_end_hour"]
        adj = pd.Series(0.0, index=d.index)
        prev = 0
        for hour, bonus in TIMING:
            adj = adj.mask(end.notna() & (end >= prev) & (end < hour), bonus)
            prev = hour
        adj = adj.mask(end.notna() & (end >= prev), TIMING[-1][1])
        score = score + adj
    if "snow_ongoing_at_9am" in d:
        score = score + np.where(d["snow_ongoing_at_9am"].fillna(False),
                                 STILL_SNOWING_BONUS, 0.0)

    return pd.Series(score, index=d.index, name="score")


def calibrate(df, y):
    """Fit score-to-probability constants on labelled rows.

    Exposed so a fair comparison can refit the heuristic inside each validation
    fold. Using constants fitted on the whole history would let the heuristic
    be scored with calibration that had already seen the test seasons, which
    quietly stacks any model-versus-heuristic comparison in its favour.
    """
    from sklearn.linear_model import LogisticRegression

    score = severity_score(df).values.reshape(-1, 1)
    lr = LogisticRegression().fit(score, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def predict(df, closed=None, disrupted=None):
    """Probabilities for the two thresholds, plus the score behind them.

    closed / disrupted optionally override the stored calibration with
    (slope, intercept) fitted elsewhere, which is what fold-wise evaluation
    needs.

    p_disrupted is delay-or-worse and is always at least p_closed, because the
    two come from one score and ordered cut points.
    """
    cal = load_calibration()
    c_slope, c_int = closed if closed else cal["closed"]
    d_slope, d_int = disrupted if disrupted else cal["disrupted"]

    score = severity_score(df)
    out = pd.DataFrame({
        "score": score,
        "p_closed": _sigmoid(c_slope * score + c_int),
        "p_disrupted": _sigmoid(d_slope * score + d_int),
    }, index=df.index)
    # Ordinal coherence: a full closure can never be likelier than the softer
    # call that contains it. The two fits are independent, so on rare scores
    # they can cross by a hair; clamping is cheaper than constraining the fit.
    out["p_disrupted"] = out[["p_closed", "p_disrupted"]].max(axis=1)
    return out


def explain(row):
    """Plain-language reasons, for showing a parent why the number is what it is."""
    reasons = []
    snow_in = (row.get("snow_total_mm") or 0) / MM_PER_INCH
    if snow_in >= 0.5:
        reasons.append(f"{snow_in:.1f} in of snow expected")
    if (row.get("ice_mm") or 0) > 0:
        reasons.append(f"{row['ice_mm']:.1f} mm of ice")
    if (row.get("temp_min_c") or 99) <= COLD_C:
        reasons.append(f"extreme cold ({row['temp_min_c']:.0f} C)")
    if (row.get("depth_mm") or 0) >= DEEP_MM:
        reasons.append("snow already on the ground")
    if ((row.get("gust_kmh") or 0) >= DRIFT_GUST_KMH
            and (row.get("snow_total_mm") or 0) >= DRIFT_MIN_SNOW_MM):
        reasons.append("wind strong enough to drift")
    end = row.get("snow_end_hour")
    if end is not None and not pd.isna(end):
        clock = int(end) % 24
        if end >= 28:
            reasons.append(f"snow still falling at {clock}:00, into the morning")
        elif end < 24:
            reasons.append("snow ends before midnight, time to plough")
    if row.get("snow_ongoing_at_9am"):
        reasons.append("still snowing at 9am")
    return reasons or ["nothing much in the forecast"]
