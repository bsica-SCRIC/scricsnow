"""The one definition of a feature row, shared on purpose.

Training and prediction both import this module rather than each building their
own columns. If they ever computed features differently the model would be
trained on one definition and served another, which does not raise and does not
crash. Accuracy just quietly degrades and nothing points at the cause. One
implementation imported by both sides is the only safe arrangement.

THE HARD PART: history and forecasts are different objects.

    history   NOAA GHCN daily observations, blended across a district's nearest
              snowfall-reporting stations. Daily totals, no hourly detail.
    forecast  NWS gridpoint, hourly, for the district's own grid cell.

THE RULE THIS MODULE NOW ENFORCES: a column is a model feature only if both
sources can produce the SAME PHYSICAL QUANTITY for it. Everything else is
carried as a diagnostic, available to the heuristic and to the page, and kept
out of the model matrix.

That rule cost us the evening/morning snowfall split, and the reason is worth
writing down, because the split looked fine for a long time and was not.

    forecast  the capture window is 18:00 to 09:00, about fifteen hours, and
              the overnight portion of it is known to the hour.
    history   GHCN gives SNOW[D], one number for a calendar day. There is no
              way to cut it at midnight.

The previous version faked the split: snow_evening_mm was SNOW[D-1] and
snow_morning_mm was SNOW[D], so the model's snow_total_mm was a forty-eight
hour sum while the forecast served it a fifteen hour one. Same column name,
different physical quantity, and the largest coefficient in the model sat on
the worst-affected column. The three columns were also exactly collinear by
construction (total = evening + morning), so the fitted split between them was
an artifact of the L2 penalty rather than anything the data said.

So the model now gets ONE snow column:

    snow_total_mm   the snowfall attached to this decision
                    obs:  SNOW[D], the calendar day of the morning in question
                    fcst: the 18:00-to-09:00 window total

These are still not identical: twenty-four hours against fifteen. GHCN daily
boundaries are not all midnight either. Airport stations run midnight to
midnight, but co-op observers often read at 7am, so their SNOW[D] covers
roughly 7am D-1 to 7am D, and blending stations mixes conventions. But the two
are now the same OBJECT, the storm this morning's call is about, rather than
two quantities differing by a whole extra day.

KNOWN AND UNFIXABLE WITH THIS DATA: for midnight-convention stations, SNOW[D]
includes snow that fell in the afternoon of D, hours after the superintendent
made the call. That is look-ahead in the training features and it cannot be
removed, because GHCN daily carries no observation-time metadata we can use to
tell the conventions apart. It is the main reason to read the cross-validated
AUC as an optimistic bound rather than as an estimate of what tomorrow morning
will feel like. The published scorecard, which scores frozen forecasts against
observed mornings, is the number with no look-ahead in it.

WHAT IS DELIBERATELY ABSENT, and why each one:

    snow_end_hour, snow_ongoing_at_9am
        Probably the strongest single predictors of delay-versus-closure, and
        NWS gives both. GHCN daily cannot, so a model trained on history has no
        column for them. The heuristic uses them.

    snow_overnight_mm
        The exact midnight-to-09:00 total. Real and precise in a forecast,
        impossible in history, per the argument above.

    gust_kmh
        NWS forecasts wind reliably, but only 3.7% of historical district-days
        have it, because co-op observers rarely report wind. A column imputed
        96% of the time spends a parameter to learn nothing. The heuristic
        still uses forecast gusts directly.

    elev_gap_m
        The elevation difference between a district and the stations reporting
        its weather. This is a property of our MEASUREMENT, not of the world.
        At prediction time the gridpoint forecast is for the district's own
        cell, so the gap is structurally zero on every served row, while
        training rows averaged 55 m. Keeping it meant every live prediction sat
        half a standard deviation below the training mean on a column with a
        real positive coefficient: a constant offset applied to the whole
        region that no validation split ever measured. The station blend
        already down-weights distant stations by distance, and that is where
        the correction belongs.

    ice_mm
        The same definition mismatch as the snow columns, caught later because
        the column is small and rare enough to hide in.

            history   there is no ice measurement in GHCN, so this is DERIVED:
                      the day's total liquid precipitation, in mm, whenever the
                      maximum temperature was at or below 1 C and little snow
                      fell. That is a plausible proxy for a freezing-rain day.
            forecast  NWS iceAccumulation, which is forecast GLAZE THICKNESS.

        Liquid precipitation and glaze thickness are not the same quantity and
        do not even share an order of magnitude: ten millimetres of rain at
        minus one leaves a few millimetres of ice. So the model learned a
        coefficient against the larger quantity and was served the smaller one,
        which made it badly under-weight forecast ice. Working the fitted
        coefficient back into real units, it priced a quarter inch of glaze at
        about 0.4 inches of snow, while the heuristic, whose term was hand-set
        in glaze units, prices it at three.

        The history cannot settle which is right. There are 30 district-mornings
        with 3 mm or more of the proxy and two of them closed; eleven at 6 mm or
        more and one closed. The bands are not even monotonic (14.6% closure at
        1-3 mm, 5.3% at 3-6 mm), which is what noise looks like. A coefficient
        fitted on that is not evidence that ice does not matter, it is an
        almost-empty column.

        Dropping it improved every metric on both heads, which settles the
        engineering question even though it leaves the meteorological one open.
        Freezing rain is now scored only by the heuristic, where the term is in
        the right units and rests on a judgement about bus routes rather than on
        thirty observations.

    dow
        Day of week. Dropped because it earned nothing: removing it moved
        out-of-fold AUC by 0.0000 and Brier by 0.00003, on both heads.

        There is also a structural reason it could never have helped the thing
        this tool is actually for. dow is constant across every district on a
        given night, so it cannot change the district-by-district ordering the
        page displays. It can only raise or lower the whole region at once, and
        the Brier score says it does not usefully do that either.

        If anyone re-adds it, note that the training grid is built with
        freq="B", so the column only ever took values 0 to 4. A linear term
        fitted on Monday-to-Friday extrapolates off the end for a Saturday.

    depth_mm
        Standing snow. Physically real, and the heuristic still scores it,
        but it did not survive as a FITTED term. Its coefficient swung from
        0.02 to 0.31 across the five validation folds, because it is 47%
        correlated with minimum temperature and 39% with distance from
        midwinter, so it was mostly re-stating "deep winter, and cold" in a
        third way. Dropping it left out-of-fold AUC flat (0.903 to 0.902) and
        improved both PR-AUC (0.370 to 0.386) and Brier (0.0217 to 0.0211),
        and this tool publishes probabilities, so calibration outranks the
        third decimal of a ranking metric.

        It is also close to inert in production: no forecast of depth exists,
        so unless a caller passes observed depth the column is imputed on
        nearly every served row and contributes a constant. An unstable
        coefficient that is usually imputed and occasionally real is the worst
        of both. The mechanism it stands for, ploughs starting behind on ground
        that is already covered, is kept in heuristic.py where the term is
        hand-set and cannot swing. predict() still accepts depth_mm and it
        still moves the heuristic number.

A PATTERN WORTH NAMING, because it took four separate discoveries to see it.
Every feature removed above was removed for the same underlying reason: the
column did not mean the same thing in the two places it was used. The snow
split was fabricated in history. elev_gap_m was structurally zero in service.
ice_mm was liquid precipitation on one side and glaze on the other. depth_mm
was real on both sides but so sparse in service that it was imputed on nearly
every row. Four columns, one mistake, made four times.

The model got smaller and better every single time. That is not a coincidence
and it is not a general argument for fewer features. It is what happens when
you delete columns that were contributing noise dressed as signal.

SELECTION HYGIENE. days_from_peak, the square-root snow transform, and the
removals of depth_mm, ice_mm and dow were all chosen on development-season
cross-validation. The two holdout seasons were not consulted while choosing,
and they confirmed the result afterwards rather than guiding it: the closure
head went from 0.875 to 0.911 out of sample. Anyone adding a feature should
keep that order, because a holdout looked at twice is not a holdout.
"""

import numpy as np
import pandas as pd

# Order matters: it is the column order of the model matrix, and model.json
# stores coefficients positionally against it.
FEATURES = [
    "snow_sqrt",
    "temp_min_c",
    "elev_m",
    "closures_used",
    "closed_prev",
    "closed_prev_snow",
    "days_from_peak",
]

# How far back "we just had one" reaches, in calendar days. Three, so that a
# Monday sees the previous Friday.
RECENT_CLOSURE_DAYS = 3

# What counts as the district having cancelled school, for closures_used and
# closed_prev. BOTH kinds, and the second one is easy to miss.
#
# conditional_closed is a district calendar code for a scheduled conditional day
# that got used. build_training already promotes it to severity 2, so the LABEL
# has always treated it as a closure. The features did not: they filtered on
# kind == "closed" alone and so were blind to 173 of 1,167 closure events, about
# 15%. A row could therefore be labelled "this district closed" while the
# feature describing the very next morning said "this district has not closed
# recently". Label and feature disagreeing about the same event is a defect
# whatever it does to the metrics, and it happens to improve them slightly.
#
# Caveat kept in view: events.csv also holds unused reserve days parked in May
# and June under this same code. They sit outside the October-April window so
# they cannot reach a training row through a three-day lookback, and
# closures_used counts strictly earlier days so a May entry cannot reach a
# February one either. If the window ever widens, revisit this.
#
# The serving side matches by carrying only "closed" from the WBNG feed, because
# a district that would have used a conditional day simply announces a closure.
# See predict.closure_history for why "remote" is deliberately not in that list.
CLOSURE_KINDS = ("closed", "conditional_closed")

# closed_prev AND closed_prev_snow, always together, because the effect REVERSES
# with snowfall and a single term cannot say that.
#
# Measured on the development seasons, closure rate split on whether the
# district closed in the previous three days:
#
#     snowfall     open before    closed before    difference
#     none              0.4%             4.8%          +4.4
#     under 1 in        3.2%             8.2%          +5.0
#     1 to 3 in        10.4%            12.7%          +2.3
#     3 to 6 in        39.3%             7.0%         -32.3
#     6 in or more     86.5%            32.3%         -54.2
#
# Two different mechanisms, pointing opposite ways. On a quiet morning, having
# closed recently means the district is in a bad stretch: snow on the ground,
# roads not recovered, and the closure rate goes UP. On a big-snow morning it
# means the district has just spent a day, and it toughs the second one out.
# Eighty-six percent becomes thirty-two.
#
# That second row is the answer to a real defect. The model used to be
# overconfident above 70% by about twenty points, saying 82% where 61% of
# mornings closed, and this is a large part of why: it was averaging over
# districts that had just closed and districts that had not, which behave
# nothing alike under the same snow.
#
# The interaction is not optional. Adding closed_prev on its own makes every
# metric WORSE than leaving it out entirely (closure AUC 0.916 to 0.898),
# because a single coefficient has to average +5 and -54 into one number. With
# the interaction, closure AUC goes to 0.929 and the top-band gap halves.

# The model sees the SQUARE ROOT of snowfall in mm, not the raw depth.
#
# Snow does not act linearly on a superintendent. The distance from a clear road
# to three inches is most of the decision; the distance from twelve inches to
# fifteen is almost none of it, because by then the answer was already yes. A
# linear term has to price both the same and it splits the difference, which
# left dry midwinter mornings reading around 3% when nothing at all was falling.
#
# Out of fold on the development seasons, against the raw column: closure AUC
# 0.902 to 0.916, PR-AUC 0.386 to 0.389, Brier 0.02113 to 0.02056. Better on
# every metric and on both heads. log1p was also tried and is too aggressive,
# flattening the top of the range where a genuine blizzard needs to separate
# from an ordinary storm (PR-AUC falls to 0.344).
#
# snow_total_mm stays in the frame in raw mm. It is what the heuristic scores,
# what the page prints as inches, and what a human reads in an export.
def snow_transform(mm):
    """Raw millimetres to the model's snow column. One definition, both sides."""
    return np.sqrt(np.clip(np.asarray(mm, dtype=float), 0.0, None))

# Depth of winter, counted in days from 1 December, so 62 is roughly 1 February.
#
# days_from_peak REPLACED a plain linear days_into_season, and the reason is
# that closures are not monotonic in the date. Measured on the development
# seasons, the closure rate by month runs 0.0% in October, 0.7% in November,
# 2.6% in December, 4.8% in January, 5.7% in February, then back down to 2.6%
# in March and 2.7% in April. That is a hump, and a linear term can only tilt.
#
# The tilt it chose was upward, which meant the old model believed that six
# inches in late April was a likelier closure than the same six inches in
# January. It is not; April was simply on the far end of the line.
#
# The distance-from-peak form is not fitted to a knife edge. Out-of-fold AUC
# for the closure head is 0.898 to 0.904 anywhere in a peak between 55 and 76,
# against 0.886 for the linear term, so 1 February is chosen because it is the
# defensible middle of winter rather than because it squeezed out a decimal.
WINTER_PEAK_DAY = 62

# Carried through the canonical frame but NOT fed to the model. Each is
# available from one source only, or is too sparse in history to fit. Named
# here so the asymmetry is visible rather than discovered.
#
#   snow_total_mm       both. Raw millimetres. The model sees snow_sqrt; this
#                       is what the heuristic scores and what humans read.
#   snow_prior_mm       obs only.  SNOW[D-1], for diagnostics and label review.
#   snow_overnight_mm   fcst only. Exact midnight-to-09:00 total.
#   gust_kmh            both, but 3.7% coverage in history. Heuristic only.
#   ice_mm              both, but not the same quantity on each side, and only
#                       thirty real events. Heuristic only; see the note above.
#   depth_mm            both. Heuristic only; see the note above.
#   dow                 both. Truthful here, 0 to 6, deliberately NOT clipped:
#                       a diagnostic that reported Saturday as Friday would be
#                       lying in an export to save a model that no longer reads
#                       the column. See the note above before re-adding it.
#   days_into_season    both. Signed, and kept because days_from_peak is folded
#                       and cannot tell an October morning from an April one.
#                       Readable in exports; not a feature, see WINTER_PEAK_DAY.
DIAGNOSTICS = ["snow_total_mm", "snow_prior_mm", "snow_overnight_mm",
               "gust_kmh", "ice_mm", "depth_mm", "dow", "days_into_season"]

# Roughly 1,100 weather events across the whole history. At the usual rule of
# thumb of ten events per parameter, twelve features is the ceiling. We sit at
# five, and the headroom is deliberate: prefer replacing a weak feature over
# adding one, and never add a column only one source can produce.
#
# Five is not a target either. Four columns have been removed from this list on
# evidence, and each removal improved the out-of-fold numbers, but that is a
# fact about those four columns rather than a principle. A genuinely new signal
# measured the same way on both sides would be welcome.
MAX_SENSIBLE_FEATURES = 12


def from_observations(obs):
    """Canonical rows from blended GHCN daily observations.

    `obs` needs: leaid, date, snow_mm, snow_prior_mm, depth_mm, prcp_mm,
    tmin_c, tmax_c, gust_ms, elev_m.

    depth_mm is expected already lagged to the end of the previous day by the
    caller. Depth measured on the morning of D includes the storm that closed
    school, so using it unlagged would let the outcome predict itself.
    """
    df = pd.DataFrame(index=obs.index)
    df["leaid"] = obs["leaid"]
    df["date"] = obs["date"]

    # SNOW[D] alone, NOT SNOW[D-1] + SNOW[D]. See the module docstring.
    df["snow_total_mm"] = obs["snow_mm"].fillna(0.0)
    df["snow_sqrt"] = snow_transform(df["snow_total_mm"])
    df["snow_prior_mm"] = obs["snow_prior_mm"]
    df["snow_overnight_mm"] = np.nan
    df["depth_mm"] = obs["depth_mm"].fillna(0.0)

    # GHCN has no ice accumulation. Freezing rain is precipitation falling with
    # the temperature at or below freezing, which is recoverable from the daily
    # values and is the condition that closes schools on tiny amounts.
    freezing = (obs["tmax_c"].fillna(99) <= 1.0)
    df["ice_mm"] = np.where(freezing & (df["snow_total_mm"] < 5.0),
                            obs["prcp_mm"].fillna(0.0), 0.0)

    df["temp_min_c"] = obs["tmin_c"]
    # GHCN wind is m/s; the forecast side is km/h. Convert here so no column
    # ever carries two units.
    df["gust_kmh"] = obs["gust_ms"] * 3.6
    df["elev_m"] = obs["elev_m"]
    return df


def from_forecast(fcst, roster):
    """Canonical rows from an NWS overnight snapshot.

    snow_total_mm is the 18:00-to-09:00 window collect/nws.py captures, which
    is the closest available match to GHCN's SNOW[D]. The exact overnight split
    is kept as a diagnostic: it is real here and unavailable in history, so it
    informs the page and the heuristic and never reaches the model matrix.
    """
    r = roster[["leaid", "elev_m"]]
    df = fcst.merge(r, on="leaid", how="left")

    out = pd.DataFrame(index=df.index)
    out["leaid"] = df["leaid"]
    out["date"] = pd.to_datetime(df["target_date"])

    out["snow_total_mm"] = df["snow_mm"].fillna(0.0)
    out["snow_sqrt"] = snow_transform(out["snow_total_mm"])
    out["snow_overnight_mm"] = df["snow_mm_overnight"].fillna(0.0)
    out["snow_prior_mm"] = np.nan

    # No forecast of standing snow depth exists. Callers should supply the
    # latest observed depth. Left MISSING rather than zeroed when they do not,
    # so the imputation value stored at training time decides what an unknown
    # depth is worth, instead of this module asserting that unknown means none.
    out["depth_mm"] = df["depth_mm"] if "depth_mm" in df.columns else np.nan

    out["ice_mm"] = df["ice_mm"].fillna(0.0)
    out["temp_min_c"] = df["temp_min_c"]
    out["gust_kmh"] = df["gust_max_kmh"]
    out["elev_m"] = df["elev_m"]
    return out


def add_calendar_context(df, events=None, season_start_month=7):
    """Features about the season and the district's own recent history.

    The decision is not purely meteorological. A district that has burned its
    emergency days gets more reluctant across a season (closures_used), and one
    that closed two days ago is much more reluctant TODAY under heavy snow
    (closed_prev, with its interaction).

    `events` needs columns leaid, date, kind, with kind == "closed" marking a
    closure. Both training and serving derive every column below from that one
    shape, on purpose: the training side builds it from the closure history and
    the serving side from data/outcomes, and they must not drift.

    PASSING NO EVENTS IS A REAL CHOICE AND USUALLY THE WRONG ONE. With
    events=None every history column is zero, which is not "unknown", it is the
    positive claim that this district has closed nothing all season and did not
    close yesterday. On a second storm morning that is exactly backwards and it
    is the direction that inflates the number. predict() therefore loads the
    history from disk rather than defaulting to None; see predict.closure_history.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # Not clipped. dow is a diagnostic now, not a feature, so there is no
    # coefficient to protect from a weekend and no reason to misreport one.
    out["dow"] = out["date"].dt.dayofweek
    out["season"] = out["date"].map(
        lambda d: d.year if d.month >= season_start_month else d.year - 1)

    # Days since December 1 of the season. Negative in the autumn. Kept as a
    # readable diagnostic; the model sees the folded version below.
    dec1 = out["season"].map(lambda s: pd.Timestamp(year=s, month=12, day=1))
    out["days_into_season"] = (out["date"] - dec1).dt.days
    # Distance from the depth of winter, in days, always positive. October and
    # April both sit far out; January and February sit near zero. See the note
    # on WINTER_PEAK_DAY for why this is folded rather than signed.
    out["days_from_peak"] = (out["days_into_season"] - WINTER_PEAK_DAY).abs()

    if events is None or len(events) == 0:
        out["closures_used"] = 0
        out["closed_prev"] = 0.0
        out["closed_prev_snow"] = 0.0
        return out

    ev = events[events["kind"].isin(CLOSURE_KINDS)].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["season"] = ev["date"].map(
        lambda d: d.year if d.month >= season_start_month else d.year - 1)
    ev = ev.sort_values("date")

    # Strictly-earlier count only. Including the current day would leak the
    # label straight into the features.
    counts = []
    by_key = {k: g["date"].values for k, g in ev.groupby(["leaid", "season"])}
    for leaid, season, date in zip(out["leaid"], out["season"], out["date"]):
        prior = by_key.get((leaid, season))
        counts.append(0 if prior is None else int((prior < date.to_datetime64()).sum()))
    out["closures_used"] = counts

    # Did this district close in the last RECENT_CLOSURE_DAYS calendar days?
    #
    # A lookback over the closure LIST, deliberately, rather than reading the
    # previous row of a grid. Serving has no grid: it holds one forecast row for
    # tomorrow and a history of who closed when. A definition that needs the
    # training layout could not be reproduced there, which is how the last four
    # train/serve mismatches started.
    #
    # The two agree on 97.8% of training rows, and the lookback scores better on
    # the closure head besides.
    closed_on = set(zip(ev["leaid"], ev["date"]))
    offsets = [pd.Timedelta(days=k) for k in range(1, RECENT_CLOSURE_DAYS + 1)]
    out["closed_prev"] = [
        1.0 if any((leaid, date - off) in closed_on for off in offsets) else 0.0
        for leaid, date in zip(out["leaid"], out["date"])]

    if "snow_sqrt" not in out.columns:
        raise ValueError(
            "add_calendar_context needs snow_sqrt to build closed_prev_snow. "
            "Call from_observations or from_forecast first.")
    out["closed_prev_snow"] = out["closed_prev"] * out["snow_sqrt"]
    return out


def matrix(df, impute=None):
    """The model matrix, in FEATURES order, with no missing values.

    `impute` is the fill value per column. Pass the values stored at training
    time. Letting each call compute its own medians would mean one district on
    one night imputes from that night's 50 rows, while training imputed from
    25,000: the same input scoring differently depending on what it arrived
    alongside. Train/serve skew of exactly the kind this module exists to stop.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame is missing columns: {missing}")
    x = df[FEATURES].astype(float)
    if impute is None:
        impute = x.median(numeric_only=True).to_dict()
    return x.fillna(pd.Series(impute)).fillna(0.0)


def impute_values(df):
    """Per-column fill values to store in the model artifact."""
    return df[FEATURES].astype(float).median(numeric_only=True).to_dict()
