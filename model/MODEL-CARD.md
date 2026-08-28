# Model card: scricsnow

The model behind the snow day predictor. What it is, what it eats, how well it
does, and where it stops being trustworthy. The numbers come from
`model/artifacts/metrics.json`, which the training run writes; if that file and
this one ever disagree, that file is right.

## What it does

Given a National Weather Service forecast for one school district's own patch of
ground, it puts out two probabilities for the next school morning:

- **closed** - a full closure
- **delayed or worse** - a delay, a closure, or a remote day

Two binary logistic regressions over one shared seven-column feature matrix, one
per outcome, clamped afterwards so a closure can never come out likelier than
the disruption that contains it. Fitted in scikit-learn, served as plain
arithmetic in numpy. No neural network, no gradient boosting, no GPU. Fitting
takes about two seconds.

A hand-built heuristic scores the same forecast alongside it, and both numbers
are published. They are not an ensemble and are never averaged. The heuristic
exists because it can see four things the fitted model cannot, since the
historical record has no column for any of them: what hour the snow stops,
whether it is still falling at nine, the wind, and the depth already on the
ground.

**Intended use:** curiosity and entertainment for families in New York's
Southern Tier. It is not advice, not a reason to keep a child home, and no
district uses it to decide anything.

## Inputs

Seven columns. The order is load-bearing: coefficients are stored positionally,
and `predict.load()` refuses to serve an artifact whose feature list disagrees
with `features.py`.

| # | Feature | Unit | Meaning |
|---|---|---|---|
| 1 | `snow_sqrt` | sqrt(mm) | Square root of snowfall for the morning in question |
| 2 | `temp_min_c` | C | Minimum temperature over the window |
| 3 | `elev_m` | m | District ground elevation, USGS 3DEP |
| 4 | `closures_used` | count | Closures this district has already spent this season |
| 5 | `closed_prev` | 0/1 | Did this district close in the previous three days |
| 6 | `closed_prev_snow` | sqrt(mm) | `closed_prev` times `snow_sqrt` |
| 7 | `days_from_peak` | days | Absolute distance from 1 February |

Standardised coefficients, closure head: `snow_sqrt` +1.211, `days_from_peak`
-0.737, `closed_prev` +0.336, `closed_prev_snow` -0.301, `temp_min_c` +0.207,
`closures_used` -0.072, `elev_m` -0.033, intercept -5.065. The delay-or-worse
head has the same shape. Snowfall is the spine, seasonality is the only other
large term, and everything else is a modifier, which is what you want from a
snow day model.

**One rule decides what gets to be a feature: it has to mean the same physical
thing in a forecast and in the historical record.** Anything that does not is
routed to the heuristic, where a hand-set coefficient cannot be corrupted by a
definition mismatch. Storm timing, wind, standing depth and ice all fail that
test and are scored by the heuristic alone.

## Four things worth knowing about the shape

**Snow enters as a square root.** The distance from a clear road to three inches
is most of the decision; the distance from twelve inches to fifteen is almost
none of it. A linear term has to price both the same.

**The season term folds around midwinter rather than running with the
calendar.** Closure rates by month are a hump: 0.0% October, 0.7% November, 2.6%
December, 4.8% January, 5.7% February, 2.6% March, 2.7% April. A straight line
through a hump can only tilt, and the tilt it picks is upward.

**Elevation runs backwards from the obvious guess.** Holding snowfall constant,
on days with two inches or more, districts below 300 m closed about 24% of the
time and districts above 450 m closed 15 to 18%. Higher districts are better at
snow: more ploughs per road mile, and a community that expects it.

**Minimum temperature is positive, conditional on snowfall.** Given that six
inches fell, a night near freezing is heavy wet snow and a night at minus twenty
is powder. "Cold closes schools" is really "nothing closes above freezing", and
the snow column already carries that.

## The reluctance effect

The strongest non-weather signal in the model, and the reason two of the seven
columns are one mechanism. Split the record on whether the district closed in
the previous three days:

| Snowfall | Open before | Closed before | Difference |
|---|---|---|---|
| none | 0.4% | 4.8% | +4.4 |
| under 1 in | 3.2% | 8.2% | +5.0 |
| 1 to 3 in | 10.4% | 12.7% | +2.3 |
| 3 to 6 in | 39.3% | **7.0%** | **-32.3** |
| 6 in or more | 86.5% | **32.3%** | **-54.2** |

**The effect reverses.** On a quiet morning, having closed recently means the
district is in a bad stretch and the rate goes up. On a heavy morning it means
the day is already spent, and 86.5% becomes 32.3% under the same snow. That is
why the model carries both a main effect and an interaction: a single term would
have to average +5 and -54 into one number, and doing so is measurably worse
than having no term at all.

## Training data

27,245 district-mornings across thirteen seasons, 2011 to 2025, for 50 districts
in Broome, Chenango, Delaware, Greene, Otsego, Schoharie and Tioga counties. 696
closures, 209 delays, positive rate 3.32%. Weather from NOAA GHCN daily, blended
per district across its nearest snow-reporting stations and weighted by distance.

2019 and 2020 are excluded on purpose: closing school meant something different
during COVID and those seasons would teach the wrong lesson.

Two details shape everything else. **Open days are constructed, not read:** the
source calendar holds exception days only, so ordinary open mornings are built
from the school-year grid minus the recorded exceptions. And **a silent station
is not zero snow:** co-op observers skip days, and blending a non-report as 0 mm
would dilute a real storm toward calm, which is the dangerous direction to be
wrong in.

Closure labels are believed only if the weather backs them up. That removes real
errors, such as unused reserve days parked at the end of the calendar and the
eleven districts that closed for the 2024 solar eclipse under a clear April sky.
Failing labels are dropped, never flipped to open.

## How well it does

**Validation is grouped by season, and that is the whole ballgame.** A storm is
not one row, it is forty; on a single January morning, forty-six of these
districts closed under the same snow. Split at random and near-duplicates land
on both sides, so the model gets scored on storms it has already seen. Five-fold
`GroupKFold` on season, plus the two most recent seasons held out of fitting
entirely.

Closure head:

| | Model | Heuristic |
|---|---|---|
| AUC, out of fold | 0.928 | 0.867 |
| PR-AUC, out of fold | 0.413 | 0.397 |
| Brier, out of fold | 0.0201 | 0.0211 |
| AUC, held-out seasons | 0.902 | 0.868 |
| Brier, held-out seasons | 0.0161 | 0.0168 |

The delay-or-worse head is similar: 0.936 AUC out of fold, 0.890 held out. The
model leads the heuristic in every cell. What it reliably buys is **ordering**,
which is what the district-by-district ranking on the page depends on; on
calibration alone the heuristic is close behind.

Calibration, out of fold. Storms are shown next to mornings because a
calibration table has the same forty-rows-one-storm problem the validation split
does:

| Band | Mornings | Storms | We said | Actually closed |
|---|---|---|---|---|
| 0 to 2% | 13,220 | 1,283 | 0.5% | 0.5% |
| 2 to 5% | 1,116 | 305 | 3.2% | 3.6% |
| 5 to 10% | 648 | 218 | 7.1% | 7.9% |
| 10 to 20% | 404 | 146 | 13.9% | 15.1% |
| 20 to 40% | 332 | 94 | 27.9% | 26.2% |
| 40 to 70% | 182 | 47 | 52.3% | 43.4% |
| 70 to 100% | 84 | 26 | 83.8% | 75.0% |

Below 40% it is within two points everywhere. Above 40% it runs somewhat
overconfident, on a few dozen storms, which is not enough evidence to call a
measured bias. The most likely cause is the limitation the feature list already
admits to: the model cannot see storm timing, and two mornings with nine inches
look identical to it whether the snow stopped at 10pm or is still falling at 8am.

## Limitations

Ranked by how much they should affect your confidence.

1. **The snow column does not measure the same hours in training as in service.**
   The forecast side is an 18:00-to-09:00 window, about fifteen hours. The
   history side is a GHCN calendar day, and not even a consistent one: airport
   stations read midnight to midnight while co-op observers often read at 7am.
   Same storm, different number of hours. This is a property of the data rather
   than of the code, so it can be narrowed and documented but not closed.
2. **There is look-ahead in the training features.** For midnight-convention
   stations the day's total includes snow that fell after the superintendent
   made the call, and GHCN carries no observation-time metadata to strip it out.
   Read every number above as an optimistic bound. The site's scorecard, which
   grades frozen 8pm calls against observed mornings, has no look-ahead in it.
3. **Trained on observed snowfall, served forecast snowfall.** Six inches
   measured is not the same object as six inches predicted. Recalibrating
   against forecasts, once a full season of paired data exists, is the highest
   value work remaining.
4. **The label filter shares thresholds with the model's features**, so it
   preferentially removes the positives the model would have found hardest. It
   dropped 79 closures and 45 delays; that count bounds the flattery.
5. **The recency features depend on the morning outcome job having run.** If
   yesterday's outcomes are missing, every district reads as "has not closed
   recently", which is exactly the wrong default on day two of a storm.
6. **Nothing in it knows about the district next door.** Peer behaviour is used
   only to decide whether to believe a training label, never as a feature.
7. **Two districts, Afton and Johnson City, have no history behind them.** They
   get full predictions, from weather and terrain, but contribute nothing to
   fitting.
8. **One lead time only.** Fitted and measured at roughly a twelve-hour lead, so
   the site names the next school day and declines to put a number further out.
9. **The shipped artifact is fitted on all thirteen seasons**, including the two
   `metrics.json` reports as held out. That is the right choice for a served
   model, and it means the held-out figures read as a floor rather than as a
   description of the artifact.

## Artifacts and retraining

Retraining is **not part of this repository**. The training set is built from
raw closure history that is not public, so `build_training.py`, `train.py` and
their tests live in a separate private repo. The artifacts below ship here as a
fitted result; nothing here regenerates them, and nothing here needs to.

```powershell
# In the private training repo, not this one:
python internal\build_training.py   # rebuild training.csv from labels + GHCN
python internal\train.py            # fit, validate, write three artifacts
python -m pytest internal\tests -q

# In this repo:
python -m pytest model/ collect/ app/ -q
```

| File | Contents |
|---|---|
| `model/artifacts/model.json` | Coefficients, standardisation, imputation values, feature list |
| `model/artifacts/heuristic.json` | The heuristic's fitted score-to-probability constants |
| `model/artifacts/metrics.json` | Validation results, reliability tables, caveats, training composition |

No artifact contains anything about any individual person, and no personal data
of any kind is used, stored or published. Every input is a public feed.

**If you change `features.py`, you must retrain.** Coefficients are positional,
so an edited feature list would silently shift every one of them onto the wrong
column. `predict.load()` compares the two on every load and raises rather than
serving wrong numbers.

Published predictions are frozen at the moment of the 8pm call and the accuracy
record is scored against that frozen number, so retraining can never rewrite the
past.
