"""How accurate were we? This one is meant to be shown.

A prediction tool that never publishes its record is asking to be believed on
trust. This keeps the receipts: every morning we forecast, what we said, and
what actually happened.

Two things have to agree on what "we scored this morning" means:

    calls       data/calls/calls_*.csv, the 8pm call, frozen when made
    observed    which district-mornings we can vouch for, from outcomes.py

SCORED AGAINST THE FROZEN CALL, NOT A RECOMPUTED ONE. The prediction could be
re-derived from the stored forecast at any time, but then retraining the model
would rewrite the past: every previous call would shift, and the published
accuracy would improve without a single new observation. The call is written
once and scored exactly as written.

SCORED ONLY WHERE WE ACTUALLY OBSERVED. Absence of a closure is not evidence of
an open morning unless somebody was watching that district that day, and the
two ways of getting this wrong are not symmetrical.

The first was already handled: an empty outcomes file for 12 January could mean
nobody closed OR the job never ran, and scoring the second as "we correctly
predicted open for 50 districts" would be flattering nonsense. The checks log
fixed that, per date.

The second was not, and it is per DISTRICT. The WBNG viewing area is not the
SCRIC region, so there are districts the feed will never carry. Reading their
silence as "open" awarded this tool a free correct call for every uncovered
district on every morning of the season, and scored a real closure at one of
them as a miss. `outcomes.observed()` is now the gate, and it answers per
district-morning rather than per date.

The 05:00 capture is compared separately, to measure what the extra nine hours
are worth. It never contributes to the published score, because it is not what
anyone read at bedtime.
"""

import os

import numpy as np
import pandas as pd

import outcomes as O
import predict as P
import schoolday as SD

ROOT = os.path.join(os.path.dirname(__file__), "..")
FORECAST_DIR = os.path.join(ROOT, "data", "forecasts")
CALLS_DIR = os.path.join(ROOT, "data", "calls")
OUTCOME_DIR = os.path.join(ROOT, "data", "outcomes")
CHECKS_CSV = os.path.join(OUTCOME_DIR, "checks.csv")
SCORECARD_CSV = os.path.join(ROOT, "data", "scorecard.csv")

EVENING = "evening"


def _read_all(directory, prefix):
    if not os.path.isdir(directory):
        return pd.DataFrame()
    frames = [pd.read_csv(os.path.join(directory, f), dtype={"leaid": str})
              for f in sorted(os.listdir(directory))
              if f.startswith(prefix) and f.endswith(".csv")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_inputs():
    """The frozen calls, the resolved outcomes, and what we can vouch for.

    Outcomes arrive already merged across manual, calendar and news, so the
    scorecard never has to know there is more than one source. Precedence lives
    in outcomes.py, which is also where the coverage question is answered.
    """
    calls = _read_all(CALLS_DIR, "calls_")
    stacked = O.load_all()
    return calls, O.resolve(stacked), O.observed(stacked=stacked)


def build(calls, outcomes, observed, roster=None, model=None):
    """One row per district-morning we both called and observed.

    `calls` is the frozen record, so nothing is predicted here. roster and
    model are accepted only so callers need not care about that.

    `observed` is a set of (date, leaid). It is a required argument and not a
    defaulted one on purpose: this is the gate that keeps the published record
    honest, and the version of this function that inferred it was the version
    that got it wrong for a year. A caller that has nothing to pass should pass
    an empty set and score nothing.
    """
    if calls is None or calls.empty or not observed:
        return pd.DataFrame()

    keys = list(zip(calls["target_date"].astype(str), calls["leaid"]))
    pred = calls[[k in observed for k in keys]].copy()
    if pred.empty:
        return pd.DataFrame()

    # Never score a day school was out anyway. A call should not exist for one,
    # but if a stray got through, scoring it would hand us a free correct
    # "open" for a morning nobody was ever going to school.
    calendar = SD.load()
    in_session = [SD.reason_off(pd.Timestamp(d).date(), calendar) is None
                  for d in pred["target_date"]]
    pred = pred[in_session]
    if pred.empty:
        return pd.DataFrame()

    # SCORE EACH HEAD AGAINST ITS OWN OUTCOME.
    #
    # This used to compare p_closed, which is trained on full closures only,
    # against closed-OR-delay-OR-remote. Every delay therefore counted as a
    # closure the model had failed to predict. Delays are about 23% of all
    # disruptions in the history, so roughly a quarter of the "misses" on the
    # published record would have been events p_closed was never asked about,
    # and the calibration table would have shown the model as under-confident
    # everywhere for the same reason.
    #
    # The model has two heads precisely so this distinction can be made. It has
    # to be made here too.
    #
    # "early" is not a disruption for our purposes: an early dismissal is a
    # mid-morning decision, and what gets scored is the overnight call.
    closed, disrupted = set(), set()
    if not outcomes.empty and "status" in outcomes.columns:
        shut = outcomes[outcomes["status"] == "closed"]
        closed = set(zip(shut["date"].astype(str), shut["leaid"]))
        any_shut = outcomes[outcomes["status"].isin(["closed", "delay", "remote"])]
        disrupted = set(zip(any_shut["date"].astype(str), any_shut["leaid"]))

    keys = list(zip(pred["target_date"].astype(str), pred["leaid"]))
    pred["actual_closed"] = [k in closed for k in keys]
    pred["actual_disrupted"] = [k in disrupted for k in keys]
    pred["predicted_closed"] = pred["p_closed"] >= 0.5
    pred["correct"] = pred["predicted_closed"] == pred["actual_closed"]
    if "p_disrupted" in pred.columns:
        pred["predicted_disrupted"] = pred["p_disrupted"] >= 0.5
        pred["correct_disrupted"] = (pred["predicted_disrupted"]
                                     == pred["actual_disrupted"])

    cols = [c for c in ("target_date", "leaid", "district", "p_closed",
                        "p_disrupted", "snow_in", "predicted_closed",
                        "actual_closed", "correct", "predicted_disrupted",
                        "actual_disrupted", "correct_disrupted")
            if c in pred.columns]
    sort_by = [c for c in ("target_date", "district") if c in pred.columns]
    return pred[cols].sort_values(sort_by)


def summarize(scorecard):
    """Headline numbers, phrased the way they will be shown."""
    if scorecard.empty:
        return {"scored_days": 0, "district_mornings": 0}

    y = scorecard["actual_closed"].astype(int).to_numpy()
    p = scorecard["p_closed"].to_numpy()

    # Brier is the honest headline for a probability. Plain accuracy looks
    # spectacular on a 3% base rate purely by always guessing "open", so it is
    # reported alongside the base rate that makes it meaningful.
    brier = float(np.mean((p - y) ** 2))
    always_open = float(np.mean((0.0 - y) ** 2))

    caught = int(((p >= 0.5) & (y == 1)).sum())
    missed = int(((p < 0.5) & (y == 1)).sum())
    false_alarms = int(((p >= 0.5) & (y == 0)).sum())

    out = {
        "scored_days": int(scorecard["target_date"].nunique()),
        "district_mornings": int(len(scorecard)),
        "closures_seen": int(y.sum()),
        "brier": round(brier, 4),
        "brier_if_we_always_said_open": round(always_open, 4),
        "skill_vs_always_open": (round(1 - brier / always_open, 3)
                                 if always_open > 0 else None),
        "closures_called": caught,
        "closures_missed": missed,
        "false_alarms": false_alarms,
    }

    # The delay-or-worse head, scored against delay-or-worse. Reported next to
    # the closure numbers rather than folded into them, because a district that
    # ran two hours late is not a district that closed, and averaging the two
    # questions together is what the old scorecard did wrong.
    if "actual_disrupted" in scorecard.columns and "p_disrupted" in scorecard.columns:
        yd = scorecard["actual_disrupted"].astype(int).to_numpy()
        pd_ = scorecard["p_disrupted"].to_numpy()
        out["disruptions_seen"] = int(yd.sum())
        out["brier_disrupted"] = round(float(np.mean((pd_ - yd) ** 2)), 4)
        out["disruptions_called"] = int(((pd_ >= 0.5) & (yd == 1)).sum())
        out["disruptions_missed"] = int(((pd_ < 0.5) & (yd == 1)).sum())

    return out


def calibration(scorecard, bins=(0, .05, .15, .3, .5, .75, 1.0)):
    """Did 70% mean 70%? The table to put on the page."""
    if scorecard.empty:
        return pd.DataFrame()
    band = pd.cut(scorecard["p_closed"], list(bins), include_lowest=True)
    out = (scorecard.assign(band=band)
           .groupby("band", observed=True)
           .agg(mornings=("actual_closed", "size"),
                we_said=("p_closed", "mean"),
                actually_closed=("actual_closed", "mean"))
           .reset_index())
    out["we_said"] = (out["we_said"] * 100).round(1)
    out["actually_closed"] = (out["actually_closed"] * 100).round(1)
    return out


def lead_time_value(calls, forecasts, outcomes, observed, roster, model=None):
    """What the extra nine hours buy: the 8pm call versus the 05:00 forecast.

    This one DOES recompute, from the morning capture, because no call was ever
    published at 5am. It is a diagnostic and never feeds the published score.
    """
    card = build(calls, outcomes, observed, roster)
    if card.empty or forecasts is None or forecasts.empty:
        return None
    if "capture_slot" not in forecasts.columns:
        return None

    mo = forecasts[forecasts["capture_slot"] == "morning"]
    if mo.empty:
        return None

    scored = []
    for date, group in mo.groupby("target_date"):
        out = P.predict(group, roster, model=model)
        out["target_date"] = date
        scored.append(out)
    morning = pd.concat(scored, ignore_index=True)

    both = card.merge(morning[["target_date", "leaid", "p_closed"]],
                      on=["target_date", "leaid"], suffixes=("_call", "_5am"))
    if both.empty:
        return None

    y = both["actual_closed"].astype(int).to_numpy()
    return {
        "paired_mornings": int(len(both)),
        "call_8pm_brier": round(float(np.mean((both["p_closed_call"] - y) ** 2)), 4),
        "forecast_5am_brier": round(float(np.mean((both["p_closed_5am"] - y) ** 2)), 4),
    }


def refresh(roster=None):
    """Rebuild the scorecard from everything on disk."""
    if roster is None:
        roster = pd.read_csv(os.path.join(ROOT, "reference", "roster.csv"),
                             dtype={"leaid": str})
    calls, outcomes, observed = load_inputs()
    card = build(calls, outcomes, observed, roster)
    if not card.empty:
        card.to_csv(SCORECARD_CSV, index=False)
    return card, summarize(card)


if __name__ == "__main__":
    card, summary = refresh()
    for k, v in summary.items():
        print(f"{k}: {v}")
    if card.empty:
        print("\nNothing scored yet. This fills in once forecasts and observed "
              "mornings overlap, which starts in December.")
    else:
        print(f"\nwrote {SCORECARD_CSV}")
        print()
        print(calibration(card).to_string(index=False))
