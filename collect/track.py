"""Track how the prediction moves, and freeze the one we stand behind.

Two outputs, and the distinction between them is the whole design:

    data/calls/{season}.csv               THE 8PM CALL. Written once per night,
                                          never rewritten. This is what gets
                                          scored and what the page stands behind.

    data/prediction_history/{season}.csv  The line that moves. Appended hourly
                                          through the evening and small hours,
                                          only on nights where something is
                                          actually happening.

Why freeze the call rather than recompute it. The scorecard could re-derive an
evening prediction from the stored forecast any time, but then retraining the
model would silently rewrite history: last winter's calls would change, and the
published accuracy would improve without a single new observation. A record
that can be edited after the fact is not a record. So the call is written once,
stamped with the model that made it, and left alone.

The hourly history is deliberately thin -- probability and snowfall, not the
whole feature set -- and is written only when it would show something. On a
clear January Tuesday this writes nothing and commits nothing.
"""

import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "model"))

import nws  # noqa: E402
import predict as P  # noqa: E402
import schoolday as SD  # noqa: E402

ROOT = os.path.join(HERE, "..")
ROSTER_CSV = os.path.join(ROOT, "reference", "roster.csv")
CALLS_DIR = os.path.join(ROOT, "data", "calls")
HISTORY_DIR = os.path.join(ROOT, "data", "prediction_history")

# A night is worth recording if anything is happening anywhere in the region.
# Both tests, because a low-probability night with real snow on the way is
# exactly when watching the line move is interesting.
WORTH_RECORDING_P = 0.05
WORTH_RECORDING_SNOW_IN = 0.2

# The call is THE 8PM CALL, so it may only be made near 8pm. The hourly job
# passes --call on every run and this window decides which one counts; the
# "already made" guard then refuses the rest. Without the window a 4pm run
# would publish the call four hours early and the name would be a lie.
CALL_HOURS = (19, 21)


def season_of(date):
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def _season_path(directory, date, prefix):
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{prefix}_{season_of(date)}.csv")


def predictions(roster=None, target_date=None, slot=None, verbose=True):
    """Fetch every district and score it. Returns (frame, failures)."""
    if roster is None:
        roster = pd.read_csv(ROSTER_CSV, dtype={"leaid": str})

    fc, failures = nws.snapshot(roster=roster, target_date=target_date,
                                slot=slot, verbose=verbose)
    if fc.empty:
        return pd.DataFrame(), failures

    out = P.predict(fc, roster)
    out["target_date"] = fc["target_date"].iloc[0]
    out["capture_slot"] = fc["capture_slot"].iloc[0]
    out["captured_at"] = fc["captured_at"].iloc[0]
    return out, failures


def worth_recording(pred):
    if pred.empty:
        return False
    return bool((pred["p_closed"] >= WORTH_RECORDING_P).any()
                or (pred["snow_in"] >= WORTH_RECORDING_SNOW_IN).any())


def write_history(pred, history_dir=HISTORY_DIR):
    """Append a thin row per district. Quiet nights write nothing."""
    if not worth_recording(pred):
        return None

    path = _season_path(history_dir, pred["target_date"].iloc[0], "history")
    thin = pred[["captured_at", "target_date", "leaid", "p_closed",
                 "p_disrupted", "snow_in"]].copy()

    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"leaid": str})
        # Same capture re-run replaces itself rather than doubling up.
        key = set(zip(thin["captured_at"], thin["leaid"]))
        old = old[[(c, l) not in key
                   for c, l in zip(old["captured_at"], old["leaid"])]]
        thin = pd.concat([old, thin], ignore_index=True)

    thin.sort_values(["target_date", "captured_at", "leaid"]).to_csv(
        path, index=False)
    return path


def existing_call(target_date, calls_dir=CALLS_DIR):
    path = _season_path(calls_dir, target_date, "calls")
    if not os.path.exists(path):
        return set()
    old = pd.read_csv(path, dtype={"leaid": str})
    want = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    return set(old.loc[old["target_date"] == want, "leaid"])


def in_call_window(now=None):
    now = now or datetime.now(nws.LOCAL_TZ)
    return CALL_HOURS[0] <= now.hour <= CALL_HOURS[1]


def write_call(pred, model=None, calls_dir=CALLS_DIR, force=False, now=None):
    """Write the night's official call. Refuses to overwrite one that exists.

    force is for a genuine correction, not for a better forecast arriving
    later. Overwriting on a whim is how a published accuracy record quietly
    becomes fiction.
    """
    if pred.empty:
        return None, "no predictions"
    if not force and not in_call_window(now):
        return None, "outside the 8pm call window"

    # No call for a day school is out anyway. Publishing one would put a
    # closure probability on Christmas and then score it as if it meant
    # something.
    target_date = pd.Timestamp(pred["target_date"].iloc[0]).date()
    calendar = SD.load()
    why = SD.reason_off(target_date, calendar)
    if why and not force:
        return None, f"no school on {target_date} ({why}); no call made"

    target = pred["target_date"].iloc[0]
    already = existing_call(target, calls_dir)
    if already and not force:
        return None, f"call already made for {target} ({len(already)} districts)"

    model = model or P.load()
    row = pred[["target_date", "leaid", "district", "p_closed", "p_disrupted",
                "snow_in"]].copy()
    row["called_at"] = datetime.now(nws.LOCAL_TZ).isoformat(timespec="seconds")
    row["model_trained_at"] = model.get("trained_at")

    path = _season_path(calls_dir, target, "calls")
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"leaid": str})
        old = old[old["target_date"] != pd.Timestamp(target).strftime("%Y-%m-%d")]
        row = pd.concat([old, row], ignore_index=True)

    row.sort_values(["target_date", "district"]).to_csv(path, index=False)
    return path, "written"


def load_call(target_date, calls_dir=CALLS_DIR):
    path = _season_path(calls_dir, target_date, "calls")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"leaid": str})
    want = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    return df[df["target_date"] == want].copy()


def load_history(target_date, history_dir=HISTORY_DIR):
    path = _season_path(history_dir, target_date, "history")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"leaid": str})
    want = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    df = df[df["target_date"] == want].copy()
    df["captured_at"] = pd.to_datetime(df["captured_at"], format="mixed",
                                       utc=True)
    return df.sort_values("captured_at")


if __name__ == "__main__":
    make_call = "--call" in sys.argv
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = pd.Timestamp(args[0]).date() if args else None

    pred, failures = predictions(target_date=target)
    if pred.empty:
        print("nothing captured; nothing written")
        raise SystemExit(0)

    top = pred.nlargest(3, "p_closed")[["district", "p_closed", "snow_in"]]
    print(f"\nhighest {len(top)}:")
    for _, r in top.iterrows():
        print(f"  {r['district'][:44]:46} {r['p_closed'] * 100:5.1f}%  "
              f"{r['snow_in']:.1f} in")

    path = write_history(pred)
    print(f"\nhistory: {'wrote ' + path if path else 'quiet night, not recorded'}")

    if make_call:
        path, why = write_call(pred, force=force)
        print(f"call: {why}" + (f" -> {path}" if path else ""))
