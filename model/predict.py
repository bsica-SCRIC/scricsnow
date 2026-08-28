"""Turn tonight's forecast into tomorrow's probability.

Loads model/artifacts/model.json and applies it to an NWS snapshot. No training
data is needed or present: the artifact carries the coefficients, the scaling,
and the imputation values, and nothing in it describes any district.

The heuristic stays available as a fallback and as a second opinion. It is not
dead weight: it can see three things the fitted model cannot, because GHCN
history has no column for any of them. Storm end hour, wind gusts, and the
exact midnight split of the overnight snowfall. When the two disagree sharply
on a night with a late-ending storm, the heuristic is the one with more
information.

WHAT ARRIVES HERE VERSUS WHAT THE MODEL WAS TRAINED ON. snow_total_mm is the
18:00-to-09:00 forecast window; in training it was GHCN SNOW[D], a calendar
day. Those are the same storm and not the same number of hours, which is the
largest single reason to trust the published scorecard over the training
metrics. features.py carries the full argument.
"""

import json
import os

import numpy as np
import pandas as pd

import features as F
import heuristic as H

ARTIFACT = os.path.join(os.path.dirname(__file__), "artifacts", "model.json")
OUTCOME_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outcomes")


def closure_history(directory=OUTCOME_DIR):
    """Who closed, and when, in the shape add_calendar_context expects.

    Built from data/outcomes, the public record the morning job writes, NOT
    from the internal closure history. Serving must never need internal data.

    This exists because the history features were silently dead in production.
    predict() took an optional `events` argument, none of the three callers
    passed it, and so closures_used was zero on every number the site has ever
    published while training rows averaged 1.36. Nothing raised. An optional
    argument that is wrong when omitted is a trap, so the loading now lives here
    and predict() uses it by default.

    Only full closures count. A delay is a different decision and the label the
    model is trained on is a closure.
    """
    if not os.path.isdir(directory):
        return pd.DataFrame(columns=["leaid", "date", "kind"])

    frames = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("outcomes_") and name.endswith(".csv")):
            continue
        try:
            frames.append(pd.read_csv(os.path.join(directory, name),
                                      dtype={"leaid": str}))
        except (OSError, ValueError):
            # A truncated outcome file should cost us one day of history, not
            # the whole evening's forecast.
            continue
    if not frames:
        return pd.DataFrame(columns=["leaid", "date", "kind"])

    out = pd.concat(frames, ignore_index=True)
    if "status" not in out.columns or "leaid" not in out.columns:
        return pd.DataFrame(columns=["leaid", "date", "kind"])

    out = out[out["status"] == "closed"].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].notna()]
    out["kind"] = "closed"
    return out[["leaid", "date", "kind"]].drop_duplicates()


def load(path=ARTIFACT):
    """Load the artifact and check it still matches the feature definition.

    The coefficient list is positional against FEATURES. If someone edits
    features.py without retraining, every coefficient silently shifts onto the
    wrong column: no error, no crash, just quietly wrong numbers. So the two
    are compared on every load.
    """
    with open(path, encoding="utf-8") as fh:
        model = json.load(fh)

    if model["features"] != F.FEATURES:
        raise ValueError(
            "model.json was trained on a different feature set than "
            "features.py now defines.\n"
            f"  artifact: {model['features']}\n"
            f"  current : {F.FEATURES}\n"
            "Coefficients are positional, so serving this would apply them to "
            "the wrong columns. Retrain before serving."
        )
    return model


def _apply(model, target, x):
    coef = np.asarray(model[target]["coef"], dtype=float)
    z = x @ coef + float(model[target]["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def from_features(feat, model=None):
    """Probabilities from a canonical feature frame."""
    model = load() if model is None else model

    x = F.matrix(feat, impute=model["impute"]).to_numpy(dtype=float)
    mean = np.asarray(model["scaler"]["mean"], dtype=float)
    scale = np.asarray(model["scaler"]["scale"], dtype=float)
    x = (x - mean) / np.where(scale == 0, 1.0, scale)

    p_closed = _apply(model, "closed", x)
    p_disrupted = np.maximum(_apply(model, "disrupted", x), p_closed)

    return pd.DataFrame({"p_closed": p_closed, "p_disrupted": p_disrupted},
                        index=feat.index)


def predict(forecast, roster, events=None, model=None, depth_mm=None):
    """Full path: NWS snapshot to a ranked table of districts.

    events is the closure history. Leave it out and it is LOADED FROM DISK, not
    treated as empty, because empty is a claim rather than a gap: it asserts
    that nobody has closed all season and nobody closed yesterday, which on the
    second morning of a long storm is precisely backwards and is the direction
    that inflates the published number. To score without any history, pass an
    explicit empty frame.

    depth_mm is optional standing snow depth per leaid. There is no forecast of
    it and GHCN observations run a few days behind, so it is frequently absent.
    Absent is left as MISSING rather than zero, so the imputation value stored
    at training time decides what an unknown depth is worth. On this training
    set the median depth is 0 mm, so the served number lands in the same place
    either way; the difference is that it will keep landing in the right place
    if the median ever moves, and that "we do not know" stops being asserted as
    "there is none". On the second morning of a long storm this is still a real
    omission, and passing depth_mm is worth it when a caller has the data.
    """
    # from_forecast merges and so returns a fresh RangeIndex. Reset here too,
    # otherwise the two frames align on different indexes further down.
    forecast = forecast.reset_index(drop=True)
    feat = F.from_forecast(forecast, roster)
    if depth_mm is not None:
        # No fillna: a leaid absent from the mapping is unknown, not snow-free.
        feat["depth_mm"] = feat["leaid"].map(depth_mm)
    if events is None:
        events = closure_history()
    feat = F.add_calendar_context(feat, events=events)

    model_p = from_features(feat, model=model)

    # The heuristic also gets the timing columns, which never reach the model
    # matrix because history has no hourly detail to train them on. Gusts are
    # already carried through by from_forecast, so pulling gust_max_kmh across
    # as well would produce two columns both named gust_kmh.
    extra = [c for c in ("snow_end_hour", "snow_ongoing_at_9am")
             if c in forecast.columns]
    heur_in = pd.concat([feat, forecast[extra]], axis=1)
    if "gust_kmh" not in heur_in.columns:
        heur_in["gust_kmh"] = 0.0
    assert heur_in.columns.is_unique, "duplicate feature columns"
    heur_p = H.predict(heur_in)

    out = pd.DataFrame({
        "date": feat["date"],
        "leaid": feat["leaid"],
        "p_closed": model_p["p_closed"],
        "p_disrupted": model_p["p_disrupted"],
        "p_closed_heuristic": heur_p["p_closed"],
        "snow_in": (feat["snow_total_mm"] / H.MM_PER_INCH).round(1),
    })
    out = out.merge(roster[["leaid", "district"]], on="leaid", how="left")
    out["why"] = [", ".join(H.explain(r)) for _, r in heur_in.iterrows()]
    return out.sort_values("p_closed", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    root = os.path.join(os.path.dirname(__file__), "..")
    roster = pd.read_csv(os.path.join(root, "reference", "roster.csv"),
                         dtype={"leaid": str})

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        fdir = os.path.join(root, "data", "forecasts")
        files = sorted(f for f in os.listdir(fdir) if f.endswith(".csv"))
        if not files:
            print("no forecast snapshots yet; run collect/nws.py")
            raise SystemExit(1)
        path = os.path.join(fdir, files[-1])

    fc = pd.read_csv(path, dtype={"leaid": str})
    latest = fc[fc["target_date"] == fc["target_date"].max()].copy()

    out = predict(latest, roster)
    print(f"forecast for {latest['target_date'].iloc[0]}  "
          f"({len(out)} districts)\n")
    show = out.head(12)[["district", "p_closed", "p_disrupted", "snow_in", "why"]]
    show = show.assign(
        p_closed=(show["p_closed"] * 100).round(1),
        p_disrupted=(show["p_disrupted"] * 100).round(1))
    print(show.to_string(index=False))
