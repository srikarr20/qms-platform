from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


WINDOW_SIZES = [50, 100, 200]
SMOOTHING = [2, 3, 5]
GRID = 32
EPS = 1e-15


def frame_from_zip(zf, name):
    return np.loadtxt(
        io.BytesIO(zf.read(name))
    )


def event_coordinates(frame):
    y, x = np.nonzero(frame > 0)

    if len(x) == 0:
        return np.empty((0, 2), dtype=float)

    return np.column_stack([
        x.astype(float),
        y.astype(float),
    ])


def histogram_from_events(X):
    hist, _, _ = np.histogram2d(
        X[:, 1],
        X[:, 0],
        bins=GRID,
        range=[
            [0, 512],
            [0, 512],
        ],
    )

    p = hist.ravel().astype(float)

    if np.sum(p) > 0:
        p /= np.sum(p)

    return p


def cosine_similarity(p, q):
    denom = (
        np.linalg.norm(p)
        * np.linalg.norm(q)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(p, q) / denom
    )


def js_divergence(p, q):
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS

    p /= np.sum(p)
    q /= np.sum(q)

    m = 0.5 * (p + q)

    return float(
        0.5 * np.sum(
            p * np.log(p / m)
        )
        +
        0.5 * np.sum(
            q * np.log(q / m)
        )
    )


def moving_average(x, n):
    x = np.asarray(x, dtype=float)

    out = np.empty_like(x)

    for i in range(len(x)):
        lo = max(0, i - n + 1)
        out[i] = np.mean(x[lo:i + 1])

    return out


def classify(dc, dj):
    if dc > 0 and dj < 0:
        return "converging"

    if dc < 0 and dj > 0:
        return "drifting"

    return "stable_or_mixed"


def analyze(clouds, window_size, smoothing):
    windows = []

    for start in range(
        0,
        len(clouds),
        window_size,
    ):
        selected = clouds[
            start:
            start + window_size
        ]

        valid = [
            x for x in selected
            if len(x)
        ]

        if not valid:
            continue

        X = np.vstack(valid)

        windows.append({
            "frame_start":
                start + 1,
            "frame_end":
                start + len(selected),
            "distribution":
                histogram_from_events(X),
        })

    final_cloud = np.vstack([
        x for x in clouds
        if len(x)
    ])

    final_distribution = (
        histogram_from_events(
            final_cloud
        )
    )

    cosine = np.asarray([
        cosine_similarity(
            w["distribution"],
            final_distribution,
        )
        for w in windows
    ])

    js = np.asarray([
        js_divergence(
            w["distribution"],
            final_distribution,
        )
        for w in windows
    ])

    cs = moving_average(
        cosine,
        smoothing,
    )

    js_s = moving_average(
        js,
        smoothing,
    )

    dc = np.diff(
        cs,
        prepend=cs[0],
    )

    dj = np.diff(
        js_s,
        prepend=js_s[0],
    )

    states = [
        "initial"
    ]

    for i in range(1, len(windows)):
        states.append(
            classify(
                dc[i],
                dj[i],
            )
        )

    first_drift = next(
        (
            i
            for i, s in enumerate(states)
            if s == "drifting"
        ),
        None,
    )

    max_cosine = int(
        np.argmax(cosine)
    )

    min_js = int(
        np.argmin(js)
    )

    return {
        "window_size":
            window_size,
        "smoothing":
            smoothing,

        "n_windows":
            len(windows),

        "first_drift_window":
            first_drift,

        "first_drift_frame":
            (
                windows[first_drift][
                    "frame_start"
                ]
                if first_drift is not None
                else None
            ),

        "maximum_cosine_frame":
            windows[max_cosine][
                "frame_start"
            ],

        "minimum_js_frame":
            windows[min_js][
                "frame_start"
            ],

        "state_counts": {
            state: states.count(state)
            for state in sorted(
                set(states)
            )
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith(".asc")
        )

        clouds = []

        for name in names:
            frame = frame_from_zip(
                zf,
                name,
            )

            clouds.append(
                event_coordinates(
                    frame
                )
            )

    results = []

    for window_size in WINDOW_SIZES:
        for smoothing in SMOOTHING:

            result = analyze(
                clouds,
                window_size,
                smoothing,
            )

            results.append(result)

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

    drift_frames = [
        r["first_drift_frame"]
        for r in results
        if r["first_drift_frame"]
        is not None
    ]

    summary = {
        "configurations":
            len(results),

        "detected_drift_configurations":
            len(drift_frames),

        "first_drift_frame_mean":
            float(
                np.mean(drift_frames)
            ),

        "first_drift_frame_std":
            float(
                np.std(drift_frames)
            ),

        "first_drift_frame_min":
            int(
                np.min(drift_frames)
            ),

        "first_drift_frame_max":
            int(
                np.max(drift_frames)
            ),

        "interpretation": (
            "Robustness is supported if drift onset remains "
            "in approximately the same acquisition region across "
            "window-size and smoothing choices."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-009",
        "dataset":
            "Glasgow Heralded Diffraction SM",
        "results":
            results,
        "summary":
            summary,
    }

    outdir = Path(
        "experiments/qms_real_009/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_009_state_robustness.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("=== SUMMARY ===")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
