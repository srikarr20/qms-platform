from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


WINDOW_FRAMES = 100
GRID = 32
EPS = 1e-15

ROLLING_REFERENCE_WINDOWS = 5
STABILITY_CONFIRM_WINDOWS = 4

# Provisional thresholds.
MAX_JS_TO_ROLLING = 0.004
MIN_COSINE_TO_ROLLING = 0.995


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
        range=[[0, 512], [0, 512]],
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


def robust_z(value, baseline):
    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    median = np.median(
        baseline
    )

    mad = np.median(
        np.abs(
            baseline - median
        )
    )

    scale = 1.4826 * mad

    if scale <= EPS:
        return 0.0

    return float(
        (value - median)
        / scale
    )


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

        windows = []

        for start in range(
            0,
            len(names),
            WINDOW_FRAMES,
        ):
            selected = names[
                start:
                start + WINDOW_FRAMES
            ]

            clouds = []

            for name in selected:
                frame = frame_from_zip(
                    zf,
                    name,
                )

                coords = event_coordinates(
                    frame
                )

                if len(coords):
                    clouds.append(coords)

            if not clouds:
                continue

            X = np.vstack(clouds)

            windows.append({
                "window_index":
                    len(windows),
                "frame_start":
                    start + 1,
                "frame_end":
                    start + len(selected),
                "distribution":
                    histogram_from_events(X),
            })

    rows = []

    stable_run = 0
    lock_index = None
    locked_reference = None
    lock_js_baseline = []

    for i, row in enumerate(windows):

        if i < ROLLING_REFERENCE_WINDOWS:
            rows.append({
                "window_index":
                    i,
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "phase":
                    "converging",
                "state":
                    "reference_building",
            })
            continue

        if lock_index is None:

            reference = np.mean(
                [
                    windows[j][
                        "distribution"
                    ]
                    for j in range(
                        i - ROLLING_REFERENCE_WINDOWS,
                        i,
                    )
                ],
                axis=0,
            )

            current = row[
                "distribution"
            ]

            cosine = cosine_similarity(
                current,
                reference,
            )

            js = js_divergence(
                current,
                reference,
            )

            stable_now = (
                cosine >= MIN_COSINE_TO_ROLLING
                and
                js <= MAX_JS_TO_ROLLING
            )

            if stable_now:
                stable_run += 1
            else:
                stable_run = 0

            result = {
                "window_index":
                    i,
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "phase":
                    "convergence_monitoring",
                "cosine_to_rolling_reference":
                    cosine,
                "js_to_rolling_reference":
                    js,
                "stable_run":
                    stable_run,
                "state":
                    (
                        "stability_candidate"
                        if stable_now
                        else "converging"
                    ),
            }

            rows.append(result)

            if (
                stable_run
                >= STABILITY_CONFIRM_WINDOWS
            ):
                lock_index = i

                start = (
                    i
                    - STABILITY_CONFIRM_WINDOWS
                    + 1
                )

                locked_reference = np.mean(
                    [
                        windows[j][
                            "distribution"
                        ]
                        for j in range(
                            start,
                            i + 1,
                        )
                    ],
                    axis=0,
                )

                lock_js_baseline = [
                    js_divergence(
                        windows[j][
                            "distribution"
                        ],
                        locked_reference,
                    )
                    for j in range(
                        start,
                        i + 1,
                    )
                ]

                result[
                    "state"
                ] = "baseline_locked"

                result[
                    "baseline_lock_frame"
                ] = row["frame_end"]

            continue

        current = row[
            "distribution"
        ]

        cosine = cosine_similarity(
            current,
            locked_reference,
        )

        js = js_divergence(
            current,
            locked_reference,
        )

        z = robust_z(
            js,
            lock_js_baseline,
        )

        if z >= 5.0:
            state = "drift_alert"
        elif z >= 3.0:
            state = "warning"
        else:
            state = "nominal"

        rows.append({
            "window_index":
                i,
            "frame_start":
                row["frame_start"],
            "frame_end":
                row["frame_end"],
            "phase":
                "locked_monitoring",
            "cosine_to_locked_reference":
                cosine,
            "js_to_locked_reference":
                js,
            "robust_js_z":
                z,
            "state":
                state,
        })

    warnings = [
        r
        for r in rows
        if r.get("state")
        == "warning"
    ]

    alerts = [
        r
        for r in rows
        if r.get("state")
        == "drift_alert"
    ]

    summary = {
        "rolling_reference_windows":
            ROLLING_REFERENCE_WINDOWS,

        "stability_confirm_windows":
            STABILITY_CONFIRM_WINDOWS,

        "max_js_to_rolling":
            MAX_JS_TO_ROLLING,

        "min_cosine_to_rolling":
            MIN_COSINE_TO_ROLLING,

        "baseline_lock_window":
            lock_index,

        "baseline_lock_frame":
            (
                windows[
                    lock_index
                ]["frame_end"]
                if lock_index
                is not None
                else None
            ),

        "n_warning_windows":
            len(warnings),

        "n_alert_windows":
            len(alerts),

        "first_warning":
            (
                warnings[0]
                if warnings
                else None
            ),

        "first_drift_alert":
            (
                alerts[0]
                if alerts
                else None
            ),

        "interpretation": (
            "Baseline is locked only after a sustained "
            "causally detected stable period. Later "
            "measurements are compared against that "
            "fixed reference."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-014",
        "dataset":
            "Glasgow Heralded Diffraction SM",
        "method":
            "causal stabilization detection followed by frozen-reference monitoring",
        "windows":
            rows,
        "summary":
            summary,
        "caution": (
            "Thresholds remain provisional and are "
            "evaluated on a single detector sequence."
        ),
    }

    outdir = Path(
        "experiments/qms_real_014/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_014_auto_baseline_lock.json"
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
