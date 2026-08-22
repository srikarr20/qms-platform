from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


WINDOW_FRAMES = 100
BASELINE_WINDOWS = 5
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
        range=[[0, 512], [0, 512]],
    )

    p = hist.ravel().astype(float)

    if np.sum(p) > 0:
        p /= np.sum(p)

    return p


def cosine_similarity(p, q):
    denom = np.linalg.norm(p) * np.linalg.norm(q)

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
        0.5 * np.sum(p * np.log(p / m))
        + 0.5 * np.sum(q * np.log(q / m))
    )


def robust_z(value, baseline):
    baseline = np.asarray(baseline, dtype=float)

    median = np.median(baseline)
    mad = np.median(
        np.abs(baseline - median)
    )

    scale = 1.4826 * mad

    if scale <= EPS:
        return 0.0

    return float(
        (value - median) / scale
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
            n for n in zf.namelist()
            if n.lower().endswith(".asc")
        )

        windows = []

        for start in range(
            0,
            len(names),
            WINDOW_FRAMES,
        ):
            selected = names[
                start:start + WINDOW_FRAMES
            ]

            clouds = []

            for name in selected:
                frame = frame_from_zip(
                    zf,
                    name,
                )

                coords = event_coordinates(frame)

                if len(coords):
                    clouds.append(coords)

            if not clouds:
                continue

            X = np.vstack(clouds)

            windows.append({
                "window_index": len(windows),
                "frame_start": start + 1,
                "frame_end": start + len(selected),
                "distribution":
                    histogram_from_events(X),
            })

    baseline_distribution = np.mean(
        [
            windows[i]["distribution"]
            for i in range(BASELINE_WINDOWS)
        ],
        axis=0,
    )

    baseline_js = []

    for i in range(BASELINE_WINDOWS):
        baseline_js.append(
            js_divergence(
                windows[i]["distribution"],
                baseline_distribution,
            )
        )

    rows = []

    for i, row in enumerate(windows):
        js = js_divergence(
            row["distribution"],
            baseline_distribution,
        )

        cosine = cosine_similarity(
            row["distribution"],
            baseline_distribution,
        )

        z = robust_z(
            js,
            baseline_js,
        )

        if i < BASELINE_WINDOWS:
            state = "baseline"
        elif z >= 5.0:
            state = "drift_alert"
        elif z >= 3.0:
            state = "warning"
        else:
            state = "nominal"

        result = {
            "window_index":
                row["window_index"],
            "frame_start":
                row["frame_start"],
            "frame_end":
                row["frame_end"],
            "cosine_to_baseline":
                cosine,
            "js_to_baseline":
                js,
            "robust_js_z":
                z,
            "state":
                state,
        }

        rows.append(result)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    warnings = [
        r for r in rows
        if r["state"] == "warning"
    ]

    alerts = [
        r for r in rows
        if r["state"] == "drift_alert"
    ]

    summary = {
        "baseline_windows":
            BASELINE_WINDOWS,
        "baseline_frame_end":
            windows[
                BASELINE_WINDOWS - 1
            ]["frame_end"],

        "n_windows":
            len(rows),

        "n_warning_windows":
            len(warnings),

        "n_alert_windows":
            len(alerts),

        "first_warning":
            warnings[0]
            if warnings
            else None,

        "first_drift_alert":
            alerts[0]
            if alerts
            else None,

        "interpretation": (
            "Reference distribution is frozen using only "
            "the first five acquisition windows. "
            "All later comparisons are causal."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-013",
        "dataset":
            "Glasgow Heralded Diffraction SM",
        "method":
            "frozen-reference causal monitoring",
        "windows":
            rows,
        "summary":
            summary,
        "caution": (
            "The initial acquisition period may itself represent "
            "measurement convergence rather than a known-good "
            "stationary operating state. Therefore an alert "
            "indicates departure from the early reference, "
            "not necessarily hardware degradation."
        ),
    }

    outdir = Path(
        "experiments/qms_real_013/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_013_frozen_reference.json"
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
