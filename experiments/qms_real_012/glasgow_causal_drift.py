from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


WINDOW_FRAMES = 100
REFERENCE_WINDOWS = 5
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

    return float(np.dot(p, q) / denom)


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

    parser.add_argument(
        "--window-frames",
        type=int,
        default=WINDOW_FRAMES,
    )

    parser.add_argument(
        "--reference-windows",
        type=int,
        default=REFERENCE_WINDOWS,
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
            args.window_frames,
        ):
            selected = names[
                start:start + args.window_frames
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
                "window_index": len(windows),
                "frame_start": start + 1,
                "frame_end": start + len(selected),
                "n_events": len(X),
                "distribution":
                    histogram_from_events(X),
            })

    rows = []

    js_history = []

    for i, row in enumerate(windows):

        if i < args.reference_windows:
            result = {
                "window_index":
                    row["window_index"],
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "state":
                    "reference_building",
            }

            rows.append(result)
            continue

        reference = np.mean(
            [
                windows[j]["distribution"]
                for j in range(
                    i - args.reference_windows,
                    i,
                )
            ],
            axis=0,
        )

        current = row["distribution"]

        cosine = cosine_similarity(
            current,
            reference,
        )

        js = js_divergence(
            current,
            reference,
        )

        if len(js_history) >= 5:
            z_js = robust_z(
                js,
                js_history[-10:],
            )
        else:
            z_js = 0.0

        # Provisional, causal anomaly rule.
        # Requires substantial JS increase.
        if z_js >= 3.0:
            state = "drift_alert"
        elif z_js >= 2.0:
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
            "n_events":
                row["n_events"],
            "cosine_to_prior_reference":
                cosine,
            "js_to_prior_reference":
                js,
            "robust_js_z":
                z_js,
            "state":
                state,
        }

        rows.append(result)

        js_history.append(js)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    alerts = [
        r for r in rows
        if r.get("state")
        in {"warning", "drift_alert"}
    ]

    first_warning = next(
        (
            r for r in rows
            if r.get("state") == "warning"
        ),
        None,
    )

    first_alert = next(
        (
            r for r in rows
            if r.get("state") == "drift_alert"
        ),
        None,
    )

    summary = {
        "window_frames":
            args.window_frames,
        "reference_windows":
            args.reference_windows,
        "n_windows":
            len(windows),
        "n_warning_or_alert_windows":
            len(alerts),
        "first_warning":
            first_warning,
        "first_drift_alert":
            first_alert,
        "interpretation": (
            "All comparisons are causal: each window is compared "
            "only with earlier windows. No final-distribution "
            "reference is used."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-012",
        "dataset":
            "Glasgow Heralded Diffraction SM",
        "method":
            "causal rolling-reference distribution monitoring",
        "windows":
            rows,
        "summary":
            summary,
        "caution": (
            "Thresholds are provisional and have not yet been "
            "validated across independent detector datasets."
        ),
    }

    outdir = Path(
        "experiments/qms_real_012/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_012_causal_drift.json"
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
