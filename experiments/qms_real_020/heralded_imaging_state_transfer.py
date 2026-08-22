from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


WINDOW_FRAMES = 100
GRID = 32
SMOOTH_WINDOW = 3
EPS = 1e-15


def frame_from_zip(zf, name):
    return np.loadtxt(io.BytesIO(zf.read(name)))


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


def moving_average(x, n=SMOOTH_WINDOW):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    windows = []

    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.lower().endswith(".asc")
        )

        all_clouds = []

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
                frame = frame_from_zip(zf, name)
                coords = event_coordinates(frame)

                if len(coords):
                    clouds.append(coords)

            if not clouds:
                continue

            X = np.vstack(clouds)
            all_clouds.append(X)

            windows.append({
                "window_index": len(windows),
                "frame_start": start + 1,
                "frame_end": start + len(selected),
                "distribution":
                    histogram_from_events(X),
            })

    final_distribution = histogram_from_events(
        np.vstack(all_clouds)
    )

    cosine = np.asarray([
        cosine_similarity(
            row["distribution"],
            final_distribution,
        )
        for row in windows
    ])

    js = np.asarray([
        js_divergence(
            row["distribution"],
            final_distribution,
        )
        for row in windows
    ])

    cosine_smooth = moving_average(cosine)
    js_smooth = moving_average(js)

    dc = np.diff(
        cosine_smooth,
        prepend=cosine_smooth[0],
    )

    dj = np.diff(
        js_smooth,
        prepend=js_smooth[0],
    )

    results = []

    for i, row in enumerate(windows):
        state = (
            "initial"
            if i == 0
            else classify(dc[i], dj[i])
        )

        results.append({
            "window_index":
                row["window_index"],

            "frame_start":
                row["frame_start"],

            "frame_end":
                row["frame_end"],

            "cosine_to_final":
                float(cosine[i]),

            "js_divergence_to_final":
                float(js[i]),

            "smoothed_cosine":
                float(cosine_smooth[i]),

            "smoothed_js":
                float(js_smooth[i]),

            "delta_smoothed_cosine":
                float(dc[i]),

            "delta_smoothed_js":
                float(dj[i]),

            "state":
                state,
        })

    counts = {}

    for row in results:
        counts[row["state"]] = (
            counts.get(row["state"], 0) + 1
        )

    transitions = []
    previous = None

    for row in results:
        state = row["state"]

        if state == "initial":
            continue

        if previous is None:
            previous = state
            continue

        if state != previous:
            transitions.append({
                "window_index":
                    row["window_index"],
                "frame_start":
                    row["frame_start"],
                "from":
                    previous,
                "to":
                    state,
            })

        previous = state

    first_drift = next(
        (
            row for row in results
            if row["state"] == "drifting"
        ),
        None,
    )

    summary = {
        "raw_frames":
            len(names),

        "n_windows":
            len(results),

        "window_frames":
            WINDOW_FRAMES,

        "smoothing_windows":
            SMOOTH_WINDOW,

        "state_counts":
            counts,

        "first_detected_drift":
            (
                {
                    "window_index":
                        first_drift["window_index"],

                    "frame_start":
                        first_drift["frame_start"],

                    "frame_end":
                        first_drift["frame_end"],
                }
                if first_drift
                else None
            ),

        "state_transitions":
            transitions,

        "maximum_cosine_window":
            int(np.argmax(cosine)),

        "minimum_js_window":
            int(np.argmin(js)),

        "cosine_start":
            float(cosine[0]),

        "cosine_end":
            float(cosine[-1]),

        "js_start":
            float(js[0]),

        "js_end":
            float(js[-1]),

        "interpretation": (
            "The same 100-frame convergence diagnostics, "
            "three-window smoothing, and state classification "
            "rule used for the diffraction datasets are applied "
            "unchanged to Heralded Imaging MM."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-020",

        "dataset":
            "Glasgow Heralded Imaging MM set 1",

        "source_level":
            "raw 512x512 ASC detector frames",

        "method":
            "unchanged cross-objective state-classifier transfer",

        "windows":
            results,

        "summary":
            summary,

        "caution": (
            "Only 1000 raw frames are available, producing "
            "10 windows at the fixed 100-frame resolution. "
            "State-transition conclusions are therefore less "
            "statistically supported than the 4070-frame "
            "diffraction acquisitions."
        ),
    }

    outdir = Path(
        "experiments/qms_real_020/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_020_heralded_imaging_mm.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
