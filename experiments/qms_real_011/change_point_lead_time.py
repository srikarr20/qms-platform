from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "effective_dimension",
    "first_pc_fraction",
    "radial_std",
    "radial_p99",
    "cosine_to_final",
    "js_divergence_to_final",
]


def piecewise_error(y, split):
    y = np.asarray(y, dtype=float)

    x1 = np.arange(split, dtype=float)
    x2 = np.arange(
        split,
        len(y),
        dtype=float,
    )

    y1 = y[:split]
    y2 = y[split:]

    p1 = np.polyfit(
        x1,
        y1,
        1,
    )

    p2 = np.polyfit(
        x2,
        y2,
        1,
    )

    pred1 = np.polyval(
        p1,
        x1,
    )

    pred2 = np.polyval(
        p2,
        x2,
    )

    error = (
        np.sum(
            (y1 - pred1) ** 2
        )
        +
        np.sum(
            (y2 - pred2) ** 2
        )
    )

    return float(error), p1, p2


def best_change_point(y):
    best = None

    # Require at least five windows
    # on each side.
    for split in range(
        5,
        len(y) - 5,
    ):
        error, p1, p2 = (
            piecewise_error(
                y,
                split,
            )
        )

        if (
            best is None
            or error < best["error"]
        ):
            best = {
                "split":
                    split,
                "error":
                    error,
                "slope_before":
                    float(p1[0]),
                "slope_after":
                    float(p2[0]),
            }

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--drift-frame",
        type=int,
        default=3034,
    )

    args = parser.parse_args()

    raw = json.loads(
        args.input_json.read_text()
    )

    rows = raw["windows"]

    results = {}

    for metric in METRICS:
        y = np.asarray(
            [
                row[metric]
                for row in rows
            ],
            dtype=float,
        )

        cp = best_change_point(y)

        split = cp["split"]

        frame = int(
            rows[split][
                "frame_start"
            ]
        )

        cp["change_frame"] = frame

        cp["lead_frames_vs_drift"] = int(
            args.drift_frame - frame
        )

        cp["before_mean"] = float(
            np.mean(
                y[:split]
            )
        )

        cp["after_mean"] = float(
            np.mean(
                y[split:]
            )
        )

        results[metric] = cp

    output = {
        "experiment":
            "QMS-REAL-011",
        "dataset":
            raw["dataset"],
        "drift_reference_frame":
            args.drift_frame,
        "method":
            (
                "Two-segment least-squares linear "
                "change-point search with at least "
                "five windows on each side."
            ),
        "metrics":
            results,
        "caution":
            (
                "Change points are retrospective. "
                "Positive lead time does not by itself "
                "establish prospective predictive capability."
            ),
    }

    outdir = Path(
        "experiments/qms_real_011/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_011_change_points.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
