from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LOOKBACKS = [3, 5, 8]


def slope(y):
    y = np.asarray(y, dtype=float)

    if len(y) < 2:
        return 0.0

    x = np.arange(len(y), dtype=float)

    return float(
        np.polyfit(x, y, 1)[0]
    )


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

    drift_index = next(
        i
        for i, row in enumerate(rows)
        if row["frame_start"] >= args.drift_frame
    )

    metrics = [
        "effective_dimension",
        "first_pc_fraction",
        "radial_std",
        "radial_p99",
        "cosine_to_final",
        "js_divergence_to_final",
    ]

    results = []

    for lookback in LOOKBACKS:
        start = max(
            0,
            drift_index - lookback,
        )

        subset = rows[
            start:
            drift_index
        ]

        metric_slopes = {}

        for metric in metrics:
            values = [
                row[metric]
                for row in subset
            ]

            metric_slopes[metric] = slope(
                values
            )

        result = {
            "lookback_windows":
                lookback,
            "window_start_index":
                start,
            "window_end_index":
                drift_index - 1,
            "frame_start":
                subset[0]["frame_start"],
            "frame_end":
                subset[-1]["frame_end"],
            "metric_slopes":
                metric_slopes,
        }

        results.append(result)

    rolling = []

    for i in range(5, len(rows)):
        subset = rows[i - 5:i]

        rolling.append({
            "window_index":
                rows[i]["window_index"],
            "frame_start":
                rows[i]["frame_start"],
            "effective_dimension_slope":
                slope([
                    r["effective_dimension"]
                    for r in subset
                ]),
            "first_pc_fraction_slope":
                slope([
                    r["first_pc_fraction"]
                    for r in subset
                ]),
            "cosine_slope":
                slope([
                    r["cosine_to_final"]
                    for r in subset
                ]),
            "js_slope":
                slope([
                    r["js_divergence_to_final"]
                    for r in subset
                ]),
        })

    output = {
        "experiment":
            "QMS-REAL-010",
        "dataset":
            raw["dataset"],
        "drift_reference_frame":
            args.drift_frame,
        "drift_reference_window":
            drift_index,
        "pre_drift_slopes":
            results,
        "rolling_five_window_slopes":
            rolling,
        "interpretation_note": (
            "This experiment tests whether representation and "
            "distribution metrics show systematic directional changes "
            "before the operational drift transition. It is exploratory "
            "and does not yet establish predictive lead time."
        ),
    }

    outdir = Path(
        "experiments/qms_real_010/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_010_early_warning.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("=== PRE-DRIFT SLOPES ===")

    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
