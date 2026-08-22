from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROLLING_SLOPE_WINDOWS = 4
WARNING_PERSISTENCE = 2
ALERT_PERSISTENCE = 3

SLOPE_Z_WARNING = 2.0
SLOPE_Z_ALERT = 3.0

EPS = 1e-15


def slope(y):
    y = np.asarray(y, dtype=float)

    if len(y) < 2:
        return 0.0

    x = np.arange(len(y), dtype=float)

    return float(
        np.polyfit(x, y, 1)[0]
    )


def robust_z(value, baseline):
    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    median = np.median(baseline)

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
        "--input-json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    raw = json.loads(
        args.input_json.read_text()
    )

    rows = [
        r for r in raw["windows"]
        if r.get("phase")
        == "locked_monitoring"
    ]

    js_values = np.asarray(
        [
            r["js_to_locked_reference"]
            for r in rows
        ],
        dtype=float,
    )

    rolling_slopes = []

    for i in range(
        ROLLING_SLOPE_WINDOWS - 1,
        len(js_values),
    ):
        segment = js_values[
            i - ROLLING_SLOPE_WINDOWS + 1:
            i + 1
        ]

        rolling_slopes.append({
            "row_index":
                i,
            "window_index":
                rows[i]["window_index"],
            "frame_start":
                rows[i]["frame_start"],
            "frame_end":
                rows[i]["frame_end"],
            "js_to_locked_reference":
                float(js_values[i]),
            "js_slope":
                slope(segment),
        })

    # Build slope baseline from earliest available
    # post-lock slope estimates only.
    baseline_count = min(
        5,
        len(rolling_slopes),
    )

    baseline_slopes = [
        r["js_slope"]
        for r in rolling_slopes[
            :baseline_count
        ]
    ]

    warning_run = 0
    alert_run = 0

    results = []

    first_warning = None
    first_alert = None

    for i, row in enumerate(
        rolling_slopes
    ):
        z = robust_z(
            row["js_slope"],
            baseline_slopes,
        )

        if i < baseline_count:
            state = "slope_baseline"
            warning_run = 0
            alert_run = 0

        else:
            if z >= SLOPE_Z_WARNING:
                warning_run += 1
            else:
                warning_run = 0

            if z >= SLOPE_Z_ALERT:
                alert_run += 1
            else:
                alert_run = 0

            if (
                alert_run
                >= ALERT_PERSISTENCE
            ):
                state = "drift_acceleration_alert"

            elif (
                warning_run
                >= WARNING_PERSISTENCE
            ):
                state = "acceleration_warning"

            else:
                state = "nominal"

        result = {
            **row,
            "slope_robust_z":
                float(z),
            "warning_run":
                warning_run,
            "alert_run":
                alert_run,
            "state":
                state,
        }

        results.append(result)

        if (
            state
            == "acceleration_warning"
            and first_warning is None
        ):
            first_warning = {
                "window_index":
                    row["window_index"],
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "js_slope":
                    row["js_slope"],
                "slope_robust_z":
                    float(z),
            }

        if (
            state
            == "drift_acceleration_alert"
            and first_alert is None
        ):
            first_alert = {
                "window_index":
                    row["window_index"],
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "js_slope":
                    row["js_slope"],
                "slope_robust_z":
                    float(z),
            }

    summary = {
        "rolling_slope_windows":
            ROLLING_SLOPE_WINDOWS,

        "slope_baseline_count":
            baseline_count,

        "slope_z_warning":
            SLOPE_Z_WARNING,

        "slope_z_alert":
            SLOPE_Z_ALERT,

        "warning_persistence":
            WARNING_PERSISTENCE,

        "alert_persistence":
            ALERT_PERSISTENCE,

        "first_acceleration_warning":
            first_warning,

        "first_drift_acceleration_alert":
            first_alert,

        "comparison_reference": {
            "retrospective_robust_drift_frame":
                3034,
            "note":
                (
                    "Used only for post-hoc comparison. "
                    "It is not used by the causal detector."
                ),
        },

        "interpretation": (
            "This experiment monitors the rate of change "
            "of JS divergence after baseline lock. "
            "It tests for acceleration in departure rather "
            "than absolute distance from the baseline."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-016",
        "dataset":
            raw["dataset"],
        "method":
            "causal drift-acceleration monitoring",
        "results":
            results,
        "summary":
            summary,
        "caution": (
            "Slope thresholds and persistence parameters "
            "are provisional. This remains a single-sequence "
            "exploratory validation."
        ),
    }

    outdir = Path(
        "experiments/qms_real_016/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_016_drift_acceleration.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

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
