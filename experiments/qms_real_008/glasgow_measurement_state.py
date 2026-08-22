from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SMOOTH_WINDOW = 3


def moving_average(x, n=SMOOTH_WINDOW):
    x = np.asarray(x, dtype=float)

    result = np.empty_like(x)

    for i in range(len(x)):
        lo = max(0, i - n + 1)
        result[i] = np.mean(x[lo:i + 1])

    return result


def classify(dc, dj):
    if dc > 0 and dj < 0:
        return "converging"

    if dc < 0 and dj > 0:
        return "drifting"

    return "stable_or_mixed"


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

    rows = raw["windows"]

    cosine = np.asarray(
        [r["cosine_to_final"] for r in rows],
        dtype=float,
    )

    js = np.asarray(
        [r["js_divergence_to_final"] for r in rows],
        dtype=float,
    )

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

    for i, row in enumerate(rows):
        state = (
            "initial"
            if i == 0
            else classify(
                dc[i],
                dj[i],
            )
        )

        result = {
            "window_index":
                row["window_index"],
            "frame_start":
                row["frame_start"],
            "frame_end":
                row["frame_end"],
            "n_events":
                row["n_events"],

            "cosine_to_final":
                row["cosine_to_final"],
            "js_divergence_to_final":
                row["js_divergence_to_final"],

            "smoothed_cosine":
                float(cosine_smooth[i]),
            "smoothed_js":
                float(js_smooth[i]),

            "delta_smoothed_cosine":
                float(dc[i]),
            "delta_smoothed_js":
                float(dj[i]),

            "effective_dimension":
                row["effective_dimension"],
            "first_pc_fraction":
                row["first_pc_fraction"],

            "state":
                state,
        }

        results.append(result)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    counts = {}

    for row in results:
        counts[row["state"]] = (
            counts.get(
                row["state"],
                0,
            )
            + 1
        )

    non_initial = [
        r for r in results
        if r["state"] != "initial"
    ]

    first_drift = next(
        (
            r
            for r in non_initial
            if r["state"] == "drifting"
        ),
        None,
    )

    best_js = min(
        results,
        key=lambda r:
            r["js_divergence_to_final"],
    )

    best_cosine = max(
        results,
        key=lambda r:
            r["cosine_to_final"],
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

    summary = {
        "classification_rule": {
            "converging":
                "delta cosine > 0 and delta JS < 0",
            "drifting":
                "delta cosine < 0 and delta JS > 0",
            "stable_or_mixed":
                "otherwise",
            "smoothing_windows":
                SMOOTH_WINDOW,
        },

        "state_counts":
            counts,

        "minimum_js_window": {
            "window_index":
                best_js["window_index"],
            "frame_start":
                best_js["frame_start"],
            "frame_end":
                best_js["frame_end"],
            "js":
                best_js[
                    "js_divergence_to_final"
                ],
        },

        "maximum_cosine_window": {
            "window_index":
                best_cosine["window_index"],
            "frame_start":
                best_cosine["frame_start"],
            "frame_end":
                best_cosine["frame_end"],
            "cosine":
                best_cosine[
                    "cosine_to_final"
                ],
        },

        "first_detected_drift":
            (
                {
                    "window_index":
                        first_drift[
                            "window_index"
                        ],
                    "frame_start":
                        first_drift[
                            "frame_start"
                        ],
                    "frame_end":
                        first_drift[
                            "frame_end"
                        ],
                }
                if first_drift
                else None
            ),

        "state_transitions":
            transitions,

        "caution": (
            "State labels are operational heuristics based on "
            "agreement between two independent distribution-convergence "
            "indicators. They are not calibrated hardware-failure labels."
        ),
    }

    output = {
        "dataset":
            raw["dataset"],
        "experiment":
            "QMS-REAL-008",
        "windows":
            results,
        "summary":
            summary,
    }

    outdir = Path(
        "experiments/qms_real_008/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_008_state_classification.json"
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
