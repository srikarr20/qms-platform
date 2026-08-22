from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SMOOTH_WINDOW = 3


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

            "state":
                state,
        })

    counts = {}

    for row in results:
        state = row["state"]

        counts[state] = (
            counts.get(state, 0) + 1
        )

    transitions = []

    previous = None

    for row in results:

        if row["state"] == "initial":
            continue

        if previous is None:
            previous = row["state"]
            continue

        if row["state"] != previous:
            transitions.append({
                "window_index":
                    row["window_index"],

                "frame_start":
                    row["frame_start"],

                "from":
                    previous,

                "to":
                    row["state"],
            })

        previous = row["state"]

    first_drift = next(
        (
            r for r in results
            if r["state"] == "drifting"
        ),
        None,
    )

    summary = {
        "smoothing_windows":
            SMOOTH_WINDOW,

        "classification_rule": {
            "converging":
                "delta cosine > 0 and delta JS < 0",

            "drifting":
                "delta cosine < 0 and delta JS > 0",

            "stable_or_mixed":
                "otherwise",
        },

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

        "interpretation": (
            "The state-classification rule is transferred "
            "unchanged from QMS-REAL-008. No thresholds or "
            "state definitions are fitted to this dataset."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-019",

        "dataset":
            raw["dataset"],

        "method":
            "unchanged state-classifier transfer",

        "windows":
            results,

        "summary":
            summary,
    }

    outdir = Path(
        "experiments/qms_real_019/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_019_state_transfer.json"
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
