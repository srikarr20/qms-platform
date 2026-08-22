from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def first_drift(rows):
    for row in rows:
        if row.get("state") == "drifting":
            return row
    return None


def state_counts(rows):
    counts = {}

    for row in rows:
        state = row.get("state")

        if state is None:
            continue

        counts[state] = counts.get(state, 0) + 1

    return counts


def summarize_dataset(
    name,
    family,
    raw_frames,
    rows,
):
    cosine = np.asarray(
        [
            float(r["cosine_to_final"])
            for r in rows
        ],
        dtype=float,
    )

    js = np.asarray(
        [
            float(r["js_divergence_to_final"])
            for r in rows
        ],
        dtype=float,
    )

    drift = first_drift(rows)

    max_cosine_i = int(
        np.argmax(cosine)
    )

    min_js_i = int(
        np.argmin(js)
    )

    max_cosine_window = int(
        rows[max_cosine_i]["window_index"]
    )

    min_js_window = int(
        rows[min_js_i]["window_index"]
    )

    if drift is not None:
        drift_midpoint = (
            float(drift["frame_start"])
            + float(drift["frame_end"])
        ) / 2.0

        normalized_drift_position = (
            drift_midpoint
            / float(raw_frames)
        )

        drift_window_index = int(
            drift["window_index"]
        )

    else:
        normalized_drift_position = None
        drift_window_index = None

    max_cosine_relative = (
        (
            float(rows[max_cosine_i]["frame_start"])
            + float(rows[max_cosine_i]["frame_end"])
        )
        / 2.0
        / float(raw_frames)
    )

    min_js_relative = (
        (
            float(rows[min_js_i]["frame_start"])
            + float(rows[min_js_i]["frame_end"])
        )
        / 2.0
        / float(raw_frames)
    )

    return {
        "dataset":
            name,

        "family":
            family,

        "raw_frames":
            raw_frames,

        "n_windows":
            len(rows),

        "state_counts":
            state_counts(rows),

        "cosine_start":
            float(cosine[0]),

        "cosine_end":
            float(cosine[-1]),

        "js_start":
            float(js[0]),

        "js_end":
            float(js[-1]),

        "maximum_cosine_window":
            max_cosine_window,

        "maximum_cosine_relative_position":
            max_cosine_relative,

        "minimum_js_window":
            min_js_window,

        "minimum_js_relative_position":
            min_js_relative,

        "first_drift_window":
            drift_window_index,

        "normalized_drift_position":
            normalized_drift_position,
    }


def mean_std(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(np.mean(values)),

        "std":
            float(np.std(values)),

        "min":
            float(np.min(values)),

        "max":
            float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--heralded-diffraction",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ghost-diffraction",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--heralded-imaging",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ghost-imaging",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    hd = load(
        args.heralded_diffraction
    )

    gd = load(
        args.ghost_diffraction
    )

    hi = load(
        args.heralded_imaging
    )

    gi = load(
        args.ghost_imaging
    )

    datasets = [
        summarize_dataset(
            "Heralded Diffraction SM",
            "diffraction",
            4070,
            hd["windows"],
        ),

        summarize_dataset(
            "Ghost Diffraction SM",
            "diffraction",
            4070,
            gd["windows"],
        ),

        summarize_dataset(
            "Heralded Imaging MM set 1",
            "imaging",
            1000,
            hi["windows"],
        ),

        summarize_dataset(
            "Ghost Imaging MM set 1",
            "imaging",
            1000,
            gi["windows"],
        ),
    ]

    drift_positions = [
        d["normalized_drift_position"]
        for d in datasets
        if d["normalized_drift_position"]
        is not None
    ]

    max_cosine_positions = [
        d["maximum_cosine_relative_position"]
        for d in datasets
    ]

    min_js_positions = [
        d["minimum_js_relative_position"]
        for d in datasets
    ]

    diffraction_drift = [
        d["normalized_drift_position"]
        for d in datasets
        if d["family"] == "diffraction"
        and d["normalized_drift_position"]
        is not None
    ]

    imaging_drift = [
        d["normalized_drift_position"]
        for d in datasets
        if d["family"] == "imaging"
        and d["normalized_drift_position"]
        is not None
    ]

    summary = {
        "n_datasets":
            len(datasets),

        "families": {
            "diffraction": 2,
            "imaging": 2,
        },

        "datasets":
            datasets,

        "all_normalized_drift_positions":
            drift_positions,

        "all_drift_position_stats":
            mean_std(
                drift_positions
            ),

        "diffraction_drift_position_stats":
            mean_std(
                diffraction_drift
            ),

        "imaging_drift_position_stats":
            mean_std(
                imaging_drift
            ),

        "maximum_cosine_relative_position_stats":
            mean_std(
                max_cosine_positions
            ),

        "minimum_js_relative_position_stats":
            mean_std(
                min_js_positions
            ),

        "interpretation": (
            "This experiment aggregates four real Glasgow "
            "acquisitions spanning diffraction and imaging. "
            "All datasets use the same distribution-level "
            "convergence diagnostics and unchanged state "
            "classification rule. Relative positions are "
            "descriptive acquisition coordinates rather than "
            "universal thresholds."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-023",

        "method":
            "four-dataset aggregate transfer analysis",

        "summary":
            summary,

        "caution": (
            "All four acquisitions originate from the same "
            "Glasgow experimental platform. Imaging sequences "
            "contain only 10 fixed windows each. Consistent "
            "relative timing supports cross-condition transfer "
            "but does not establish universal detector behavior."
        ),
    }

    outdir = Path(
        "experiments/qms_real_023/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_023_four_dataset_summary.json"
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
