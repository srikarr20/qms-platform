from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def state_counts(rows):
    counts = {}

    for row in rows:
        state = row.get("state")

        if state is None:
            continue

        counts[state] = counts.get(state, 0) + 1

    return counts


def first_drift(rows):
    for row in rows:
        if row.get("state") == "drifting":
            return {
                "window_index": row["window_index"],
                "frame_start": row["frame_start"],
                "frame_end": row["frame_end"],
            }

    return None


def extract_dataset(
    name,
    raw_frames,
    rows,
):
    cosine = [
        float(r["cosine_to_final"])
        for r in rows
    ]

    js = [
        float(r["js_divergence_to_final"])
        for r in rows
    ]

    max_cosine_idx = max(
        range(len(cosine)),
        key=lambda i: cosine[i],
    )

    min_js_idx = min(
        range(len(js)),
        key=lambda i: js[i],
    )

    drift = first_drift(rows)

    if drift is not None:
        drift_midpoint = (
            drift["frame_start"]
            + drift["frame_end"]
        ) / 2.0

        normalized_drift_position = (
            drift_midpoint / raw_frames
        )

    else:
        normalized_drift_position = None

    return {
        "dataset":
            name,

        "raw_frames":
            raw_frames,

        "n_windows":
            len(rows),

        "cosine_start":
            cosine[0],

        "cosine_end":
            cosine[-1],

        "cosine_improvement":
            cosine[-1] - cosine[0],

        "js_start":
            js[0],

        "js_end":
            js[-1],

        "js_reduction":
            js[0] - js[-1],

        "maximum_cosine_window":
            rows[max_cosine_idx][
                "window_index"
            ],

        "minimum_js_window":
            rows[min_js_idx][
                "window_index"
            ],

        "first_drift":
            drift,

        "normalized_drift_position":
            normalized_drift_position,

        "state_counts":
            state_counts(rows),
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

    args = parser.parse_args()

    heralded_diffraction = load(
        args.heralded_diffraction
    )

    ghost_diffraction = load(
        args.ghost_diffraction
    )

    heralded_imaging = load(
        args.heralded_imaging
    )

    # REAL-008 contains state classification
    # for Heralded Diffraction SM.
    hd_rows = heralded_diffraction["windows"]

    # REAL-019 contains transferred state
    # classification for Ghost Diffraction SM.
    gd_rows = ghost_diffraction["windows"]

    # REAL-020 already contains the imaging
    # convergence/state output.
    hi_rows = heralded_imaging["windows"]

    datasets = [
        extract_dataset(
            "Heralded Diffraction SM",
            4070,
            hd_rows,
        ),

        extract_dataset(
            "Ghost Diffraction SM",
            4070,
            gd_rows,
        ),

        extract_dataset(
            "Heralded Imaging MM set 1",
            1000,
            hi_rows,
        ),
    ]

    drift_positions = [
        d["normalized_drift_position"]
        for d in datasets
        if d["normalized_drift_position"]
        is not None
    ]

    summary = {
        "n_datasets":
            len(datasets),

        "datasets":
            datasets,

        "normalized_drift_positions":
            drift_positions,

        "normalized_drift_position_min":
            (
                min(drift_positions)
                if drift_positions
                else None
            ),

        "normalized_drift_position_max":
            (
                max(drift_positions)
                if drift_positions
                else None
            ),

        "normalized_drift_position_mean":
            (
                sum(drift_positions)
                / len(drift_positions)
                if drift_positions
                else None
            ),

        "interpretation": (
            "This experiment aggregates fixed QMS "
            "distribution-level convergence and state "
            "classification results across three real "
            "Glasgow acquisition conditions. Relative "
            "drift position is reported as the midpoint "
            "of the first drifting window divided by "
            "total raw frame count."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-021",

        "method":
            "multi-dataset transfer summary",

        "summary":
            summary,

        "caution": (
            "The datasets are from the same experimental "
            "platform and have different acquisition lengths. "
            "The imaging dataset provides only 10 windows. "
            "Normalized drift timing is descriptive and "
            "should not be interpreted as a universal "
            "detector-failure threshold."
        ),
    }

    outdir = Path(
        "experiments/qms_real_021/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_021_multidataset_summary.json"
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
