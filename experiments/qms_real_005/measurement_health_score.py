from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "effective_dimension": 1.0,
    "first_pc_fraction": -1.0,
    "radial_std": 1.0,
    "radial_p99": 1.0,
}


def pearson(x, y):
    return float(
        np.corrcoef(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )[0, 1]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    raw = json.loads(
        args.bootstrap_json.read_text()
    )

    conditions = raw["conditions"]

    reference = next(
        row
        for row in conditions
        if row["frequency_hz"] == 0
    )

    reference_stats = {}

    for metric in METRICS:
        m = reference["metrics"][metric]

        reference_stats[metric] = {
            "mean": float(m["mean"]),
            "std": float(m["std"]),
        }

    results = []

    for row in conditions:
        components = {}

        for metric, direction in METRICS.items():
            observed = float(
                row["metrics"][metric]["observed"]
            )

            ref_mean = (
                reference_stats[metric]["mean"]
            )

            ref_std = (
                reference_stats[metric]["std"]
            )

            if ref_std == 0:
                z = 0.0
            else:
                z = (
                    direction
                    * (observed - ref_mean)
                    / ref_std
                )

            components[metric] = float(z)

        degradation_index = float(
            np.mean(
                list(components.values())
            )
        )

        health_score = float(
            100.0
            * np.exp(
                -max(
                    degradation_index,
                    0.0,
                ) / 25.0
            )
        )

        result = {
            "frequency_hz":
                row["frequency_hz"],
            "QNDFid":
                row["QNDFid"],
            "component_z_scores":
                components,
            "degradation_index":
                degradation_index,
            "health_score":
                health_score,
        }

        results.append(result)

    qnd = [
        r["QNDFid"]
        for r in results
    ]

    health = [
        r["health_score"]
        for r in results
    ]

    degradation = [
        r["degradation_index"]
        for r in results
    ]

    output = {
        "definition": (
            "Label-free reference-relative measurement health score. "
            "Uses only raw-IQ geometry and 0 Hz bootstrap statistics. "
            "QNDFid is not used to construct the score."
        ),
        "reference_condition_hz": 0,
        "metrics": METRICS,
        "conditions": results,
        "validation": {
            "health_score_QNDFid_pearson":
                pearson(
                    health,
                    qnd,
                ),
            "degradation_QNDFid_pearson":
                pearson(
                    degradation,
                    qnd,
                ),
        },
        "caution": (
            "Score scaling is provisional and dataset-specific. "
            "This experiment tests whether a label-free reference-relative "
            "health index tracks experimental degradation; it does not "
            "establish a universal calibration."
        ),
    }

    outdir = Path(
        "experiments/qms_real_005/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_005_health_score.json"
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
