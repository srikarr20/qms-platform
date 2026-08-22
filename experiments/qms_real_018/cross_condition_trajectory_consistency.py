from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "effective_dimension",
    "first_pc_fraction",
    "cosine_to_final",
    "js_divergence_to_final",
]


def normalize_minmax(x):
    x = np.asarray(x, dtype=float)

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    if xmax == xmin:
        return np.zeros_like(x)

    return (
        (x - xmin)
        / (xmax - xmin)
    )


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def rmse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (x - y) ** 2
            )
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--heralded-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ghost-json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    heralded = json.loads(
        args.heralded_json.read_text()
    )

    ghost = json.loads(
        args.ghost_json.read_text()
    )

    h_rows = heralded["windows"]
    g_rows = ghost["windows"]

    n = min(
        len(h_rows),
        len(g_rows),
    )

    h_rows = h_rows[:n]
    g_rows = g_rows[:n]

    comparisons = {}

    for metric in METRICS:
        h = np.asarray(
            [
                row[metric]
                for row in h_rows
            ],
            dtype=float,
        )

        g = np.asarray(
            [
                row[metric]
                for row in g_rows
            ],
            dtype=float,
        )

        h_norm = normalize_minmax(h)
        g_norm = normalize_minmax(g)

        comparisons[metric] = {
            "raw_pearson":
                pearson(h, g),

            "normalized_pearson":
                pearson(
                    h_norm,
                    g_norm,
                ),

            "normalized_rmse":
                rmse(
                    h_norm,
                    g_norm,
                ),

            "heralded_start":
                float(h[0]),

            "heralded_end":
                float(h[-1]),

            "ghost_start":
                float(g[0]),

            "ghost_end":
                float(g[-1]),
        }

    h_cosine = np.asarray(
        [
            row["cosine_to_final"]
            for row in h_rows
        ],
        dtype=float,
    )

    g_cosine = np.asarray(
        [
            row["cosine_to_final"]
            for row in g_rows
        ],
        dtype=float,
    )

    h_js = np.asarray(
        [
            row["js_divergence_to_final"]
            for row in h_rows
        ],
        dtype=float,
    )

    g_js = np.asarray(
        [
            row["js_divergence_to_final"]
            for row in g_rows
        ],
        dtype=float,
    )

    h_best_cosine = int(
        np.argmax(h_cosine)
    )

    g_best_cosine = int(
        np.argmax(g_cosine)
    )

    h_best_js = int(
        np.argmin(h_js)
    )

    g_best_js = int(
        np.argmin(g_js)
    )

    summary = {
        "matched_windows":
            n,

        "metric_comparisons":
            comparisons,

        "best_region_comparison": {
            "heralded_max_cosine_window":
                h_best_cosine,

            "ghost_max_cosine_window":
                g_best_cosine,

            "max_cosine_window_difference":
                abs(
                    h_best_cosine
                    - g_best_cosine
                ),

            "heralded_min_js_window":
                h_best_js,

            "ghost_min_js_window":
                g_best_js,

            "min_js_window_difference":
                abs(
                    h_best_js
                    - g_best_js
                ),
        },

        "interpretation": (
            "High cross-condition correlations after "
            "normalization indicate reproducible temporal "
            "measurement-state evolution across independent "
            "Glasgow acquisition conditions."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-018",

        "dataset_a":
            heralded["dataset"],

        "dataset_b":
            ghost["dataset"],

        "normalization":
            "per-dataset min-max scaling",

        "summary":
            summary,

        "caution": (
            "These are related acquisitions from the same "
            "experimental platform. Strong consistency supports "
            "cross-condition transfer, not yet cross-laboratory "
            "or universal detector generality."
        ),
    }

    outdir = Path(
        "experiments/qms_real_018/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_018_cross_condition_consistency.json"
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
