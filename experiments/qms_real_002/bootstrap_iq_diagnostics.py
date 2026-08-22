from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from qms_core.representation import effective_dimension


FREQUENCIES = [0, 10, 50, 250, 500, 1000]
BOOTSTRAPS = 500
SEED = 42


def vector(data, key):
    return np.asarray(data[key], dtype=float).ravel()


def iq_cloud(data, state):
    return np.column_stack([
        vector(data, f"I_{state}"),
        vector(data, f"Q_{state}"),
    ])


def centroid_separation(g, e):
    return float(
        np.linalg.norm(
            np.mean(g, axis=0)
            - np.mean(e, axis=0)
        )
    )


def pooled_rms_spread(g, e):
    cg = g - np.mean(g, axis=0)
    ce = e - np.mean(e, axis=0)

    variance = (
        np.sum(cg * cg)
        + np.sum(ce * ce)
    ) / (len(g) + len(e))

    return float(np.sqrt(variance))


def normalized_separation(g, e):
    spread = pooled_rms_spread(g, e)

    if spread == 0:
        return 0.0

    return centroid_separation(g, e) / spread


def mahalanobis_separation(g, e):
    mu_g = np.mean(g, axis=0)
    mu_e = np.mean(e, axis=0)

    cov_g = np.cov(g, rowvar=False)
    cov_e = np.cov(e, rowvar=False)

    pooled = 0.5 * (cov_g + cov_e)

    delta = mu_g - mu_e

    value = (
        delta.T
        @ np.linalg.pinv(pooled)
        @ delta
    )

    return float(
        np.sqrt(max(value, 0.0))
    )


def compute_metrics(g, e):
    combined = np.vstack([g, e])

    return {
        "centroid_separation":
            centroid_separation(g, e),
        "pooled_rms_spread":
            pooled_rms_spread(g, e),
        "normalized_separation":
            normalized_separation(g, e),
        "mahalanobis_separation":
            mahalanobis_separation(g, e),
        "effective_dimension":
            effective_dimension(combined),
    }


def percentile_ci(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(np.mean(values)),
        "std":
            float(np.std(values)),
        "ci95_low":
            float(np.percentile(values, 2.5)),
        "ci95_high":
            float(np.percentile(values, 97.5)),
    }


def scalar(data, key):
    return float(
        np.asarray(data[key]).ravel()[0]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    results = []

    for hz in FREQUENCIES:
        path = (
            args.data_dir
            / f"IQblobs_{hz}Hz.mat"
        )

        data = loadmat(path)

        g = iq_cloud(data, "g")
        e = iq_cloud(data, "e")

        observed = compute_metrics(
            g,
            e,
        )

        bootstrap_values = {
            key: []
            for key in observed
        }

        for _ in range(args.bootstraps):
            idx_g = rng.integers(
                0,
                len(g),
                size=len(g),
            )

            idx_e = rng.integers(
                0,
                len(e),
                size=len(e),
            )

            sample_g = g[idx_g]
            sample_e = e[idx_e]

            metrics = compute_metrics(
                sample_g,
                sample_e,
            )

            for key, value in metrics.items():
                bootstrap_values[key].append(
                    value
                )

        summary = {
            key: {
                "observed":
                    observed[key],
                **percentile_ci(
                    bootstrap_values[key]
                ),
            }
            for key in observed
        }

        row = {
            "frequency_hz": hz,
            "n_ground": len(g),
            "n_excited": len(e),
            "QNDFid":
                scalar(data, "QNDFid"),
            "metrics": summary,
        }

        results.append(row)

        print()
        print(f"===== {hz} Hz =====")
        print(
            json.dumps(
                row,
                indent=2,
            )
        )

    output = {
        "dataset":
            "ISTA All-Optical SCQ Readout Fig. 4a",
        "bootstraps":
            args.bootstraps,
        "seed":
            SEED,
        "conditions":
            results,
        "interpretation_note":
            (
                "Bootstrap intervals quantify within-condition "
                "shot-level stability. They do not provide uncertainty "
                "on correlation across the six operating conditions."
            ),
    }

    outdir = Path(
        "experiments/qms_real_002/evidence"
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_002_bootstrap.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
