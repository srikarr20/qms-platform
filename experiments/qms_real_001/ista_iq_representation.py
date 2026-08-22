from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from qms_core.representation import effective_dimension


FREQUENCIES = [0, 10, 50, 250, 500, 1000]


def vector(data, key):
    return np.asarray(data[key], dtype=float).ravel()


def iq_cloud(data, state):
    return np.column_stack(
        [
            vector(data, f"I_{state}"),
            vector(data, f"Q_{state}"),
        ]
    )


def centroid_separation(g, e):
    mu_g = np.mean(g, axis=0)
    mu_e = np.mean(e, axis=0)

    return float(
        np.linalg.norm(mu_g - mu_e)
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

    return float(np.sqrt(max(value, 0.0)))


def scalar(data, key):
    return float(
        np.asarray(data[key]).ravel()[0]
    )


def pearson(x, y):
    return float(
        np.corrcoef(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )[0, 1]
    )


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x)

    ranks = np.empty_like(
        order,
        dtype=float,
    )
    ranks[order] = np.arange(len(x))

    return ranks


def spearman(x, y):
    return pearson(
        rankdata(x),
        rankdata(y),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    rows = []

    for hz in FREQUENCIES:
        path = (
            args.data_dir
            / f"IQblobs_{hz}Hz.mat"
        )

        data = loadmat(path)

        g = iq_cloud(data, "g")
        e = iq_cloud(data, "e")

        combined = np.vstack([g, e])

        row = {
            "frequency_hz": hz,
            "n_ground": len(g),
            "n_excited": len(e),
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
            "Pgg": scalar(data, "Pgg"),
            "Pee": scalar(data, "Pee"),
            "QNDFid": scalar(data, "QNDFid"),
        }

        rows.append(row)

        print(
            json.dumps(
                row,
                indent=2,
            )
        )

    qnd = [r["QNDFid"] for r in rows]

    metrics = [
        "centroid_separation",
        "pooled_rms_spread",
        "normalized_separation",
        "mahalanobis_separation",
        "effective_dimension",
    ]

    correlations = {}

    for metric in metrics:
        values = [
            r[metric]
            for r in rows
        ]

        correlations[metric] = {
            "pearson_with_QNDFid":
                pearson(values, qnd),
            "spearman_with_QNDFid":
                spearman(values, qnd),
        }

    result = {
        "dataset":
            "ISTA All-Optical SCQ Readout Fig. 4a",
        "experimental_conditions":
            len(rows),
        "conditions": rows,
        "correlations": correlations,
        "caution":
            (
                "Only six experimental conditions are available. "
                "Correlations are exploratory and QNDFid is a "
                "derived validation target from the same experimental files."
            ),
    }

    outdir = Path(
        "experiments/qms_real_001/evidence"
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_001_results.json"
    )

    outfile.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("=== CORRELATIONS ===")

    print(
        json.dumps(
            correlations,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
