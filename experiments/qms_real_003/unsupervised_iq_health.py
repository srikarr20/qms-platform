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


def combined_cloud(data):
    g = np.column_stack([
        vector(data, "I_g"),
        vector(data, "Q_g"),
    ])

    e = np.column_stack([
        vector(data, "I_e"),
        vector(data, "Q_e"),
    ])

    return np.vstack([g, e])


def scalar(data, key):
    return float(
        np.asarray(data[key]).ravel()[0]
    )


def covariance_metrics(X):
    centered = X - np.mean(X, axis=0)

    cov = np.cov(
        centered,
        rowvar=False,
    )

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    total = float(np.sum(eigvals))

    if total <= 0:
        anisotropy = 0.0
        first_fraction = 0.0
    else:
        first_fraction = float(
            eigvals[0] / total
        )

        anisotropy = float(
            (
                eigvals[0] - eigvals[-1]
            ) / total
        )

    return {
        "effective_dimension":
            effective_dimension(X),
        "first_pc_fraction":
            first_fraction,
        "anisotropy":
            anisotropy,
        "cov_trace":
            total,
        "cov_determinant":
            float(np.linalg.det(cov)),
    }


def radial_metrics(X):
    center = np.mean(X, axis=0)

    r = np.linalg.norm(
        X - center,
        axis=1,
    )

    return {
        "radial_mean":
            float(np.mean(r)),
        "radial_std":
            float(np.std(r)),
        "radial_p90":
            float(np.percentile(r, 90)),
        "radial_p99":
            float(np.percentile(r, 99)),
    }


def entropy_2d(X, bins=80):
    hist, _, _ = np.histogram2d(
        X[:, 0],
        X[:, 1],
        bins=bins,
    )

    p = hist.ravel().astype(float)
    p = p[p > 0]

    p /= np.sum(p)

    return float(
        -np.sum(
            p * np.log(p)
        )
    )


def pearson(x, y):
    return float(
        np.corrcoef(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )[0, 1]
    )


def rankdata(x):
    x = np.asarray(x, dtype=float)

    order = np.argsort(x)
    sorted_x = x[order]

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    i = 0

    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and sorted_x[j] == sorted_x[i]
        ):
            j += 1

        rank = 0.5 * (
            i + j - 1
        )

        ranks[
            order[i:j]
        ] = rank

        i = j

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

        X = combined_cloud(data)

        metrics = {
            **covariance_metrics(X),
            **radial_metrics(X),
            "entropy_2d":
                entropy_2d(X),
        }

        row = {
            "frequency_hz": hz,
            "n_samples": len(X),
            "QNDFid":
                scalar(data, "QNDFid"),
            **metrics,
        }

        rows.append(row)

        print()
        print(
            json.dumps(
                row,
                indent=2,
            )
        )

    qnd = np.asarray(
        [r["QNDFid"] for r in rows]
    )

    metric_names = [
        key
        for key in rows[0]
        if key not in {
            "frequency_hz",
            "n_samples",
            "QNDFid",
        }
    ]

    correlations = {}

    for metric in metric_names:
        values = np.asarray(
            [r[metric] for r in rows],
            dtype=float,
        )

        correlations[metric] = {
            "pearson_with_QNDFid":
                pearson(values, qnd),
            "spearman_with_QNDFid":
                spearman(values, qnd),
        }

    output = {
        "dataset":
            "ISTA All-Optical SCQ Readout Fig. 4a",
        "label_usage":
            "QMS diagnostics use combined IQ cloud only; g/e labels are not used in metric computation.",
        "experimental_conditions":
            len(rows),
        "conditions":
            rows,
        "correlations":
            correlations,
        "caution":
            (
                "Only six experimental operating conditions are available. "
                "QNDFid is used only as a validation target and correlations "
                "remain exploratory."
            ),
    }

    outdir = Path(
        "experiments/qms_real_003/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_003_results.json"
    )

    outfile.write_text(
        json.dumps(
            output,
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
