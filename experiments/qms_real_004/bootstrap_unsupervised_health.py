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
    return np.asarray(
        data[key],
        dtype=float,
    ).ravel()


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


def metrics(X):
    centered = X - np.mean(
        X,
        axis=0,
    )

    cov = np.cov(
        centered,
        rowvar=False,
    )

    eig = np.sort(
        np.linalg.eigvalsh(cov)
    )[::-1]

    total = float(np.sum(eig))

    first_pc = (
        float(eig[0] / total)
        if total > 0
        else 0.0
    )

    anisotropy = (
        float(
            (eig[0] - eig[-1])
            / total
        )
        if total > 0
        else 0.0
    )

    radius = np.linalg.norm(
        centered,
        axis=1,
    )

    return {
        "effective_dimension":
            effective_dimension(X),
        "first_pc_fraction":
            first_pc,
        "anisotropy":
            anisotropy,
        "cov_determinant":
            float(np.linalg.det(cov)),
        "radial_std":
            float(np.std(radius)),
        "radial_p99":
            float(
                np.percentile(
                    radius,
                    99,
                )
            ),
    }


def summarize(values):
    x = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(np.mean(x)),
        "std":
            float(np.std(x)),
        "ci95_low":
            float(
                np.percentile(
                    x,
                    2.5,
                )
            ),
        "ci95_high":
            float(
                np.percentile(
                    x,
                    97.5,
                )
            ),
    }


def scalar(data, key):
    return float(
        np.asarray(
            data[key]
        ).ravel()[0]
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

    rng = np.random.default_rng(
        SEED
    )

    results = []

    for hz in FREQUENCIES:
        path = (
            args.data_dir
            / f"IQblobs_{hz}Hz.mat"
        )

        data = loadmat(path)
        X = combined_cloud(data)

        observed = metrics(X)

        bootstrap = {
            key: []
            for key in observed
        }

        for _ in range(
            args.bootstraps
        ):
            idx = rng.integers(
                0,
                len(X),
                size=len(X),
            )

            sample = X[idx]

            result = metrics(
                sample
            )

            for key, value in result.items():
                bootstrap[key].append(
                    value
                )

        row = {
            "frequency_hz": hz,
            "n_samples": len(X),
            "QNDFid":
                scalar(
                    data,
                    "QNDFid",
                ),
            "metrics": {
                key: {
                    "observed":
                        observed[key],
                    **summarize(
                        bootstrap[key]
                    ),
                }
                for key in observed
            },
        }

        results.append(row)

        print()
        print(
            f"===== {hz} Hz ====="
        )

        print(
            json.dumps(
                row,
                indent=2,
            )
        )

    output = {
        "dataset":
            "ISTA All-Optical SCQ Readout Fig. 4a",
        "diagnostic_mode":
            "label-free combined IQ cloud",
        "bootstraps":
            args.bootstraps,
        "seed":
            SEED,
        "conditions":
            results,
    }

    outdir = Path(
        "experiments/qms_real_004/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_004_bootstrap.json"
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
