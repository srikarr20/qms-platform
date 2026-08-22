from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qms_core.representation import (
    effective_dimension,
    explained_variance_fraction,
)


DATA = Path(
    "experiments/qms_qst_001/data/"
    "kwiat_2qubit_reference.json"
)

SCALES = [10.0, 3.0, 1.0, 0.3, 0.1]
TRIALS = 500
SEED = 42


def measurement_context(basis):
    axis = {
        "H": "Z",
        "V": "Z",
        "D": "X",
        "A": "X",
        "R": "Y",
        "L": "Y",
    }
    return axis[basis[0]], axis[basis[1]]


def normalized_vector(rows):
    totals = {}

    for row in rows:
        ctx = measurement_context(row["basis"])
        totals[ctx] = totals.get(ctx, 0.0) + row["count"]

    values = []

    for row in rows:
        ctx = measurement_context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            return None

        values.append(row["count"] / total)

    return np.asarray(values, dtype=float)


def ideal_vector(rows):
    converted = [
        {
            "basis": row["basis"],
            "count": float(row["counts"][-1]),
        }
        for row in rows
    ]
    return normalized_vector(converted)


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def main():
    raw = json.loads(DATA.read_text())
    rows = raw["data"]

    ideal = ideal_vector(rows)

    rng = np.random.default_rng(SEED)

    results = []

    for scale in SCALES:
        samples = []
        similarities = []

        for _ in range(TRIALS):
            noisy_rows = []

            for row in rows:
                original = float(row["counts"][-1])
                lam = max(original * scale, 0.0)

                noisy_rows.append(
                    {
                        "basis": row["basis"],
                        "count": int(rng.poisson(lam)),
                    }
                )

            vector = normalized_vector(noisy_rows)

            if vector is None:
                continue

            samples.append(vector)
            similarities.append(
                cosine_similarity(vector, ideal)
            )

        X = np.vstack(samples)

        row = {
            "count_scale": scale,
            "valid_trials": len(samples),
            "effective_dimension":
                effective_dimension(X),
            "variance_explained_first_2":
                explained_variance_fraction(X, 2),
            "variance_explained_first_5":
                explained_variance_fraction(X, 5),
            "mean_similarity_to_ideal":
                float(np.mean(similarities)),
            "std_similarity_to_ideal":
                float(np.std(similarities)),
        }

        results.append(row)
        print(row)

    outdir = Path(
        "experiments/qms_rep_002/evidence"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = (
        outdir /
        "qms_rep_002_results.json"
    )

    outfile.write_text(
        json.dumps(results, indent=2)
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
