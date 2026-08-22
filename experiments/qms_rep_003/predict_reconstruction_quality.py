from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qms_core.representation import effective_dimension


DATA = Path(
    "experiments/qms_qst_001/data/"
    "kwiat_2qubit_reference.json"
)

SCALES = [10.0, 3.0, 1.0, 0.3, 0.1]
TRIALS = 300
SEED = 42


def ket(label: str) -> np.ndarray:
    states = {
        "H": np.array([1, 0], dtype=complex),
        "V": np.array([0, 1], dtype=complex),
        "D": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "A": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "R": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "L": np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }
    return states[label]


def projector(basis) -> np.ndarray:
    psi = np.kron(
        ket(basis[0]),
        ket(basis[1]),
    )
    return np.outer(psi, psi.conj())


def context(basis):
    axis = {
        "H": "Z",
        "V": "Z",
        "D": "X",
        "A": "X",
        "R": "Y",
        "L": "Y",
    }
    return axis[basis[0]], axis[basis[1]]


def normalized_measurements(rows):
    totals = {}

    for row in rows:
        ctx = context(row["basis"])
        totals[ctx] = totals.get(ctx, 0.0) + row["count"]

    measurements = []

    for row in rows:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            return None

        measurements.append(
            (
                projector(row["basis"]),
                row["count"] / total,
            )
        )

    return measurements


def measurement_vector(measurements):
    return np.asarray(
        [p for _, p in measurements],
        dtype=float,
    )


def reconstruct(measurements):
    A = np.asarray(
        [
            m.T.reshape(-1)
            for m, _ in measurements
        ],
        dtype=complex,
    )

    b = np.asarray(
        [
            p
            for _, p in measurements
        ],
        dtype=complex,
    )

    rho_vec = np.linalg.pinv(A) @ b

    rho = rho_vec.reshape(4, 4)
    rho = (rho + rho.conj().T) / 2

    trace = np.trace(rho)

    if abs(trace) > 0:
        rho = rho / trace

    return rho


def bell_state():
    hh = np.kron(
        ket("H"),
        ket("H"),
    )
    vv = np.kron(
        ket("V"),
        ket("V"),
    )

    psi = (hh + vv) / np.sqrt(2)

    return psi


def fidelity_to_bell(rho):
    psi = bell_state()

    return float(
        np.real(
            psi.conj().T @ rho @ psi
        )
    )


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def main():
    raw = json.loads(DATA.read_text())
    reference_rows = raw["data"]

    ideal_rows = [
        {
            "basis": row["basis"],
            "count": float(row["counts"][-1]),
        }
        for row in reference_rows
    ]

    ideal_measurements = normalized_measurements(
        ideal_rows
    )
    ideal_vector = measurement_vector(
        ideal_measurements
    )

    rng = np.random.default_rng(SEED)

    all_vectors = []
    records = []

    for scale in SCALES:
        for trial in range(TRIALS):
            noisy_rows = []

            for row in reference_rows:
                original = float(
                    row["counts"][-1]
                )

                lam = max(
                    original * scale,
                    0.0,
                )

                noisy_rows.append(
                    {
                        "basis": row["basis"],
                        "count": int(
                            rng.poisson(lam)
                        ),
                    }
                )

            measurements = normalized_measurements(
                noisy_rows
            )

            if measurements is None:
                continue

            vector = measurement_vector(
                measurements
            )

            rho = reconstruct(
                measurements
            )

            similarity = cosine_similarity(
                vector,
                ideal_vector,
            )

            fidelity = fidelity_to_bell(
                rho
            )

            eigvals = np.linalg.eigvalsh(
                rho
            )

            record = {
                "count_scale": scale,
                "trial": trial,
                "similarity_to_ideal":
                    similarity,
                "fidelity":
                    fidelity,
                "min_eigenvalue":
                    float(np.min(eigvals)),
            }

            records.append(record)
            all_vectors.append(vector)

    X = np.vstack(all_vectors)

    overall_effective_dimension = (
        effective_dimension(X)
    )

    similarity = np.asarray(
        [
            r["similarity_to_ideal"]
            for r in records
        ]
    )

    fidelity = np.asarray(
        [
            r["fidelity"]
            for r in records
        ]
    )

    min_eig = np.asarray(
        [
            r["min_eigenvalue"]
            for r in records
        ]
    )

    sim_fid_corr = float(
        np.corrcoef(
            similarity,
            fidelity,
        )[0, 1]
    )

    sim_phys_corr = float(
        np.corrcoef(
            similarity,
            min_eig,
        )[0, 1]
    )

    summary = {
        "n_records": len(records),
        "overall_effective_dimension":
            overall_effective_dimension,
        "similarity_fidelity_correlation":
            sim_fid_corr,
        "similarity_min_eigenvalue_correlation":
            sim_phys_corr,
        "mean_similarity":
            float(np.mean(similarity)),
        "mean_fidelity":
            float(np.mean(fidelity)),
        "mean_min_eigenvalue":
            float(np.mean(min_eig)),
    }

    outdir = Path(
        "experiments/qms_rep_003/evidence"
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        outdir /
        "qms_rep_003_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    (
        outdir /
        "qms_rep_003_trials.json"
    ).write_text(
        json.dumps(
            records,
            indent=2,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        outdir /
        "qms_rep_003_summary.json",
    )


if __name__ == "__main__":
    main()
