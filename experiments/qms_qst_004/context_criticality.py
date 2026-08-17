import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "qms_qst_001"
    / "data"
    / "kwiat_2qubit_reference.json"
)

BASIS_FAMILY = {
    "H": "Z",
    "V": "Z",
    "D": "X",
    "A": "X",
    "R": "Y",
    "L": "Y",
}


def ket(label):
    h = np.array([1.0, 0.0], dtype=complex)
    v = np.array([0.0, 1.0], dtype=complex)

    return {
        "H": h,
        "V": v,
        "D": (h + v) / np.sqrt(2),
        "A": (h - v) / np.sqrt(2),
        "R": (h + 1j * v) / np.sqrt(2),
        "L": (h - 1j * v) / np.sqrt(2),
    }[label]


def projector(state):
    return np.outer(state, state.conj())


def measurement_projector(labels):
    psi = np.kron(
        ket(labels[0]),
        ket(labels[1]),
    )
    return projector(psi)


def context(labels):
    return (
        BASIS_FAMILY[labels[0]]
        + BASIS_FAMILY[labels[1]]
    )


def load_rows():
    with DATA_PATH.open() as f:
        return json.load(f)["data"]


def bell_reference():
    hh = np.kron(
        ket("H"),
        ket("H"),
    )

    vv = np.kron(
        ket("V"),
        ket("V"),
    )

    psi = (hh + vv) / np.sqrt(2)

    return projector(psi)


def normalized_measurements(rows, retained):
    filtered = [
        row
        for row in rows
        if context(row["basis"]) in retained
    ]

    totals = defaultdict(float)

    for row in filtered:
        totals[
            context(row["basis"])
        ] += float(
            row["counts"][-1]
        )

    measurements = []

    for row in filtered:
        ctx = context(row["basis"])

        probability = (
            float(row["counts"][-1])
            / totals[ctx]
        )

        measurements.append(
            (
                measurement_projector(
                    row["basis"]
                ),
                probability,
            )
        )

    return measurements


def reconstruct(measurements):
    a = np.asarray(
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

    rho_vec, _, rank, singular = (
        np.linalg.lstsq(
            a,
            b,
            rcond=None,
        )
    )

    rho = rho_vec.reshape(4, 4)

    rho = (
        rho + rho.conj().T
    ) / 2

    tr = np.trace(rho)

    if abs(tr) > 1e-12:
        rho /= tr

    return rho, a, rank, singular


def null_space(a, tolerance=1e-12):
    _, singular, vh = np.linalg.svd(
        a,
        full_matrices=True,
    )

    rank = np.sum(
        singular > tolerance
    )

    return vh[rank:].conj().T


def main():
    rows = load_rows()
    reference = bell_reference()

    contexts = sorted(
        {
            context(row["basis"])
            for row in rows
        }
    )

    print(
        "QMS-QST-004B — Context criticality"
    )

    print()
    print(
        "removed  rank   nullity   fidelity   "
        "error      target_null_overlap"
    )

    print(
        "-------------------------------------------------------------"
    )

    results = []

    ref_vec = reference.reshape(-1)

    for removed in contexts:
        retained = set(contexts)
        retained.remove(removed)

        measurements = (
            normalized_measurements(
                rows,
                retained,
            )
        )

        rho, a, rank, _ = reconstruct(
            measurements
        )

        fidelity = float(
            np.real(
                np.trace(
                    rho @ reference
                )
            )
        )

        error = float(
            np.linalg.norm(
                rho - reference
            )
        )

        ns = null_space(a)

        nullity = ns.shape[1]

        if nullity > 0:
            projection = (
                ns
                @ (
                    ns.conj().T
                    @ ref_vec
                )
            )

            overlap = float(
                np.linalg.norm(
                    projection
                )
            )
        else:
            overlap = 0.0

        results.append(
            (
                removed,
                rank,
                nullity,
                fidelity,
                error,
                overlap,
            )
        )

    results.sort(
        key=lambda x: x[3]
    )

    for (
        removed,
        rank,
        nullity,
        fidelity,
        error,
        overlap,
    ) in results:

        print(
            f"{removed:<9}"
            f"{rank:<7}"
            f"{nullity:<10}"
            f"{fidelity:<11.6f}"
            f"{error:<11.6f}"
            f"{overlap:.6f}"
        )

    print()
    print(
        "Lowest fidelity = most critical "
        "single context for this target state."
    )


if __name__ == "__main__":
    main()
