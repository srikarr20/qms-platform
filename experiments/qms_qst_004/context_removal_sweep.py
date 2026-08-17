import itertools
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


def ket(label: str) -> np.ndarray:
    h = np.array([1.0, 0.0], dtype=complex)
    v = np.array([0.0, 1.0], dtype=complex)

    states = {
        "H": h,
        "V": v,
        "D": (h + v) / np.sqrt(2),
        "A": (h - v) / np.sqrt(2),
        "R": (h + 1j * v) / np.sqrt(2),
        "L": (h - 1j * v) / np.sqrt(2),
    }

    return states[label]


def projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def measurement_projector(labels: list[str]) -> np.ndarray:
    psi = np.kron(
        ket(labels[0]),
        ket(labels[1]),
    )
    return projector(psi)


def context(labels: list[str]) -> str:
    return (
        BASIS_FAMILY[labels[0]]
        + BASIS_FAMILY[labels[1]]
    )


def load_rows() -> list[dict]:
    with DATA_PATH.open() as f:
        return json.load(f)["data"]


def bell_reference() -> np.ndarray:
    hh = np.kron(ket("H"), ket("H"))
    vv = np.kron(ket("V"), ket("V"))
    psi = (hh + vv) / np.sqrt(2)
    return projector(psi)


def normalize_subset(rows: list[dict], retained: set[str]):
    filtered = [
        row
        for row in rows
        if context(row["basis"]) in retained
    ]

    totals = defaultdict(float)

    for row in filtered:
        totals[context(row["basis"])] += float(
            row["counts"][-1]
        )

    result = []

    for row in filtered:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            continue

        result.append(
            (
                measurement_projector(row["basis"]),
                float(row["counts"][-1]) / total,
            )
        )

    return result


def reconstruct_linear(measurements):
    a = np.asarray(
        [m.T.reshape(-1) for m, _ in measurements],
        dtype=complex,
    )

    b = np.asarray(
        [p for _, p in measurements],
        dtype=complex,
    )

    rho_vec, _, rank, _ = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    rho = rho_vec.reshape(4, 4)
    rho = (rho + rho.conj().T) / 2

    trace = np.trace(rho)

    if abs(trace) > 1e-12:
        rho = rho / trace

    singular_values = np.linalg.svd(
        a,
        compute_uv=False,
    )

    nonzero = singular_values[
        singular_values > 1e-12
    ]

    if len(nonzero) == 0:
        condition = np.inf
    else:
        condition = float(
            nonzero.max() / nonzero.min()
        )

    return rho, rank, condition


def metrics(rho, reference):
    fidelity = float(
        np.real(
            np.trace(rho @ reference)
        )
    )

    error = float(
        np.linalg.norm(
            rho - reference
        )
    )

    return fidelity, error


def main():
    rows = load_rows()
    reference = bell_reference()

    contexts = sorted(
        {
            context(row["basis"])
            for row in rows
        }
    )

    rng = np.random.default_rng(12345)

    print(
        "QMS-QST-004A — Measurement-context removal"
    )
    print("available contexts:", contexts)
    print()

    print(
        "k   subsets   rank_mean   rank_min   "
        "F_mean     F_std      error_mean"
    )
    print(
        "----------------------------------------------------------"
    )

    for k in range(9, 0, -1):
        all_subsets = list(
            itertools.combinations(
                contexts,
                k,
            )
        )

        # Exhaustive when small enough; otherwise sample 100.
        if len(all_subsets) > 100:
            indices = rng.choice(
                len(all_subsets),
                size=100,
                replace=False,
            )

            subsets = [
                all_subsets[i]
                for i in indices
            ]
        else:
            subsets = all_subsets

        ranks = []
        fidelities = []
        errors = []

        for subset in subsets:
            retained = set(subset)

            measurements = normalize_subset(
                rows,
                retained,
            )

            rho, rank, condition = (
                reconstruct_linear(
                    measurements
                )
            )

            fidelity, error = metrics(
                rho,
                reference,
            )

            ranks.append(rank)
            fidelities.append(fidelity)
            errors.append(error)

        print(
            f"{k:<4}"
            f"{len(subsets):<10}"
            f"{np.mean(ranks):<12.3f}"
            f"{np.min(ranks):<11d}"
            f"{np.mean(fidelities):<11.6f}"
            f"{np.std(fidelities):<11.6f}"
            f"{np.mean(errors):.6f}"
        )


if __name__ == "__main__":
    main()
