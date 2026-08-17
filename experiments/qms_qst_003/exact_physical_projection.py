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


def context(labels: list[str]) -> tuple[str, str]:
    return (
        BASIS_FAMILY[labels[0]],
        BASIS_FAMILY[labels[1]],
    )


def load_rows() -> list[dict]:
    with DATA_PATH.open() as f:
        return json.load(f)["data"]


def bell_reference() -> np.ndarray:
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


def degraded_rows(
    rows: list[dict],
    background: float,
    rng: np.random.Generator,
) -> list[dict]:

    output = []

    for row in rows:
        ideal = float(row["counts"][-1])

        expected = ideal + background

        measured = int(
            rng.poisson(expected)
        )

        output.append(
            {
                "basis": row["basis"],
                "count": measured,
            }
        )

    return output


def normalized_measurements(rows: list[dict]):
    totals = defaultdict(float)

    for row in rows:
        totals[
            context(row["basis"])
        ] += row["count"]

    output = []

    for row in rows:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            return None

        output.append(
            (
                measurement_projector(
                    row["basis"]
                ),
                row["count"] / total,
            )
        )

    return output


def linear_reconstruct(measurements) -> np.ndarray:
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

    rho_vec, _, _, _ = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    rho = rho_vec.reshape(4, 4)

    rho = (
        rho + rho.conj().T
    ) / 2

    rho = rho / np.trace(rho)

    return rho


def project_vector_to_simplex(
    values: np.ndarray,
) -> np.ndarray:
    """
    Euclidean projection of a real vector onto

        x_i >= 0
        sum_i x_i = 1

    using the standard sorting / threshold algorithm.
    """
    values = np.real(values)

    u = np.sort(values)[::-1]

    cssv = np.cumsum(u) - 1.0

    indices = np.arange(
        1,
        len(values) + 1,
    )

    condition = (
        u - cssv / indices
    ) > 0

    if not np.any(condition):
        raise ValueError(
            "Simplex projection failed"
        )

    rho_index = np.nonzero(
        condition
    )[0][-1]

    theta = (
        cssv[rho_index]
        / float(rho_index + 1)
    )

    projected = np.maximum(
        values - theta,
        0.0,
    )

    projected /= projected.sum()

    return projected


def exact_physical_projection(
    rho: np.ndarray,
) -> np.ndarray:
    """
    Frobenius-nearest PSD, trace-one matrix.

    Because rho is Hermitian, the projection keeps the
    eigenvectors and projects the eigenvalues onto the
    probability simplex.
    """
    rho_h = (
        rho + rho.conj().T
    ) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(
        rho_h
    )

    projected_eigenvalues = (
        project_vector_to_simplex(
            eigenvalues
        )
    )

    rho_projected = (
        eigenvectors
        @ np.diag(projected_eigenvalues)
        @ eigenvectors.conj().T
    )

    rho_projected = (
        rho_projected
        + rho_projected.conj().T
    ) / 2

    return rho_projected


def metrics(
    rho: np.ndarray,
    reference: np.ndarray,
) -> dict:

    eig = np.linalg.eigvalsh(rho)

    fidelity = float(
        np.real(
            np.trace(
                rho @ reference
            )
        )
    )

    purity = float(
        np.real(
            np.trace(
                rho @ rho
            )
        )
    )

    error = float(
        np.linalg.norm(
            rho - reference
        )
    )

    return {
        "fidelity": fidelity,
        "purity": purity,
        "min_eigenvalue": float(
            np.min(eig)
        ),
        "error": error,
    }


def run_level(
    rows,
    reference,
    background,
    trials,
):
    linear_results = []
    projected_results = []

    projection_distances = []

    for seed in range(trials):
        rng = np.random.default_rng(seed)

        degraded = degraded_rows(
            rows,
            background,
            rng,
        )

        measurements = normalized_measurements(
            degraded
        )

        if measurements is None:
            continue

        rho_linear = linear_reconstruct(
            measurements
        )

        rho_projected = exact_physical_projection(
            rho_linear
        )

        linear_results.append(
            metrics(
                rho_linear,
                reference,
            )
        )

        projected_results.append(
            metrics(
                rho_projected,
                reference,
            )
        )

        projection_distances.append(
            float(
                np.linalg.norm(
                    rho_projected
                    - rho_linear
                )
            )
        )

    def mean(results, key):
        return float(
            np.mean(
                [r[key] for r in results]
            )
        )

    return {
        "linear_fidelity": mean(
            linear_results,
            "fidelity",
        ),
        "projected_fidelity": mean(
            projected_results,
            "fidelity",
        ),
        "linear_purity": mean(
            linear_results,
            "purity",
        ),
        "projected_purity": mean(
            projected_results,
            "purity",
        ),
        "linear_min_eig": mean(
            linear_results,
            "min_eigenvalue",
        ),
        "projected_min_eig": mean(
            projected_results,
            "min_eigenvalue",
        ),
        "linear_error": mean(
            linear_results,
            "error",
        ),
        "projected_error": mean(
            projected_results,
            "error",
        ),
        "projection_distance": float(
            np.mean(
                projection_distances
            )
        ),
    }


def main():
    rows = load_rows()
    reference = bell_reference()

    backgrounds = [
        0.0,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
    ]

    trials = 200

    print(
        "QMS-QST-003B — Exact physical-state projection"
    )

    print(
        "trials per level:",
        trials,
    )

    print()

    print(
        "bg     F_lin     F_proj    "
        "P_lin     P_proj    "
        "eig_lin    eig_proj   "
        "err_lin   err_proj   proj_dist"
    )

    print(
        "--------------------------------------------------------------------------------------"
    )

    for background in backgrounds:
        result = run_level(
            rows,
            reference,
            background,
            trials,
        )

        print(
            f"{background:<7.1f}"
            f"{result['linear_fidelity']:<10.6f}"
            f"{result['projected_fidelity']:<10.6f}"
            f"{result['linear_purity']:<10.6f}"
            f"{result['projected_purity']:<10.6f}"
            f"{result['linear_min_eig']:<11.6f}"
            f"{result['projected_min_eig']:<11.6f}"
            f"{result['linear_error']:<10.6f}"
            f"{result['projected_error']:<11.6f}"
            f"{result['projection_distance']:.6f}"
        )


if __name__ == "__main__":
    main()
