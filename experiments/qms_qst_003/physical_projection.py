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
    psi = np.kron(ket(labels[0]), ket(labels[1]))
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
    hh = np.kron(ket("H"), ket("H"))
    vv = np.kron(ket("V"), ket("V"))
    psi = (hh + vv) / np.sqrt(2)
    return projector(psi)


def degraded_rows(
    rows: list[dict],
    background: float,
    rng: np.random.Generator,
) -> list[dict]:
    out = []

    for row in rows:
        ideal = float(row["counts"][-1])
        expected = ideal + background
        measured = int(rng.poisson(expected))

        out.append(
            {
                "basis": row["basis"],
                "count": measured,
            }
        )

    return out


def normalized_measurements(rows: list[dict]):
    totals = defaultdict(float)

    for row in rows:
        totals[context(row["basis"])] += row["count"]

    result = []

    for row in rows:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            return None

        result.append(
            (
                measurement_projector(row["basis"]),
                row["count"] / total,
            )
        )

    return result


def linear_reconstruct(measurements) -> np.ndarray:
    a = np.asarray(
        [m.T.reshape(-1) for m, _ in measurements],
        dtype=complex,
    )

    b = np.asarray(
        [p for _, p in measurements],
        dtype=complex,
    )

    rho_vec, _, _, _ = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    rho = rho_vec.reshape(4, 4)
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)

    return rho


def project_to_physical(rho: np.ndarray) -> np.ndarray:
    """
    Project a Hermitian trace-one matrix onto the PSD trace-one set.

    Steps:
    1. Hermitian symmetrization
    2. Eigen-decomposition
    3. Clip negative eigenvalues to zero
    4. Renormalize eigenvalues to sum to one
    """
    rho_h = (rho + rho.conj().T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(rho_h)

    clipped = np.clip(eigenvalues, 0.0, None)

    total = np.sum(clipped)

    if total <= 0:
        raise ValueError("Physical projection failed: zero PSD trace")

    clipped = clipped / total

    rho_physical = (
        eigenvectors
        @ np.diag(clipped)
        @ eigenvectors.conj().T
    )

    rho_physical = (
        rho_physical + rho_physical.conj().T
    ) / 2

    rho_physical = rho_physical / np.trace(rho_physical)

    return rho_physical


def metrics(
    rho: np.ndarray,
    reference: np.ndarray,
) -> dict:
    eig = np.linalg.eigvalsh(rho)

    fidelity = float(
        np.real(
            np.trace(rho @ reference)
        )
    )

    purity = float(
        np.real(
            np.trace(rho @ rho)
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
        "min_eigenvalue": float(np.min(eig)),
        "error": error,
    }


def run_level(
    rows: list[dict],
    reference: np.ndarray,
    background: float,
    trials: int,
) -> dict:
    linear_results = []
    physical_results = []

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

        rho_physical = project_to_physical(
            rho_linear
        )

        linear_results.append(
            metrics(rho_linear, reference)
        )

        physical_results.append(
            metrics(rho_physical, reference)
        )

    if not linear_results:
        raise RuntimeError(
            f"No valid trials for background={background}"
        )

    def mean(results, key):
        return float(
            np.mean(
                [r[key] for r in results]
            )
        )

    return {
        "background": background,
        "linear_fidelity": mean(
            linear_results,
            "fidelity",
        ),
        "physical_fidelity": mean(
            physical_results,
            "fidelity",
        ),
        "linear_purity": mean(
            linear_results,
            "purity",
        ),
        "physical_purity": mean(
            physical_results,
            "purity",
        ),
        "linear_min_eig": mean(
            linear_results,
            "min_eigenvalue",
        ),
        "physical_min_eig": mean(
            physical_results,
            "min_eigenvalue",
        ),
        "linear_error": mean(
            linear_results,
            "error",
        ),
        "physical_error": mean(
            physical_results,
            "error",
        ),
    }


def main() -> None:
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
        "QMS-QST-003A — Physical-state projection"
    )
    print("trials per level:", trials)

    print()
    print(
        "bg     F_lin     F_phys    "
        "P_lin     P_phys    "
        "eig_lin    eig_phys   "
        "err_lin   err_phys"
    )
    print(
        "--------------------------------------------------------------------------"
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
            f"{result['physical_fidelity']:<10.6f}"
            f"{result['linear_purity']:<10.6f}"
            f"{result['physical_purity']:<10.6f}"
            f"{result['linear_min_eig']:<11.6f}"
            f"{result['physical_min_eig']:<11.6f}"
            f"{result['linear_error']:<10.6f}"
            f"{result['physical_error']:.6f}"
        )

    print()
    print("Physical projection constraints")
    print("-------------------------------")
    print("rho = rho^dagger")
    print("rho >= 0")
    print("Tr(rho) = 1")


if __name__ == "__main__":
    main()
