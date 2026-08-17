import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).parent / "data" / "kwiat_2qubit_reference.json"


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

    if label not in states:
        raise ValueError(f"Unknown basis state: {label}")

    return states[label]


def projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, np.conjugate(state))


def two_qubit_projector(labels: list[str]) -> np.ndarray:
    psi = np.kron(ket(labels[0]), ket(labels[1]))
    return projector(psi)


def measurement_context(labels: list[str]) -> tuple[str, str]:
    return BASIS_FAMILY[labels[0]], BASIS_FAMILY[labels[1]]


def load_dataset() -> dict:
    with DATA_PATH.open() as f:
        return json.load(f)


def compute_context_totals(rows: list[dict]) -> dict:
    totals = defaultdict(float)

    for row in rows:
        context = measurement_context(row["basis"])
        totals[context] += float(row["counts"][-1])

    return dict(totals)


def normalized_measurements(rows: list[dict]):
    totals = compute_context_totals(rows)

    result = []

    for row in rows:
        basis = row["basis"]
        count = float(row["counts"][-1])
        context = measurement_context(basis)

        probability = count / totals[context]
        measurement = two_qubit_projector(basis)

        result.append(
            {
                "basis": basis,
                "probability": probability,
                "measurement": measurement,
            }
        )

    return result


def reconstruct_density_matrix(measurements) -> np.ndarray:
    """
    Solve p_i = Tr(M_i rho) by linear least squares.

    Using:
        Tr(M rho) = vec(M)^* vec(rho)

    We solve an overdetermined complex linear system using all
    36 tomography measurements.
    """
    a_rows = []
    probabilities = []

    for item in measurements:
        m = item["measurement"]
        p = item["probability"]

        # For row-major flattening:
        #
        # Tr(M rho) = vec(M.T) dot vec(rho)
        #
        # No conjugation is inserted here because the trace identity
        # already accounts for matrix multiplication directly.
        a_rows.append(m.T.reshape(-1))
        probabilities.append(p)

    a = np.asarray(a_rows, dtype=complex)
    b = np.asarray(probabilities, dtype=complex)

    rho_vec, residuals, rank, singular_values = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    rho = rho_vec.reshape((4, 4))

    print()
    print("Linear system diagnostics")
    print("-------------------------")
    print("matrix shape:", a.shape)
    print("rank:", rank)
    print(
        "condition number:",
        f"{np.linalg.cond(a):.6e}",
    )

    if residuals.size:
        print(
            "least-squares residual:",
            f"{float(np.real(residuals[0])):.6e}",
        )
    else:
        reconstructed_b = a @ rho_vec
        residual = np.linalg.norm(reconstructed_b - b) ** 2

        print(
            "least-squares residual:",
            f"{residual:.6e}",
        )

    return rho


def bell_phi_plus_density() -> np.ndarray:
    hh = np.kron(ket("H"), ket("H"))
    vv = np.kron(ket("V"), ket("V"))

    psi = (hh + vv) / np.sqrt(2)

    return projector(psi)


def state_fidelity_pure(
    rho: np.ndarray,
    target_rho: np.ndarray,
) -> float:
    """
    Fidelity when target_rho is a pure state:

        F = Tr(rho target_rho)
    """
    value = np.trace(rho @ target_rho)
    return float(np.real(value))


def validate_density_matrix(rho: np.ndarray) -> None:
    trace = np.trace(rho)

    hermitian_error = np.linalg.norm(
        rho - rho.conj().T
    )

    eigenvalues = np.linalg.eigvalsh(
        (rho + rho.conj().T) / 2
    )

    purity = np.real(np.trace(rho @ rho))

    print()
    print("Density-matrix diagnostics")
    print("--------------------------")
    print("trace:", trace)
    print(
        "Hermitian error:",
        f"{hermitian_error:.6e}",
    )
    print(
        "minimum eigenvalue:",
        f"{np.min(eigenvalues):.6e}",
    )
    print(
        "purity:",
        f"{purity:.12f}",
    )
    print(
        "eigenvalues:",
        np.array2string(
            eigenvalues,
            precision=8,
            suppress_small=True,
        ),
    )


def print_density_matrix(
    label: str,
    rho: np.ndarray,
) -> None:
    print()
    print(label)
    print("-" * len(label))

    with np.printoptions(
        precision=6,
        suppress=True,
    ):
        print(rho)


def main() -> None:
    data = load_dataset()
    rows = data["data"]

    measurements = normalized_measurements(rows)

    print("QMS-QST-001D — Linear tomography reconstruction")
    print("qubits:", data["n_qubits"])
    print("measurements:", len(measurements))

    rho = reconstruct_density_matrix(measurements)

    # Numerical symmetrisation only.
    rho = (rho + rho.conj().T) / 2

    # Normalize trace to one.
    rho = rho / np.trace(rho)

    print_density_matrix(
        "Reconstructed density matrix",
        rho,
    )

    validate_density_matrix(rho)

    reference = bell_phi_plus_density()

    fidelity = state_fidelity_pure(
        rho,
        reference,
    )

    difference = np.linalg.norm(
        rho - reference
    )

    print()
    print("Reference comparison")
    print("--------------------")
    print(
        "Bell-state fidelity:",
        f"{fidelity:.12f}",
    )
    print(
        "Frobenius error:",
        f"{difference:.6e}",
    )

    tolerance = 1e-10

    passed = (
        abs(np.trace(rho) - 1.0) < tolerance
        and
        np.linalg.norm(
            rho - rho.conj().T
        ) < tolerance
        and
        fidelity > 1.0 - tolerance
    )

    print()
    print(
        "Linear tomography validation:",
        "PASS" if passed else "FAIL",
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
