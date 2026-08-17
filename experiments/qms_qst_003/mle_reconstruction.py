import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


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
    hh = np.kron(ket("H"), ket("H"))
    vv = np.kron(ket("V"), ket("V"))
    psi = (hh + vv) / np.sqrt(2)
    return projector(psi)


def degraded_rows(
    rows: list[dict],
    background: float,
    rng: np.random.Generator,
) -> list[dict]:
    result = []

    for row in rows:
        ideal = float(row["counts"][-1])
        expected = ideal + background
        measured = int(rng.poisson(expected))

        result.append(
            {
                "basis": row["basis"],
                "count": measured,
            }
        )

    return result


def grouped_measurements(rows: list[dict]):
    grouped = defaultdict(list)

    for row in rows:
        basis = row["basis"]

        grouped[context(basis)].append(
            {
                "projector": measurement_projector(basis),
                "count": float(row["count"]),
                "basis": basis,
            }
        )

    return dict(grouped)


def params_to_rho(params: np.ndarray) -> np.ndarray:
    """
    Cholesky-like parameterization:

        rho = T T^dagger / Tr(T T^dagger)

    16 real parameters describe a general 4x4 complex
    lower-triangular matrix T.
    """
    if len(params) != 16:
        raise ValueError("Expected 16 MLE parameters")

    t = np.zeros((4, 4), dtype=complex)

    # Real diagonal
    t[0, 0] = params[0]
    t[1, 1] = params[1]
    t[2, 2] = params[2]
    t[3, 3] = params[3]

    k = 4

    # Complex lower-triangular entries
    for i in range(1, 4):
        for j in range(i):
            t[i, j] = (
                params[k]
                + 1j * params[k + 1]
            )
            k += 2

    rho = t @ t.conj().T

    tr = np.real(np.trace(rho))

    if tr <= 0:
        return np.eye(4, dtype=complex) / 4.0

    return rho / tr


def rho_to_initial_params(rho: np.ndarray) -> np.ndarray:
    """
    Build a stable initial point from a physical density matrix.

    Add a tiny diagonal regularizer before Cholesky factorization.
    """
    rho = (rho + rho.conj().T) / 2

    eigvals, eigvecs = np.linalg.eigh(rho)

    eigvals = np.clip(
        eigvals,
        1e-10,
        None,
    )

    eigvals /= eigvals.sum()

    rho_reg = (
        eigvecs
        @ np.diag(eigvals)
        @ eigvecs.conj().T
    )

    t = np.linalg.cholesky(rho_reg)

    params = [
        float(np.real(t[0, 0])),
        float(np.real(t[1, 1])),
        float(np.real(t[2, 2])),
        float(np.real(t[3, 3])),
    ]

    for i in range(1, 4):
        for j in range(i):
            params.append(
                float(np.real(t[i, j]))
            )
            params.append(
                float(np.imag(t[i, j]))
            )

    return np.asarray(params, dtype=float)


def negative_log_likelihood(
    params: np.ndarray,
    grouped,
) -> float:
    rho = params_to_rho(params)

    nll = 0.0
    eps = 1e-12

    for _, outcomes in grouped.items():
        probabilities = []

        for item in outcomes:
            p = float(
                np.real(
                    np.trace(
                        rho @ item["projector"]
                    )
                )
            )

            probabilities.append(
                max(p, eps)
            )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        # Numerical safety: each context is a complete
        # four-outcome measurement and should sum to 1.
        probabilities /= probabilities.sum()

        for item, p in zip(
            outcomes,
            probabilities,
        ):
            count = item["count"]

            if count > 0:
                nll -= count * np.log(p)

    return float(nll)


def linear_reconstruct(rows: list[dict]) -> np.ndarray:
    totals = defaultdict(float)

    for row in rows:
        totals[
            context(row["basis"])
        ] += row["count"]

    measurements = []

    for row in rows:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            raise ValueError(
                f"Zero-count context: {ctx}"
            )

        measurements.append(
            (
                measurement_projector(
                    row["basis"]
                ),
                row["count"] / total,
            )
        )

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
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)

    return rho


def project_vector_to_simplex(
    values: np.ndarray,
) -> np.ndarray:
    values = np.real(values)

    u = np.sort(values)[::-1]
    cssv = np.cumsum(u) - 1.0
    indices = np.arange(1, len(values) + 1)

    condition = (
        u - cssv / indices
    ) > 0

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

    return projected / projected.sum()


def exact_physical_projection(
    rho: np.ndarray,
) -> np.ndarray:
    rho_h = (
        rho + rho.conj().T
    ) / 2

    eigvals, eigvecs = np.linalg.eigh(
        rho_h
    )

    eigvals = project_vector_to_simplex(
        eigvals
    )

    rho_phys = (
        eigvecs
        @ np.diag(eigvals)
        @ eigvecs.conj().T
    )

    return (
        rho_phys
        + rho_phys.conj().T
    ) / 2


def mle_reconstruct(
    rows: list[dict],
    initial_rho: np.ndarray,
):
    grouped = grouped_measurements(rows)

    x0 = rho_to_initial_params(
        initial_rho
    )

    result = minimize(
        negative_log_likelihood,
        x0,
        args=(grouped,),
        method="L-BFGS-B",
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    rho = params_to_rho(
        result.x
    )

    return rho, result


def metrics(
    rho: np.ndarray,
    reference: np.ndarray,
) -> dict:
    eigvals = np.linalg.eigvalsh(rho)

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
            np.min(eigvals)
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
    mle_results = []

    successes = 0

    for seed in range(trials):
        rng = np.random.default_rng(seed)

        degraded = degraded_rows(
            rows,
            background,
            rng,
        )

        rho_linear = linear_reconstruct(
            degraded
        )

        rho_projected = (
            exact_physical_projection(
                rho_linear
            )
        )

        rho_mle, result = mle_reconstruct(
            degraded,
            rho_projected,
        )

        if result.success:
            successes += 1

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

        mle_results.append(
            metrics(
                rho_mle,
                reference,
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
        "mle_fidelity": mean(
            mle_results,
            "fidelity",
        ),
        "mle_purity": mean(
            mle_results,
            "purity",
        ),
        "mle_min_eig": mean(
            mle_results,
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
        "mle_error": mean(
            mle_results,
            "error",
        ),
        "successes": successes,
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

    # Start with 50 because MLE is much more expensive
    # than linear inversion. We can raise this later.
    trials = 50

    print(
        "QMS-QST-003C — Maximum-likelihood tomography"
    )

    print(
        "trials per level:",
        trials,
    )

    print()

    print(
        "bg     F_lin     F_proj    F_mle     "
        "P_mle     eig_mle    "
        "err_lin   err_proj   err_mle    success"
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
            f"{result['mle_fidelity']:<10.6f}"
            f"{result['mle_purity']:<10.6f}"
            f"{result['mle_min_eig']:<11.6f}"
            f"{result['linear_error']:<10.6f}"
            f"{result['projected_error']:<11.6f}"
            f"{result['mle_error']:<11.6f}"
            f"{result['successes']}/{trials}"
        )


if __name__ == "__main__":
    main()
